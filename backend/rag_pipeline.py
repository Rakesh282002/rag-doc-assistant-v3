import os
import time
import pickle
import datetime

import numpy as np

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import CrossEncoder

from backend.config import (
    GOOGLE_API_KEY,
    VECTOR_STORE_DIR,
    EMBEDDING_MODEL,
    CROSS_ENCODER_MODEL,
    LLM_MODEL,
    LLM_TEMPERATURE,
    INITIAL_RETRIEVAL_K,
    MAX_RETRIEVAL_DOCS,
    SCORE_GAP,
)
from backend.semantic_cache import get_cache
from backend.location_detector import is_location_query, extract_location_from_query, should_augment_with_web_search, extract_location_from_answer, generate_maps_link

FAISS_INDEX_PATH = os.path.join(VECTOR_STORE_DIR, "faiss_index")
CHUNKS_PATH = os.path.join(VECTOR_STORE_DIR, "chunks.pkl")

RAG_PROMPT_TEMPLATE = """You are a precise document analyst. Answer the question using ONLY the retrieved passages below.

Today's date: {today}

Retrieved passages:
{context}

{conversation_context}Question: {question}

Rules:
1. Use ONLY information explicitly stated in the passages. Never add external knowledge.
2. If the answer is not in the passages, respond exactly: "This information is not mentioned in the document."
3. Be concise and direct. List items when the question asks "what are" or "what does he have".
4. Do NOT mention "chunks", "passages", or "documents" in your answer.
5. When multiple passages contribute, synthesize them into a single coherent answer.
6. For skills/certifications/experience, list ALL items found in the passages — do not summarize or omit.
7. You MAY perform simple arithmetic on explicitly stated values (e.g. counting years from date ranges). Use today's date to calculate durations for ongoing/present positions.
8. If conversation history is provided, use it to understand follow-up intent. Do NOT repeat an answer already given — provide the NEXT or DIFFERENT item requested.
9. When mentioning a company/role, if the passages indicate it is the current/present job (e.g. "Present", "Current", "till date", ongoing date range), clearly state "(currently working)" alongside it.

Answer:"""

VALIDATOR_PROMPT = """You are a strict relevance judge.

Cached question : "{cached_question}"
New question    : "{question}"

Ignore differences in phrasing, length, grammar, and wording.
Focus on TWO things only:

1. SCOPE — has a specific filter, condition, or entity been added or removed?
2. INTENT — does one question ask to LIST/DESCRIBE while the other asks to COUNT/CALCULATE/QUANTIFY?

Reply YES  — if both questions request the same information with the same scope AND the same intent.
Reply NO   — if the new question adds or removes a specific filter (e.g. a company name, date range, location).
Reply NO   — if the new question removes a filter making it broader than the cached answer covers.
Reply NO   — if one question asks to LIST or DESCRIBE and the other asks to COUNT, CALCULATE, QUANTIFY,
             or derive a number (e.g. "list experience" vs "how many years of experience?").

Examples:
  "what were his achievements?"   vs  "achievements?"                     → YES  (same scope, shorter phrasing)
  "what was his working experience?" vs "working experience?"              → YES  (same scope)
  "working experience?"           vs  "working experience at healthedge?" → NO   (company filter added)
  "working experience at healthedge?" vs "working experience?"            → NO   (cached is narrower)
  "list his experience"           vs  "how many years of experience?"     → NO   (list vs calculate)
  "what skills does he have?"     vs  "how many skills does he have?"     → NO   (describe vs count)

Reply with a single word — YES or NO. No explanation."""

# ---------------------------------------------------------------------------
# Conversation-aware query rewriter — resolves pronouns & follow-ups
# ---------------------------------------------------------------------------

REWRITE_PROMPT = """You are a query rewriter for a document search system.
Given the conversation history and a follow-up question, rewrite the follow-up into a standalone search query.

Conversation history:
{history}

Follow-up question: {question}

Rules:
1. Produce a BROAD search query that will find the relevant information in the document.
2. Do NOT include specific answers from previous turns in the rewritten query (e.g. don't say "after Henotic Technology" — say "all companies worked at" or "second company worked at").
3. For "next/second/third" questions, rewrite to ask for ALL items of that type so the document search finds everything, then note the ordinal. Example: "next company?" → "What are all the companies worked at in chronological order? Specifically the second one."
4. If the question is already standalone and doesn't reference conversation, return it unchanged.
5. Return ONLY the rewritten question — no explanation.

Rewritten question:"""


def _rewrite_with_history(question: str, chat_history: list) -> str:
    """
    Rewrite a follow-up question into a standalone query using recent chat history.
    Only invokes LLM if the question looks like it needs context.
    """
    # Need at least one prior assistant response for context
    has_prior_context = any(m["role"] == "assistant" for m in chat_history)
    if not has_prior_context:
        return question

    # Heuristic: skip rewrite for clearly standalone questions
    q_lower = question.lower().strip()
    context_words = {"next", "previous", "other", "another", "also", "else", "after",
                     "before", "same", "that", "this", "it", "its", "above", "second",
                     "third", "2nd", "3rd", "last"}
    needs_rewrite = any(w in q_lower.split() for w in context_words) or len(q_lower.split()) <= 3

    if not needs_rewrite:
        return question

    # Build compact history (last 3 turns max, exclude current question)
    history_msgs = [m for m in chat_history if not (m["role"] == "user" and m["content"] == question)]
    recent = history_msgs[-6:]  # last 3 Q&A pairs
    history_lines = []
    for msg in recent:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_lines.append(f"{role}: {msg['content'][:200]}")
    history_str = "\n".join(history_lines)

    try:
        chain = PromptTemplate.from_template(REWRITE_PROMPT) | _validator_llm() | StrOutputParser()
        rewritten = chain.invoke({"history": history_str, "question": question}).strip()
        if rewritten:
            print(f"[REWRITE] {question!r} → {rewritten!r}")
            return rewritten
    except Exception as exc:
        print(f"[REWRITE] failed ({exc}), using original question")

    return question


# ---------------------------------------------------------------------------
# Query expansion — map short/ambiguous queries to richer retrieval queries
# ---------------------------------------------------------------------------

_QUERY_EXPANSIONS: dict[str, list[str]] = {
    "name": ["full name", "candidate name"],
    "contact": ["phone number email address linkedin contact information"],
    "education": ["education degree university college bachelor masters"],
    "skills": ["technical skills programming languages tools technologies expertise"],
    "experience": ["work experience professional experience employment company role"],
    "certifications": ["certifications certificates licensed certified credentials"],
    "achievements": ["achievements awards recognition star performer accomplishments"],
    "projects": ["projects built developed implemented"],
    "summary": ["professional summary profile overview about"],
}


def _expand_query(question: str) -> str:
    """
    Expand short queries with relevant keywords to improve retrieval.
    Returns the expanded query for embedding search.
    """
    q_lower = question.lower().strip().rstrip("?").strip()
    for key, expansions in _QUERY_EXPANSIONS.items():
        if key in q_lower:
            return question + " " + " ".join(expansions)
    return question


# ---------------------------------------------------------------------------

_embeddings_cache: HuggingFaceEmbeddings | None = None
_cross_encoder_cache: CrossEncoder | None = None


def _embeddings() -> HuggingFaceEmbeddings:
    global _embeddings_cache
    if _embeddings_cache is None:
        _embeddings_cache = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings_cache


def _cross_encoder() -> CrossEncoder:
    global _cross_encoder_cache
    if _cross_encoder_cache is None:
        _cross_encoder_cache = CrossEncoder(CROSS_ENCODER_MODEL)
    return _cross_encoder_cache


def _llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=LLM_TEMPERATURE,
        streaming=True,          # Flash Live is optimised for streaming responses
        convert_system_message_to_human=True,
    )


def _validator_llm() -> ChatGoogleGenerativeAI:
    """Separate LLM instance for cache validation — temperature=0 for deterministic YES/NO."""
    return ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0,
        convert_system_message_to_human=True,
    )


def _validate_cache_candidate(question: str, cached_question: str) -> bool:
    """
    Ask the LLM whether the new question is asking for the same information
    as the cached question.  Returns True → use cache, False → run full RAG.
    Falls back to False on any error so the user always gets a fresh answer.
    """
    try:
        t = time.perf_counter()
        chain = PromptTemplate.from_template(VALIDATOR_PROMPT) | _validator_llm() | StrOutputParser()
        result = chain.invoke({"question": question, "cached_question": cached_question})
        verdict = result.strip().upper()
        print(f"[CACHE VALIDATOR] {time.perf_counter()-t:.2f}s  verdict={verdict!r}  "
              f"new={question!r}  cached={cached_question!r}")
        return verdict.startswith("YES")
    except Exception as exc:
        print(f"[CACHE VALIDATOR] failed ({exc}), treating as miss")
        return False


# ---------------------------------------------------------------------------
# Index helpers
# ---------------------------------------------------------------------------

def add_to_vector_store(chunks) -> None:
    """Embed chunks into FAISS and persist raw chunks for BM25."""
    embeddings = _embeddings()

    # --- FAISS (semantic) ---
    index_file = os.path.join(FAISS_INDEX_PATH, "index.faiss")
    if os.path.exists(index_file):
        store = FAISS.load_local(
            FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True
        )
        store.add_documents(chunks)
    else:
        os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
        store = FAISS.from_documents(chunks, embeddings)
    store.save_local(FAISS_INDEX_PATH)

    # --- Chunks pickle (BM25) ---
    existing: list = []
    if os.path.exists(CHUNKS_PATH):
        with open(CHUNKS_PATH, "rb") as f:
            existing = pickle.load(f)
    existing.extend(chunks)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(existing, f)


def _load_faiss():
    if not os.path.exists(FAISS_INDEX_PATH):
        return None
    return FAISS.load_local(
        FAISS_INDEX_PATH, _embeddings(), allow_dangerous_deserialization=True
    )


def _load_chunks() -> list:
    if not os.path.exists(CHUNKS_PATH):
        return []
    with open(CHUNKS_PATH, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Section-affinity boost — question keywords → preferred metadata section
# Prevents cross-encoder from picking an achievements/awards chunk when the
# user is clearly asking about work history, skills, education, etc.
# ---------------------------------------------------------------------------

_SECTION_AFFINITY: dict[str, set[str]] = {
    "experience": {
        "experience", "work", "worked", "working", "job", "jobs", "role", "roles",
        "position", "positions", "career", "employment", "employed", "company",
        "companies", "employer", "industry",
    },
    "education": {
        "education", "studied", "degree", "university", "college", "school",
        "bachelor", "masters", "phd", "diploma", "qualification", "academic",
    },
    "skills": {
        "skills", "skill", "technologies", "tech", "tools", "programming",
        "languages", "expertise", "proficiency", "stack",
    },
    "certifications": {
        "certification", "certifications", "certified", "certificate", "license",
    },
    "achievement": {
        "achievement", "achievements", "award", "awards", "recognition",
        "star", "performer", "accomplishment",
    },
}

_SECTION_BOOST = 2.5   # added to cross-encoder score when section matches


def _affinity_boost(question: str, doc) -> float:
    """Return a score boost when the question's topic aligns with the chunk's section."""
    q_words = set(question.lower().replace("?", " ").replace(",", " ").split())
    section = doc.metadata.get("section", "")
    for sec, keywords in _SECTION_AFFINITY.items():
        if q_words & keywords and sec == section:
            return _SECTION_BOOST
    return 0.0


# ---------------------------------------------------------------------------
# Context formatting — prepend metadata labels so LLM has structured context
# ---------------------------------------------------------------------------

def _format_chunk(doc) -> str:
    m = doc.metadata
    labels = []
    section = m.get("section", "")
    if section == "experience":
        if m.get("company"):
            labels.append(f"Company: {m['company']}")
        if m.get("role"):
            labels.append(f"Role: {m['role']}")
        if m.get("years"):
            labels.append(f"Period: {m['years']}")
    elif section == "skills":
        labels.append("Section: Skills")
        if m.get("skills"):
            labels.append(f"Skills listed: {m['skills']}")
    elif section == "education":
        labels.append("Section: Education")
    elif section == "certifications":
        labels.append("Section: Certifications")
    elif section == "summary":
        labels.append("Section: Professional Summary")
    elif section == "header":
        labels.append("Section: Personal Details / Header")
        if m.get("name"):
            labels.append(f"Name: {m['name']}")

    if labels:
        prefix = "[" + " | ".join(labels) + "]\n"
        return prefix + doc.page_content
    return doc.page_content


# ---------------------------------------------------------------------------
# Greeting / chitchat detection — skip RAG for non-document queries
# ---------------------------------------------------------------------------

_GREETINGS = {
    "hi", "hello", "hey", "hii", "hiii", "good morning", "good afternoon",
    "good evening", "howdy", "sup", "yo", "thanks", "thank you", "bye",
    "goodbye", "ok", "okay", "cool", "nice", "great",
}

_GREETING_RESPONSE = "Hello! Ask me anything about the uploaded document."


def _is_greeting(question: str) -> bool:
    q = question.lower().strip().rstrip("!.?").strip()
    return q in _GREETINGS


# ---------------------------------------------------------------------------
# Query — Hybrid retrieval + Cross-encoder reranking
# ---------------------------------------------------------------------------

def query_documents(question: str, chat_history: list | None = None) -> dict:
    t0 = time.perf_counter()

    # --- Quick exit for greetings/chitchat ---
    if _is_greeting(question):
        return {"answer": _GREETING_RESPONSE, "sources": [], "cached": False}

    # --- 0a. Rewrite follow-up questions using conversation history ---
    resolved_question = _rewrite_with_history(question, chat_history or [])
    was_rewritten = (resolved_question != question)

    # --- 0b. Semantic cache check (skip for conversation follow-ups) ---
    q_embedding = np.array(_embeddings().embed_query(resolved_question))
    if not was_rewritten:
        candidate = get_cache().get(q_embedding, resolved_question)
        if candidate:
            if _validate_cache_candidate(resolved_question, candidate["cached_question"]):
                candidate.pop("_needs_validation", None)
                print(f"[TIMING] cache+validation: {time.perf_counter()-t0:.2f}s  (HIT)")
                return candidate
            print(f"[TIMING] cache rejected by validator: {time.perf_counter()-t0:.2f}s  (→ full RAG)")
    else:
        print(f"[CACHE] Skipped — question was rewritten from conversation context")

    store = _load_faiss()
    all_chunks = _load_chunks()

    if store is None or not all_chunks:
        return {
            "answer": "No documents have been uploaded yet. Please upload a document first.",
            "sources": [],
        }
    print(f"[TIMING] load stores:      {time.perf_counter()-t0:.2f}s")

    # --- 1. Semantic search (FAISS) with query expansion ---
    t1 = time.perf_counter()
    expanded_q = _expand_query(resolved_question)
    semantic_docs = store.similarity_search(expanded_q, k=INITIAL_RETRIEVAL_K)
    print(f"[TIMING] semantic search:  {time.perf_counter()-t1:.2f}s  ({len(semantic_docs)} docs)"
          f"  expanded={expanded_q != resolved_question}")

    # --- 2. Keyword search (BM25) ---
    t2 = time.perf_counter()
    bm25 = BM25Retriever.from_documents(all_chunks, k=INITIAL_RETRIEVAL_K)
    bm25_docs = bm25.invoke(resolved_question)
    print(f"[TIMING] BM25 search:      {time.perf_counter()-t2:.2f}s  ({len(bm25_docs)} docs)")

    # --- 3. Merge & deduplicate ---
    seen_keys: set = set()
    candidates = []
    for doc in semantic_docs + bm25_docs:
        key = doc.page_content[:200]
        if key not in seen_keys:
            seen_keys.add(key)
            candidates.append(doc)
    print(f"[TIMING] candidates after dedup: {len(candidates)}")

    # --- 4. Cross-encoder reranking + section-affinity boost + adaptive relevance filter ---
    t3 = time.perf_counter()
    pairs = [(resolved_question, doc.page_content) for doc in candidates]
    scores = _cross_encoder().predict(pairs)
    boosted = [float(s) + _affinity_boost(resolved_question, doc) for s, doc in zip(scores, candidates)]
    ranked = sorted(zip(boosted, candidates), key=lambda x: x[0], reverse=True)

    best_score = float(ranked[0][0]) if ranked else -999
    print(f"[TIMING] reranking:        {time.perf_counter()-t3:.2f}s  best_score={best_score:.2f}")

    # Keep top chunks within SCORE_GAP points of the best score
    top_docs = [
        doc for score, doc in ranked[:MAX_RETRIEVAL_DOCS]
        if float(score) >= best_score - SCORE_GAP
    ]
    print(f"[RETRIEVAL] kept {len(top_docs)} chunk(s) "
          f"(scores: {[round(float(s),2) for s,_ in ranked[:len(top_docs)]]})")
    if not top_docs:
        top_docs = [ranked[0][1]]  # always send at least the best chunk 

    # --- 5. Build context and call LLM ---
    context = "\n\n---\n\n".join(_format_chunk(doc) for doc in top_docs)

    # Build conversation context for follow-up awareness
    conv_context = ""
    if chat_history:
        recent = chat_history[-6:]  # last 3 Q&A pairs
        conv_lines = []
        for msg in recent:
            role = "User" if msg["role"] == "user" else "Assistant"
            conv_lines.append(f"{role}: {msg['content'][:200]}")
        conv_context = "Recent conversation:\n" + "\n".join(conv_lines) + "\n\n"

    prompt = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
    chain = prompt | _llm() | StrOutputParser()

    t4 = time.perf_counter()
    last_exc = None
    for attempt in range(2):
        try:
            answer = chain.invoke({"context": context, "question": resolved_question, "conversation_context": conv_context, "today": datetime.date.today().strftime("%B %d, %Y")})
            break
        except Exception as exc:
            last_exc = exc
            if "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc):
                if attempt == 0:
                    time.sleep(5)
                else:
                    raise RuntimeError(
                        "Gemini quota exhausted. Wait a minute and try again."
                    ) from exc
            else:
                raise
    print(f"[TIMING] LLM generation:   {time.perf_counter()-t4:.2f}s")
    print(f"[TIMING] total:            {time.perf_counter()-t0:.2f}s")

    # --- 6. Format sources ---
    seen_src: set = set()
    sources = []
    for doc in top_docs:
        key = doc.page_content[:100]
        if key not in seen_src:
            seen_src.add(key)
            sources.append({
                "content": doc.page_content[:400] + ("..." if len(doc.page_content) > 400 else ""),
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", "N/A"),
            })

    result = {"answer": answer, "sources": sources, "cached": False}

    # --- 6b. Maps link augmentation for location queries ---
    try:
        # Append Google Maps link if a location is mentioned in the answer
        if is_location_query(resolved_question):
            current_answer = result["answer"]
            detected_location = extract_location_from_answer(current_answer)
            if detected_location:
                maps_link = generate_maps_link(detected_location)
                result["answer"] = f"{current_answer}\n\n{maps_link}"
                print(f"[MAPS] Appended Google Maps link for: {detected_location}")
    except Exception as e:
        print(f"[MAPS] Error during maps link augmentation: {str(e)}")
        result["maps_link_error"] = str(e)

    # --- 7. Store in semantic cache (skip "not found" answers) ---
    if "not mentioned in the document" not in answer.lower():
        get_cache().set(q_embedding, resolved_question, answer, sources)

    return result
