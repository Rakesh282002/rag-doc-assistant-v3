import os
import time
import pickle

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

FAISS_INDEX_PATH = os.path.join(VECTOR_STORE_DIR, "faiss_index")
CHUNKS_PATH = os.path.join(VECTOR_STORE_DIR, "chunks.pkl")

RAG_PROMPT_TEMPLATE = """You are a precise document analyst. Answer the question using ONLY the retrieved passages below.

Retrieved passages:
{context}

Question: {question}

Rules:
1. Use ONLY information explicitly stated in the passages. Never add external knowledge.
2. If the answer is not in the passages, respond exactly: "This information is not mentioned in the document."
3. Be concise and direct. List items when the question asks "what are" or "what does he have".
4. Do NOT mention "chunks", "passages", or "documents" in your answer.
5. When multiple passages contribute, synthesize them into a single coherent answer.
6. For skills/certifications/experience, list ALL items found in the passages — do not summarize or omit.
7. You MAY perform simple arithmetic on explicitly stated values (e.g. counting years from date ranges).

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
    if os.path.exists(FAISS_INDEX_PATH):
        store = FAISS.load_local(
            FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True
        )
        store.add_documents(chunks)
    else:
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
# Query — Hybrid retrieval + Cross-encoder reranking
# ---------------------------------------------------------------------------

def query_documents(question: str) -> dict:
    t0 = time.perf_counter()

    # --- 0. Semantic cache check ---
    q_embedding = np.array(_embeddings().embed_query(question))
    candidate = get_cache().get(q_embedding, question)
    if candidate:
        if _validate_cache_candidate(question, candidate["cached_question"]):
            candidate.pop("_needs_validation", None)
            print(f"[TIMING] cache+validation: {time.perf_counter()-t0:.2f}s  (HIT)")
            return candidate
        print(f"[TIMING] cache rejected by validator: {time.perf_counter()-t0:.2f}s  (→ full RAG)")

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
    expanded_q = _expand_query(question)
    semantic_docs = store.similarity_search(expanded_q, k=INITIAL_RETRIEVAL_K)
    print(f"[TIMING] semantic search:  {time.perf_counter()-t1:.2f}s  ({len(semantic_docs)} docs)"
          f"  expanded={expanded_q != question}")

    # --- 2. Keyword search (BM25) ---
    t2 = time.perf_counter()
    bm25 = BM25Retriever.from_documents(all_chunks, k=INITIAL_RETRIEVAL_K)
    bm25_docs = bm25.invoke(question)
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
    pairs = [(question, doc.page_content) for doc in candidates]
    scores = _cross_encoder().predict(pairs)
    boosted = [float(s) + _affinity_boost(question, doc) for s, doc in zip(scores, candidates)]
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

    prompt = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
    chain = prompt | _llm() | StrOutputParser()

    t4 = time.perf_counter()
    last_exc = None
    for attempt in range(2):
        try:
            answer = chain.invoke({"context": context, "question": question})
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

    # --- 7. Store in semantic cache ---
    get_cache().set(q_embedding, question, answer, sources)

    return result
