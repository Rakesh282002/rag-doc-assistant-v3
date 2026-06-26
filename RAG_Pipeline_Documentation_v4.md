# RAG Document Assistant — v4 Pipeline Documentation

## Complete Pipeline: Start to End

---

## 1. Document Upload & Indexing

**Trigger:** User uploads PDF/DOCX/TXT → clicks "Process & Index"

| Step | Action | Details |
|------|--------|---------|
| 1 | Clear previous data | All old documents, FAISS index, chunks.pkl, and semantic cache are wiped |
| 2 | Save file | Uploaded file saved to `uploads/` with a UUID filename |
| 3 | Load document | `load_document()` reads PDF/DOCX/TXT into raw text |
| 4 | Split into chunks | `split_documents()` splits text into ~800 char chunks with 150 char overlap, adding metadata (section, company, role, etc.) |
| 5 | FAISS indexing | Chunks embedded using `BAAI/bge-base-en-v1.5` (768-dim) and stored in FAISS vector index |
| 6 | BM25 store | Raw chunks pickled to `chunks.pkl` for keyword search |
| 7 | Registry update | `documents.json` updated with doc ID, filename, chunk count |

**Key Design Decision:** Only ONE document is active at a time. Uploading a new document completely replaces the previous one (vector store + cache fully cleared first).

---

## 2. Query Processing

**Trigger:** User types a question in the chat interface

### Step 0a: Conversation-Aware Query Rewrite

- Checks if there's prior conversation AND the question contains context words (`next`, `previous`, `that`, `this`, `it`, `second`, `third`, `last`, etc.) or is ≤3 words
- If YES → LLM rewrites the question into a standalone broad query
- **Example:** `"next company?"` → `"What are all the companies worked at in chronological order? Specifically the second one."`
- The rewriter is instructed to NOT include specific previous answers in the query to avoid biasing retrieval

### Step 0b: Semantic Cache Check (skipped for follow-ups)

- If question was NOT rewritten: embed question, compare against cached Q&A pairs (cosine similarity)
- If similarity ≥ 0.55 (candidate floor): ask validator LLM "is this the same question?"
- Validator checks SCOPE (filters added/removed?) and INTENT (list vs count?)
- If YES → return cached answer instantly (no retrieval/LLM call)
- **"Not found" answers are never cached** to prevent stale negative responses

### Step 1: Semantic Search (FAISS)

- Expand query with topic keywords (e.g. `"skills"` → adds `"technical skills programming languages tools technologies expertise"`)
- FAISS similarity search → top 15 candidates

### Step 2: Keyword Search (BM25)

- BM25 retriever (term frequency-based) over all chunks → top 15 candidates
- Catches exact keyword matches that embedding search might miss

### Step 3: Merge & Deduplicate

- Combine FAISS + BM25 results
- Remove duplicates by comparing first 200 characters of content

### Step 4: Cross-Encoder Reranking + Section Affinity Boost

- Cross-encoder model (`cross-encoder/ms-marco-MiniLM-L-6-v2`) scores each (question, chunk) pair for relevance
- **Section affinity boost:** +2.5 score added when question topic aligns with chunk's section metadata (e.g. question about "experience" boosts experience chunks)
- Keep top chunks within 3.0 points of the best score (maximum 5 chunks)

### Step 5: LLM Generation (Google Gemini 3.1 Flash-Lite)

- Build context from top chunks with metadata labels (Company, Role, Period, Section)
- Include recent conversation history (last 3 Q&A pairs) if it's a follow-up
- **Prompt rules enforce:**
  - Answer ONLY from retrieved passages
  - Say "not mentioned" if info isn't there
  - Don't repeat previously given answers
  - Indicate "(currently working)" for present employment
  - List ALL items found (don't summarize)

### Step 6: Format Sources

- Attach chunk snippets + source file + page number for transparency
- Displayed in expandable "sources used" section in UI

### Step 7: Cache Result

- Store question embedding + question text + answer + sources in semantic cache
- Skip caching if answer is "not mentioned in the document"
- Cache persisted to `semantic_cache.pkl`

---

## 3. Conversation Memory

| Feature | Implementation |
|---------|---------------|
| Storage | All Q&A pairs stored in `st.session_state.messages` (Streamlit session) |
| Rewrite trigger | Follow-up questions with context words or ≤3 words |
| History window | Last 3 Q&A pairs (6 messages) sent to rewriter |
| LLM context | Conversation history also injected into RAG prompt |
| Cache bypass | Rewritten questions always skip cache (context-dependent) |
| Clear | "Clear Chat History" button resets conversation |

---

## 4. Architecture Components

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Frontend | Streamlit | Chat UI, file upload, document management |
| Embeddings | BAAI/bge-base-en-v1.5 | 768-dim dense vectors for semantic search |
| Vector Store | FAISS | Fast similarity search over embeddings |
| Keyword Search | BM25Retriever | Term-frequency based retrieval |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 | Precise relevance scoring |
| LLM | Google Gemini 3.1 Flash-Lite | Answer generation + cache validation + query rewriting |
| Cache | Custom semantic cache (cosine + LLM validator) | Avoid redundant LLM calls |
| Persistence | FAISS index + pickle files | Survive app restarts |

---

## 5. Configuration (config.py)

| Parameter | Value | Description |
|-----------|-------|-------------|
| EMBEDDING_MODEL | BAAI/bge-base-en-v1.5 | Embedding model (768-dim) |
| CROSS_ENCODER_MODEL | cross-encoder/ms-marco-MiniLM-L-6-v2 | Reranker model |
| LLM_MODEL | gemini-3.1-flash-lite | Generation model |
| LLM_TEMPERATURE | 0.1 | Low temperature for precise answers |
| CHUNK_SIZE | 800 | Characters per chunk |
| CHUNK_OVERLAP | 150 | Overlap between chunks |
| INITIAL_RETRIEVAL_K | 15 | Candidates from each retriever |
| MAX_RETRIEVAL_DOCS | 5 | Top chunks sent to LLM |
| SCORE_GAP | 3.0 | Max score distance from best chunk |
| CACHE_CANDIDATE_FLOOR | 0.55 | Min cosine sim to consider cache hit |
| CACHE_TTL_DAYS | 7 | Cache entry lifetime |
| CACHE_MAX_SIZE | 500 | Max cached entries (LRU eviction) |

---

## 6. Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    DOCUMENT UPLOAD FLOW                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  User uploads file                                              │
│       │                                                         │
│       ▼                                                         │
│  Clear old vector store + cache + previous files                │
│       │                                                         │
│       ▼                                                         │
│  Load document → Split into chunks (800 chars, 150 overlap)     │
│       │                                                         │
│       ▼                                                         │
│  Embed chunks → Store in FAISS index + chunks.pkl               │
│       │                                                         │
│       ▼                                                         │
│  Update documents.json registry                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                      QUERY FLOW                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  User asks question                                             │
│       │                                                         │
│       ▼                                                         │
│  Has prior conversation + context words?                        │
│       │                                                         │
│    YES ▼                          NO ▼                          │
│  LLM rewrites to                Check semantic                  │
│  standalone query               cache                           │
│       │                            │                            │
│       │                      HIT ▼     MISS ▼                   │
│       │                   Return     Continue                   │
│       │                   cached     to RAG                     │
│       │                   answer                                │
│       │                              │                          │
│       ▼──────────────────────────────▼                          │
│                                                                 │
│  FAISS semantic search (15 docs)                                │
│       +                                                         │
│  BM25 keyword search (15 docs)                                  │
│       │                                                         │
│       ▼                                                         │
│  Merge + deduplicate                                            │
│       │                                                         │
│       ▼                                                         │
│  Cross-encoder rerank + section affinity boost                  │
│       │                                                         │
│       ▼                                                         │
│  Top 5 chunks → Gemini LLM (with conversation context)          │
│       │                                                         │
│       ▼                                                         │
│  Return answer + sources → Cache result                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 7. Key v4 Improvements

1. **Single-document replacement** — Uploading a new doc fully clears the old one (no stale data)
2. **Conversation memory** — Follow-up questions understand context from prior Q&A
3. **Smart cache bypass** — Conversation-dependent questions skip cache to avoid wrong matches
4. **No caching of "not found"** — Prevents negative answers from being served to rephrased questions
5. **Current employment indicator** — Answers include "(currently working)" when applicable
6. **Broad rewrite strategy** — Follow-up queries rewritten to find ALL items, preventing retrieval bias
