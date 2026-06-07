# AI Document Assistant — v3 (Gemini 3.1 Flash-Lite + Semantic Cache + Conversation Memory)

A local RAG (Retrieval-Augmented Generation) application that lets you upload a document and ask natural-language questions about it. This v3 edition uses **Gemini 3.1 Flash-Lite** (`gemini-3.1-flash-lite`) — a stable, cost-efficient multimodal model — and includes a **semantic cache** layer that avoids redundant LLM calls, plus **conversation memory** for follow-up questions. Answers are grounded strictly in the uploaded content — no hallucination.

### What's New in v3 (Latest)

- **Conversation memory** — follow-up questions like "next company?" are resolved using chat history via LLM query rewriting.
- **Single-document mode** — uploading a new document fully clears the previous one (vector store + cache), preventing stale data.
- **Smart cache bypass** — conversation-dependent questions skip the cache to avoid incorrect cached responses.
- **No caching of "not found"** — negative answers are never cached, preventing stale misses.
- **Current employment indicator** — responses include "(currently working)" when applicable.
- **LLM upgrade** — `gemini-3.1-flash-lite` for stability and cost efficiency.
- **Semantic caching** — repeated or similar questions are served from cache (cosine similarity + LLM validation).

### Performance Metrics

> Achieved 100% Recall@K and improved answer faithfulness by 14% using hybrid retrieval (FAISS + BM25) and cross-encoder reranking; reduced latency by 60% via semantic caching.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Streamlit UI  (:8501)                    │
│  Sidebar: upload / manage docs    Main: chat interface          │
└───────────────────────┬─────────────────────────────────────────┘
                        │  HTTP (REST)
┌───────────────────────▼─────────────────────────────────────────┐
│                     FastAPI Backend  (:8000)                     │
│  POST /upload   GET /documents   DELETE /documents/{id}         │
│  POST /ask                                                       │
└───────────┬──────────────────────────────┬──────────────────────┘
            │                              │
    ┌───────▼────────┐            ┌────────▼────────────┐
    │ document_       │            │  rag_pipeline.py    │
    │ processor.py    │            │                     │
    │                 │            │  1. Semantic cache   │
    │  Resume:        │            │     check (hit →    │
    │  section/role   │   chunks   │     return cached)  │
    │  chunking  ─────┼──────────► │  2. FAISS semantic  │
    │                 │            │     search          │
    │  Generic:       │            │  3. BM25 keyword    │
    │  Recursive      │            │     search          │
    │  splitter       │            │  4. Merge + dedup   │
    │                 │            │  5. Cross-encoder   │
    │                 │            │     reranking       │
    │                 │            │  6. Gemini LLM      │
    │                 │            │  7. Cache result    │
    └─────────────────┘            └─────────────────────┘
```

### Key Design Decisions

| Concern | Choice | Reason |
|---|---|---|
| Embeddings | `BAAI/bge-base-en-v1.5` (local) | Zero API calls, fast on CPU, 768-dim |
| LLM | `gemini-3.1-flash-lite` | Stable, cost-efficient multimodal model |
| Semantic cache | Cosine similarity + LLM validation | Avoids redundant API calls for repeated questions |
| Conversation memory | LLM-based query rewriter | Resolves follow-up references (next, previous, that, etc.) |
| Vector store | FAISS (local file) | No server needed, persists across restarts |
| Keyword search | BM25 | Catches exact-match queries that semantic search misses |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` (local) | Precise relevance scoring, no API calls |
| Document chunking | Section/role-aware (resumes) + recursive (generic) | Preserves logical boundaries |
| Document mode | Single-document replacement | Prevents stale data from old uploads |

---

## Project Structure

```
rag-doc-assistant-v3/
├── backend/
│   ├── __init__.py
│   ├── config.py               # All tunable settings in one place
│   ├── document_processor.py   # Loaders + smart chunking (resume-aware)
│   ├── rag_pipeline.py         # Retrieval, reranking, conversation memory, LLM chain
│   ├── semantic_cache.py       # Semantic caching layer (cosine + LLM validation)
│   └── main.py                 # FastAPI app + REST endpoints
├── frontend/
│   └── app.py                  # Streamlit chat UI (standalone mode)
├── streamlit_app.py            # Unified Streamlit app (for Streamlit Cloud deployment)
├── evaluation/
│   ├── evaluate.py             # RAG evaluation framework
│   ├── test_set.json           # Test questions + expected answers
│   └── results.json            # Evaluation results
├── uploads/                    # Uploaded files + documents.json registry
├── vector_store/               # FAISS index + chunks.pkl + semantic_cache.pkl
├── .env                        # API keys (not committed)
├── .env.example                # Template for .env
└── requirements.txt
```

---

## Quick Start

### Prerequisites
- Python 3.10+
- A Google AI Studio API key → [aistudio.google.com/apikey](https://aistudio.google.com/apikey)

### 1. Clone and install

```bash
git clone https://github.com/Rakesh282002/rag-doc-assistant-v3.git
cd rag-doc-assistant-v3
pip install -r requirements.txt
```

> **Note (Python 3.14 / Windows):** Some packages require pre-built wheels:
> ```bash
> pip install --only-binary=:all: numpy faiss-cpu
> ```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and set your key:
GOOGLE_API_KEY=your_key_here
```

### 3. Start the backend

```bash
uvicorn backend.main:app --reload
# Running at http://localhost:8000
# Interactive API docs at http://localhost:8000/docs
```

### 4. Start the frontend

```bash
streamlit run frontend/app.py
# Running at http://localhost:8501
```

---

## Configuration Reference

All settings live in `backend/config.py`:

| Setting | Default | Description |
|---|---|---|
| `EMBEDDING_MODEL` | `BAAI/bge-base-en-v1.5` | HuggingFace bi-encoder for FAISS indexing (768-dim) |
| `CROSS_ENCODER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranker model |
| `LLM_MODEL` | `gemini-3.1-flash-lite` | Gemini 3.1 Flash-Lite (stable, cost-efficient) |
| `LLM_TEMPERATURE` | `0.1` | Lower = more factual, less creative |
| `CHUNK_SIZE` | `800` | Max characters per chunk |
| `CHUNK_OVERLAP` | `150` | Overlap between consecutive chunks |
| `INITIAL_RETRIEVAL_K` | `15` | Candidates fetched from each retriever |
| `MAX_RETRIEVAL_DOCS` | `5` | Top chunks passed to LLM after reranking |
| `SCORE_GAP` | `3.0` | Drop chunks more than this many points below the best reranker score |
| `CACHE_CANDIDATE_FLOOR` | `0.55` | Min cosine similarity to trigger cache LLM validation |
| `CACHE_TTL_DAYS` | `7` | Days before a cache entry expires |
| `CACHE_MAX_SIZE` | `500` | Max cache entries (LRU eviction beyond this) |

---

## How It Works

### Upload flow

```
File upload
    │
    ▼
load_document()          ← PyPDFLoader / Docx2txtLoader / TextLoader
    │
    ▼
split_documents()        ← resume-aware or generic chunking
    │
    ▼
add_to_vector_store()
    ├── HuggingFace embed → FAISS.save_local()
    └── append raw chunks → chunks.pkl  (for BM25)
```

### Query flow

```
User question
    │
    ▼
Conversation rewrite             ← resolves follow-ups using chat history
    │ (standalone → unchanged)
    │ (follow-up → rewritten to standalone query)
    ▼
Semantic cache lookup            ← cosine sim + LLM validation (skipped for follow-ups)
    │ (hit → return cached answer)
    │ (miss ↓)
    ▼
FAISS similarity_search(k=15)   ← semantic candidates (with query expansion)
    +
BM25Retriever.invoke(k=15)      ← keyword candidates
    │
    ▼
Merge + deduplicate              ← by first 200 chars of content
    │
    ▼
CrossEncoder.predict()           ← score every candidate against question
    + section affinity boost     ← +2.5 when question topic matches chunk section
    │
    ▼
Keep top-5 within SCORE_GAP      ← drop outlier chunks
    │
    ▼
_format_chunk()                  ← prefix metadata labels (Company/Role/Period)
    │
    ▼
Gemini LLM (LCEL chain)          ← strict grounded-answer prompt + conversation context
    │
    ▼
Store in semantic cache           ← for future similar questions (skip "not found" answers)
    │
    ▼
{ answer, sources }
```

---

## Document Chunking

### Resume / CV documents (auto-detected)

The processor detects resumes by scanning for ≥2 of these signals in the first 3000 characters: `professional summary`, `professional experience`, `core competencies`, `technical skills`, `education`, `certifications`.

When detected, it applies a **3-stage strategy**:

1. **Section splitting** — regex finds standalone section headers (`PROFESSIONAL SUMMARY`, `CORE COMPETENCIES`, `TECHNICAL SKILLS`, `PROFESSIONAL EXPERIENCE`, `CERTIFICATIONS`, `EDUCATION`, etc.) and splits the document there.

2. **Role-level splitting** — within the experience section, each job entry becomes its own chunk. The splitter detects job boundaries by the pattern `Company | Role` or `Company – Role` followed by a date.

3. **Metadata enrichment** — every chunk carries structured metadata:
   - `section` — which resume section (experience, skills, education, certifications, header, etc.)
   - `company` — company name (for experience chunks)
   - `role` — job title (for experience chunks)
   - `years` — date range (for experience chunks)
   - `name` — candidate name (for header chunk)
   - `skills` — extracted skills list (for skills chunk)
   - `source` — path to the uploaded file

   This metadata is prepended as a label when the chunk is shown to the LLM:
   ```
   [Company: HealthEdge Software Pvt Ltd | Role: Software Engineer | Period: Jan 2024 – Present]
   <raw chunk text>
   ```

4. **Secondary split** — any section chunk still larger than `CHUNK_SIZE` is further split with `RecursiveCharacterTextSplitter`.

### Generic documents

Uses `RecursiveCharacterTextSplitter` with separators `["\n\n", "\n", ".", " ", ""]`, producing overlapping windows of `CHUNK_SIZE` characters.

---

## REST API

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `GET` | `/health` | Returns `{"status": "ok"}` |
| `POST` | `/upload` | Upload and index a document |
| `GET` | `/documents` | List all indexed documents |
| `DELETE` | `/documents/{id}` | Delete a document (file + registry) |
| `POST` | `/ask` | Ask a question, get a grounded answer |

### POST /upload

```
Content-Type: multipart/form-data
Body: file=<binary>

Response 200:
{
  "id": "ca85aead-...",
  "filename": "resume.pdf",
  "chunks": 10
}
```

### POST /ask

```json
Request:  { "question": "What is his name?" }

Response: {
  "answer": "His name is Nakka Rakesh.",
  "sources": [
    {
      "content": "Nakka Rakesh...",
      "source": "uploads/ca85aead-....pdf",
      "page": 0
    }
  ]
}
```

Interactive docs available at **http://localhost:8000/docs** when the backend is running.

---

## Supported File Types

| Extension | Loader |
|---|---|
| `.pdf` | `PyPDFLoader` (langchain-community) |
| `.docx` / `.doc` | `Docx2txtLoader` (langchain-community) |
| `.txt` | `TextLoader` (langchain-community) |

---

## Grounding & Hallucination Prevention

The LLM is given strict rules in its system prompt:

1. Answer **only** using information explicitly present in the retrieved passages.
2. If the answer is not there, respond with exactly: *"This information is not mentioned in the document."*
3. Do **not** use tangentially related facts as substitutes.
4. Do **not** reference "chunks", "passages", or source numbers.
5. Synthesize naturally across passages when the answer spans multiple chunks.
6. List ALL items found — do not summarize or omit.
7. If conversation history is provided, do NOT repeat previously given answers.
8. When mentioning a company/role, indicate "(currently working)" if the document shows it's the present job.

Additionally, the cross-encoder reranker with section affinity boost filters out irrelevant chunks, so the LLM only sees the most relevant context.

---

## Dependencies

```
fastapi, uvicorn          — REST API server
langchain-*               — LLM orchestration (LCEL pipeline)
langchain-google-genai    — Gemini LLM integration
langchain-huggingface     — Local HuggingFace embeddings
langchain-community       — FAISS vector store, BM25 retriever, document loaders
faiss-cpu                 — Local vector similarity search
sentence-transformers     — Cross-encoder reranker (ms-marco-MiniLM-L-6-v2)
rank-bm25                 — BM25 keyword retrieval
pypdf, python-docx        — PDF and Word document parsing
streamlit                 — Chat UI
python-dotenv             — .env file loading
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| "No documents have been uploaded yet" | Vector store is empty | Upload a document first |
| "This information is not mentioned" for a valid query | Retrieval missing the right chunk | Re-upload after clearing `vector_store/` |
| `429 RESOURCE_EXHAUSTED` | Gemini daily quota hit | Wait ~1 min or switch `LLM_MODEL` in config |
| `Cannot reach backend` in UI | FastAPI not running | Run `uvicorn backend.main:app --reload` |
| Upload succeeds but wrong chunk count | Old vector store from previous chunking strategy | Delete `vector_store/` folder and re-upload |

### Reset the vector store

```bash
# Windows
Remove-Item -Recurse -Force vector_store

# Linux / macOS
rm -rf vector_store/
```

Then re-upload all documents through the UI.
