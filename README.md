# AI Document Assistant — v2 (Gemini 2.0 Flash Live)

A local RAG (Retrieval-Augmented Generation) application that lets you upload documents and ask natural-language questions about them. This v2 edition uses **Gemini 2.0 Flash Live** (`gemini-2.0-flash-live-001`) with streaming enabled for faster time-to-first-token responses. Answers are grounded strictly in the uploaded content — no hallucination of information not present in the document.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Streamlit UI  (:8501)                    │
│  Sidebar: upload / manage docs    Main: chat interface          │
└───────────────────────┬─────────────────────────────────────────┘
                        │  HTTP (REST)
┌───────────────────────▼─────────────────────────────────────────┐
│                     FastAPI Backend  (:8000)                    │
│  POST /upload   GET /documents   DELETE /documents/{id}         │
│  POST /ask                                                       │
└───────────┬──────────────────────────────┬──────────────────────┘
            │                              │
    ┌───────▼────────┐            ┌────────▼────────────┐
    │ document_       │            │  rag_pipeline.py    │
    │ processor.py    │            │                     │
    │                 │            │  1. FAISS semantic  │
    │  Resume:        │            │     search          │
    │  section/role   │   chunks   │  2. BM25 keyword    │
    │  chunking  ─────┼──────────► │     search          │
    │                 │            │  3. Merge + dedup   │
    │  Generic:       │            │  4. Cross-encoder   │
    │  Recursive      │            │     reranking       │
    │  splitter       │            │  5. Gemini LLM      │
    └─────────────────┘            └─────────────────────┘
```

### Key Design Decisions

| Concern | Choice | Reason |
|---|---|---|
| Embeddings | `BAAI/bge-small-en-v1.5` (local) | Zero API calls, fast on CPU, 384-dim |
| LLM | `gemini-2.0-flash-live-001` (streaming) | Low-latency live model, streaming responses |
| Vector store | FAISS (local file) | No server needed, persists across restarts |
| Keyword search | BM25 | Catches exact-match queries that semantic search misses |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` (local) | Precise relevance scoring, no API calls |
| Document chunking | Section/role-aware (resumes) + recursive (generic) | Preserves logical boundaries |

---

## Project Structure

```
rag-doc-assistant/
├── backend/
│   ├── __init__.py
│   ├── config.py               # All tunable settings in one place
│   ├── document_processor.py   # Loaders + smart chunking
│   ├── rag_pipeline.py         # Retrieval, reranking, LLM chain
│   └── main.py                 # FastAPI app + REST endpoints
├── frontend/
│   └── app.py                  # Streamlit chat UI
├── uploads/                    # Uploaded files + documents.json registry
├── vector_store/               # FAISS index + chunks.pkl (auto-created)
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
git clone <repo-url>
cd rag-doc-assistant
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
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | HuggingFace bi-encoder for FAISS indexing |
| `CROSS_ENCODER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranker model |
| `LLM_MODEL` | `gemini-2.0-flash-live-001` | Gemini Flash Live model (streaming enabled) |
| `LLM_TEMPERATURE` | `0.3` | Lower = more factual, less creative |
| `CHUNK_SIZE` | `1000` | Max characters per chunk (generic docs) |
| `CHUNK_OVERLAP` | `200` | Overlap between consecutive chunks |
| `INITIAL_RETRIEVAL_K` | `10` | Candidates fetched from each retriever |
| `MAX_RETRIEVAL_DOCS` | `5` | Top chunks passed to LLM after reranking |
| `SCORE_GAP` | `3.0` | Drop chunks more than this many points below the best reranker score |

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
FAISS similarity_search(k=10)   ← semantic candidates
    +
BM25Retriever.invoke(k=10)      ← keyword candidates
    │
    ▼
Merge + deduplicate              ← by first 200 chars of content
    │
    ▼
CrossEncoder.predict()           ← score every candidate against question
    │
    ▼
Keep top-5 within SCORE_GAP      ← drop outlier chunks
    │
    ▼
_format_chunk()                  ← prefix metadata labels (Company/Role/Period)
    │
    ▼
Gemini LLM (LCEL chain)          ← strict grounded-answer prompt
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
   ```python
   {
     "section": "experience",
     "company": "Phenom",
     "role": "Technical Delivery Manager & Solution Architect",
     "years": "Jan 2022 – Present",
     "source": "/path/to/file.pdf"
   }
   ```
   This metadata is prepended as a label when the chunk is shown to the LLM:
   ```
   [Company: Phenom | Role: Technical Delivery Manager | Period: Jan 2022 – Present]
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
  "answer": "His name is Devarapu Bullivenkanna (Venkat Devarapu).",
  "sources": [
    {
      "content": "Devarapu Bullivenkanna (Venkat Devarapu)...",
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

Additionally, the cross-encoder reranker filters out chunks with poor relevance scores (via `SCORE_GAP`), so the LLM only sees the most relevant context.

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
"# rag-doc-assistant-v3" 
