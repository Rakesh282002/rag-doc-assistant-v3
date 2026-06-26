# AI Document Assistant — v3 (Gemini 3.1 Flash-Lite + Semantic Cache + MCP Integration)

A production-ready RAG (Retrieval-Augmented Generation) application that lets you upload a document and ask natural-language questions about it. Built with **Gemini 3.1 Flash-Lite**, **hybrid retrieval** (FAISS + BM25), **cross-encoder reranking**, **semantic caching**, **conversation memory**, and **MCP (Model Context Protocol)** integration for extended tool capabilities like web search and Google Maps links.

Answers are grounded strictly in the uploaded content — no hallucination.

---

## Key Features

| Feature | Description |
|---|---|
| **Hybrid Retrieval** | FAISS semantic search + BM25 keyword search for comprehensive document coverage |
| **Cross-Encoder Reranking** | `ms-marco-MiniLM-L-6-v2` reranks candidates for precise relevance scoring |
| **Section Affinity Boost** | +2.5 score bonus when question topic aligns with chunk section (experience, skills, header, etc.) |
| **Header Chunk Injection** | Identity/contact queries always retrieve the document header — even without metadata |
| **Semantic Caching** | Cosine similarity + LLM validation avoids redundant API calls for repeated/similar questions |
| **Conversation Memory** | LLM-based query rewriting resolves follow-up references (next, previous, that, etc.) |
| **MCP Integration** | Web search (DuckDuckGo) and Google Maps link generation via Model Context Protocol tools |
| **Location Detection** | Auto-detects location queries and augments answers with Maps links |
| **Single-Document Mode** | Uploading a new document fully clears previous data (vector store + cache) |
| **Keep-Alive** | Background health pings prevent Streamlit Cloud from sleeping |

### Performance Metrics

> Achieved **100% Recall@K** and improved answer faithfulness by **14%** using hybrid retrieval (FAISS + BM25) and cross-encoder reranking; reduced latency by **60%** via semantic caching.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Streamlit UI  (:8501 / :8502)                    │
│   Sidebar: upload / manage docs / clear cache    Main: chat         │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────────┐
│                      RAG Pipeline (backend/)                         │
│                                                                     │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────────────────┐  │
│  │ Semantic     │  │ Conversation │  │ Retrieval                 │  │
│  │ Cache        │  │ Memory       │  │                           │  │
│  │ (cosine +   │  │ (LLM query   │  │  1. FAISS (k=15)         │  │
│  │  LLM valid) │  │  rewriter)   │  │  2. BM25 (k=15)          │  │
│  └──────┬──────┘  └──────┬───────┘  │  3. Merge + header inject│  │
│         │                 │          │  4. Cross-encoder rerank  │  │
│    hit? │            rewrite?        │  5. Section affinity      │  │
│    ↓YES │                 │          │  6. Score gap filter      │  │
│  return │                 ↓          └───────────┬───────────────┘  │
│         └────────────────►──────────────────────►│                  │
│                                                  ↓                  │
│                                     Gemini 3.1 Flash-Lite (LLM)     │
│                                                  │                  │
│                                     ┌────────────▼──────────────┐   │
│                                     │  MCP Location Enrichment  │   │
│                                     │  (Maps link if location   │   │
│                                     │   query detected)         │   │
│                                     └───────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                            │
              ┌─────────────▼──────────────────┐
              │      MCP Server (stdio/SSE)     │
              │  Tools: web_search,             │
              │  search_location_info,          │
              │  generate_maps_link             │
              └────────────────────────────────┘
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
| MCP transport | stdio (local) / SSE (remote on Render) | Flexible deployment — subprocess or persistent server |
| Document mode | Single-document replacement | Prevents stale data from old uploads |

---

## Project Structure

```
rag-doc-assistant-v3/
├── backend/
│   ├── __init__.py
│   ├── config.py               # All tunable settings in one place
│   ├── document_processor.py   # Loaders + smart chunking (resume-aware)
│   ├── rag_pipeline.py         # Hybrid retrieval, reranking, LLM chain, MCP augmentation
│   ├── semantic_cache.py       # Semantic caching layer (cosine + LLM validation)
│   ├── location_detector.py    # Detects location queries for web search enrichment
│   ├── mcp_server.py           # MCP server — exposes web_search, maps tools
│   ├── mcp_client.py           # MCP client — connects via stdio/SSE/streamable-http
│   ├── web_search_tool.py      # DuckDuckGo web search (standalone fallback)
│   └── main.py                 # FastAPI app + REST endpoints
├── frontend/
│   └── app.py                  # Streamlit chat UI (standalone mode)
├── streamlit_app.py            # Unified Streamlit app (Streamlit Cloud / local)
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

### Upload Flow

```
File upload
    │
    ▼
load_document()              ← PyPDFLoader / Docx2txtLoader / TextLoader
    │
    ▼
split_documents()            ← resume-aware or generic chunking
    │
    ▼
add_to_vector_store()
    ├── HuggingFace embed → FAISS.save_local()
    └── append raw chunks → chunks.pkl  (for BM25)
```

### Query Flow

```
User question
    │
    ▼
Greeting detection                ← skip RAG for "hi", "thanks", etc.
    │
    ▼
Conversation rewrite              ← LLM resolves follow-ups using chat history
    │ (standalone → unchanged)
    │ (follow-up → rewritten to broad standalone query)
    ▼
Semantic cache lookup             ← cosine sim + LLM validation (skipped for follow-ups)
    │ (hit → return cached answer)
    │ (miss ↓)
    ▼
Query expansion                   ← "name" → "full name candidate name person name..."
    │
    ▼
FAISS similarity_search(k=15)    ← semantic candidates
    +
BM25Retriever.invoke(k=15)       ← keyword candidates
    │
    ▼
Merge + deduplicate               ← by first 200 chars of content
    + header chunk injection      ← force-include header for identity/contact queries
    │
    ▼
CrossEncoder.predict()            ← score every candidate against question
    + section affinity boost      ← +2.5 when question topic matches chunk section
    + content-based inference     ← detect header by all-caps first line when no metadata
    │
    ▼
Keep top-5 within SCORE_GAP       ← drop outlier chunks
    │
    ▼
_format_chunk()                   ← prefix metadata labels (Company/Role/Period/Section)
    │
    ▼
Gemini LLM (LCEL chain)           ← strict grounded-answer prompt + conversation context
    │
    ▼
MCP location enrichment           ← append Maps link if location query detected
    │
    ▼
Store in semantic cache            ← for future similar questions (skip "not found" answers)
    │
    ▼
{ answer, sources, mcp_tool_used? }
```

---

## MCP (Model Context Protocol) Integration

The application includes an MCP server that exposes external tools, enabling the RAG pipeline to augment answers with live data.

### MCP Server Tools

| Tool | Description |
|---|---|
| `web_search` | Search the web via DuckDuckGo (up to 10 results) |
| `search_location_info` | Search for location-specific details (attractions, climate, weather) |
| `generate_maps_link` | Generate a clickable Google Maps link for a location |

### Transport Modes

| Mode | Use Case | Config |
|---|---|---|
| **stdio** | Local development (auto-spawns subprocess) | Default when `MCP_SERVER_URL` is unset |
| **SSE** | Persistent remote server (e.g., Render) | Set `MCP_SERVER_URL=http://host:port/sse` |
| **Streamable HTTP** | Modern MCP protocol over HTTP | Set `MCP_SERVER_URL=http://host:port/mcp` |

### How Location Augmentation Works

1. After the LLM generates an answer, `location_detector.py` checks if the question is location-related
2. If yes, it extracts the location entity from the answer
3. Calls `generate_maps_link` via MCP to generate a Google Maps URL
4. Appends the clickable link to the answer

### Running MCP Server Standalone

```bash
# SSE mode (persistent HTTP service)
python -m backend.mcp_server --transport sse --port 8000

# stdio mode (used internally as subprocess)
python -m backend.mcp_server --transport stdio
```

---

## Document Chunking

### Resume / CV Documents (Auto-Detected)

The processor detects resumes by scanning for ≥2 of these signals in the first 3000 characters: `professional summary`, `professional experience`, `core competencies`, `technical skills`, `education`, `certifications`.

When detected, it applies a **3-stage strategy**:

1. **Section splitting** — regex finds standalone section headers and splits the document at each one.

2. **Role-level splitting** — within the experience section, each job entry becomes its own chunk (detected by `Company | Role` pattern followed by a date).

3. **Metadata enrichment** — every chunk carries structured metadata:
   - `section` — resume section (experience, skills, education, certifications, header, etc.)
   - `company` — company name (experience chunks)
   - `role` — job title (experience chunks)
   - `years` — date range (experience chunks)
   - `name` — candidate name (header chunk)
   - `skills` — extracted skills list (skills chunk)
   - `source` — path to the uploaded file

4. **Secondary split** — oversized section chunks are further split with `RecursiveCharacterTextSplitter`.

### Generic Documents

Uses `RecursiveCharacterTextSplitter` with separators `["\n\n", "\n", ".", " ", ""]`, producing overlapping windows of `CHUNK_SIZE` characters.

### Header Detection Without Metadata

When documents are processed with the generic chunker (no resume metadata), the pipeline still identifies the header chunk at query time by detecting an all-caps first line with 2–5 words in the first chunk. This ensures identity queries ("what is the name?") always work.

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

The cross-encoder reranker with section affinity boost filters out irrelevant chunks, so the LLM only sees the most relevant context.

---

## Dependencies

```
fastapi, uvicorn            — REST API server
langchain-*                 — LLM orchestration (LCEL pipeline)
langchain-google-genai      — Gemini LLM integration
langchain-huggingface       — Local HuggingFace embeddings
langchain-community         — FAISS vector store, BM25 retriever, document loaders
faiss-cpu                   — Local vector similarity search
sentence-transformers       — Cross-encoder reranker (ms-marco-MiniLM-L-6-v2)
rank-bm25                   — BM25 keyword retrieval
pypdf, python-docx          — PDF and Word document parsing
streamlit                   — Chat UI
python-dotenv               — .env file loading
mcp                         — Model Context Protocol (client + server)
httpx                       — HTTP client for MCP/web search
beautifulsoup4              — HTML parsing for DuckDuckGo results
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

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GOOGLE_API_KEY` | Yes | Google AI Studio API key for Gemini |
| `MCP_SERVER_URL` | No | Remote MCP server URL (leave unset for local stdio mode) |
| `STREAMLIT_APP_URL` | No | Public app URL for keep-alive pings (Streamlit Cloud) |
| `STREAMLIT_KEEPALIVE_ENABLED` | No | Enable/disable keep-alive (`true`/`false`, default `true`) |

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| "No documents have been uploaded yet" | Vector store is empty | Upload a document first |
| "This information is not mentioned" for a valid query | Retrieval missing the right chunk | Clear cache (sidebar button) and re-upload |
| "name of candidate?" returns wrong answer | Stale semantic cache entry | Click 🗑️ Clear Cache in sidebar |
| `429 RESOURCE_EXHAUSTED` | Gemini daily quota hit | Wait ~1 min or switch `LLM_MODEL` in config |
| `Cannot reach backend` in UI | FastAPI not running | Run `uvicorn backend.main:app --reload` |
| Upload succeeds but wrong chunk count | Old vector store from previous chunking | Delete `vector_store/` folder and re-upload |
| MCP tool errors in logs | MCP server not reachable | Check `MCP_SERVER_URL` or run server locally |

---

## GitHub Repository

- **RAG App**: [github.com/Rakesh282002/rag-doc-assistant-v3](https://github.com/Rakesh282002/rag-doc-assistant-v3)
- **MCP Server (Render deployment)**: [github.com/Rakesh282002/MCP_SERVER](https://github.com/Rakesh282002/MCP_SERVER)

### Reset the vector store

```bash
# Windows
Remove-Item -Recurse -Force vector_store

# Linux / macOS
rm -rf vector_store/
```

Then re-upload all documents through the UI.

---

## MCP Integration — Web Search for Location Queries

**v4 Feature:** This project now includes **Model Context Protocol (MCP)** integration with web search capabilities. When you ask questions about locations or entities mentioned in your documents, the system can augment answers with real-time web search results.

### Web Search Features

#### 1. **Location Detection**
- Automatically detects location-related queries:
  - "Where is [company] located?"
  - "What city is the office in?"
  - "Climate in [location]?"
  - "Distance to [location]?"

#### 2. **Automatic Web Search Augmentation**
- When document lacks location details → triggers web search
- Searches for: tourist attractions, climate, coordinates, business info
- Results merged seamlessly with document context

#### 3. **User Query Examples**

```
User: Where is the company headquartered?

[Document says: "Acme Corp is based in California"]

AI Response:
"The company is headquartered in California.

---

**Additional Information from Web Search:**
Search results for 'California information details climate attractions':
1. California — USA State Information
   URL: https://example.com/california
   Known for tech industry hub, diverse climate zones, major cities like SF and LA...
```

### How It Works

**New Modules:**
- `backend/web_search_tool.py` — DuckDuckGo web search (no API key needed)
- `backend/location_detector.py` — Question intent analysis for location queries

**Pipeline Integration:**
1. User asks a question
2. RAG retrieves document content
3. LLM generates initial answer
4. Location detector checks: is this a location query?
5. If answer is incomplete/missing: trigger web search
6. Augment answer with web results
7. Return combined answer to user

### Web Search Tools

#### `web_search(query: str, max_results: int = 5) -> str`
General-purpose web search using DuckDuckGo.
```python
from backend.web_search_tool import web_search

results = web_search("Paris tourist attractions", max_results=3)
print(results)
```

#### `search_location_info(location_name: str) -> str`
Specialized location search that includes climate, attractions, details.
```python
from backend.web_search_tool import search_location_info

location_details = search_location_info("London")
print(location_details)
```

### Configuration

**Enable/Disable Web Search:**
Edit `backend/rag_pipeline.py` to control when web search is triggered:

```python
# In query_documents() function around line ~475:
should_search, search_query = should_augment_with_web_search(answer, resolved_question)

if should_search and search_query:
    # Web search is triggered when:
    # 1. Answer says "not mentioned in the document"
    # 2. Question is location-related
    # 3. Answer is very short (<100 chars)
```

### Dependencies Added

```
httpx>=0.25.0              # HTTP client for web requests (replaces requests)
beautifulsoup4>=4.12.0     # HTML parsing for search results
```

### Limitations & Notes

1. **Rate Limiting:** DuckDuckGo may rate-limit after many rapid requests. Add delays if needed.
2. **No API Key Required:** Web search uses public DuckDuckGo endpoints (no authentication).
3. **Accuracy:** Web search results are informational and should be validated by user.
4. **Opt-in:** Web search only triggered for location queries or when document lacks info.
5. **Performance:** Web search adds ~2-5 seconds per query (network latency).

### Future Enhancements

- [ ] Add weather API integration (Open-Meteo)
- [ ] Caching web search results to reduce API calls
- [ ] Configurable location search templates
- [ ] Multi-source search (Google Custom Search, Bing)
- [ ] Entity linking to improve location detection

### Troubleshooting Web Search

| Issue | Cause | Fix |
|-------|-------|-----|
| Web search not triggering | Question might not be detected as location query | Check `location_detector.py` keyword list |
| "Error performing web search" | Network timeout or DuckDuckGo rate limit | Retry after a few seconds |
| Incomplete location results | Search query too vague | More specific location names work better |
| Web search results irrelevant | DuckDuckGo returned off-topic results | Refine the query or disable web search |

---
