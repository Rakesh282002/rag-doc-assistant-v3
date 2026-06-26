# RAG Document Assistant — v3 Pipeline Documentation

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

### Step 0: Greeting Detection

- Quick check against a set of known greetings/chitchat (`hi`, `hello`, `thanks`, `bye`, etc.)
- If match → return friendly response immediately (no RAG pipeline invoked)

### Step 0a: Conversation-Aware Query Rewrite

- Checks if there's prior conversation AND the question contains context words (`next`, `previous`, `that`, `this`, `it`, `second`, `third`, `last`, etc.) or is ≤3 words
- If YES → LLM rewrites the question into a standalone broad query
- **Example:** `"next company?"` → `"What are all the companies worked at in chronological order? Specifically the second one."`
- The rewriter is instructed to NOT include specific previous answers in the query to avoid biasing retrieval

### Step 0b: Semantic Cache Check (skipped for follow-ups)

- If question was NOT rewritten: embed question, compare against cached Q&A pairs (cosine similarity)
- If similarity ≥ 0.55 (candidate floor): ask validator LLM "is this the same question?"
- Validator checks SCOPE (filters added/removed?) and INTENT (list vs count?)
- Synonyms recognized: "person" / "candidate" / "applicant" treated as equivalent
- If YES → return cached answer instantly (no retrieval/LLM call)
- **"Not found" answers are never cached** to prevent stale negative responses

### Step 1: Query Expansion + Semantic Search (FAISS)

- Expand query with topic keywords:
  - `"name"` → adds `"full name candidate name person name applicant identity"`
  - `"skills"` → adds `"technical skills programming languages tools technologies expertise"`
  - `"experience"` → adds `"work experience professional experience employment company role"`
- FAISS similarity search → top 15 candidates

### Step 2: Keyword Search (BM25)

- BM25 retriever (term frequency-based) over all chunks → top 15 candidates
- Catches exact keyword matches that embedding search might miss

### Step 3: Merge, Deduplicate & Header Injection

- Combine FAISS + BM25 results
- Remove duplicates by comparing first 200 characters of content
- **Header chunk injection:** For identity/contact queries (detected by keywords: `name`, `who`, `candidate`, `contact`, `email`, `phone`, etc.), the pipeline force-includes the header chunk
  - Detected by metadata: `section == "header"`
  - OR by content pattern: first line is all-caps with 2–5 words AND it's the first chunk in the document
  - Ensures name/contact answers are never missed regardless of chunking strategy

### Step 4: Cross-Encoder Reranking + Section Affinity Boost

- Cross-encoder model (`cross-encoder/ms-marco-MiniLM-L-6-v2`) scores each (question, chunk) pair for relevance
- **Section affinity boost (+2.5):** Applied when question topic aligns with chunk's section:

| Section | Trigger Keywords |
|---------|-----------------|
| `experience` | experience, work, job, role, company, career, employment |
| `education` | education, degree, university, college, bachelor, masters |
| `skills` | skills, technologies, tools, programming, expertise |
| `certifications` | certification, certified, certificate, license |
| `achievement` | achievement, award, recognition, accomplishment |
| `header` | name, who, person, candidate, contact, email, phone, address, location |

- **Content-based inference:** When chunks have no section metadata (generic chunking), the pipeline infers `section="header"` if the first line is all-caps with 2–5 words
- Keep top chunks within 3.0 points of the best score (maximum 5 chunks)

### Step 5: LLM Generation (Google Gemini 3.1 Flash-Lite)

- Build context from top chunks with metadata labels (Company, Role, Period, Section)
- Include recent conversation history (last 3 Q&A pairs) if it's a follow-up
- Include today's date for duration calculations
- **Prompt rules enforce:**
  - Answer ONLY from retrieved passages
  - Say "not mentioned" if info isn't there
  - Don't repeat previously given answers
  - Indicate "(currently working)" for present employment
  - List ALL items found (don't summarize)
  - May perform simple arithmetic on stated values (e.g., calculating years of experience)

### Step 5b: MCP Location Enrichment

- After LLM generates the answer, `location_detector.py` checks if the query is location-related
- If YES and a location entity is detected in the answer:
  - Calls `generate_maps_link` via MCP client (stdio or remote SSE)
  - Appends a clickable Google Maps link to the answer
  - Records `mcp_tool_used` in the response

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

## 4. MCP (Model Context Protocol) Integration

### Architecture

```
RAG Pipeline (rag_pipeline.py)
    │
    ▼
Location Detector (location_detector.py)
    │ detects location query + extracts entity from answer
    ▼
MCP Client (mcp_client.py)
    │ stdio / SSE / streamable-http transport
    ▼
MCP Server (mcp_server.py)
    │ FastMCP with registered tools
    ▼
Tools: web_search, search_location_info, generate_maps_link
```

### MCP Server Tools

| Tool | Input | Output |
|------|-------|--------|
| `web_search` | `query`, `max_results` | DuckDuckGo search results (title, URL, snippet) |
| `search_location_info` | `location_name` | Web search for location details (attractions, climate, weather) |
| `generate_maps_link` | `location` | Markdown Google Maps link: `[📍 View on Google Maps](url)` |

### Transport Selection

- **No `MCP_SERVER_URL`** → stdio mode (spawns `mcp_server.py` as subprocess)
- **URL contains `/sse`** → SSE client mode
- **URL without `/sse`** → Streamable HTTP mode (modern MCP protocol)

### Error Handling

- MCP failures are non-fatal — the answer is returned without augmentation
- Errors logged as `[MCP] Error calling MCP tool: ...`
- `mcp_error` field added to response for debugging

---

## 5. Architecture Components

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Frontend | Streamlit | Chat UI, file upload, document management |
| Embeddings | BAAI/bge-base-en-v1.5 | 768-dim dense vectors for semantic search |
| Vector Store | FAISS | Fast similarity search over embeddings |
| Keyword Search | BM25Retriever | Term-frequency based retrieval |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 | Precise relevance scoring |
| LLM | Google Gemini 3.1 Flash-Lite | Answer generation + cache validation + query rewriting |
| Cache | Custom semantic cache (cosine + LLM validator) | Avoid redundant LLM calls |
| MCP Server | FastMCP (mcp library) | Web search + maps tools via Model Context Protocol |
| MCP Client | mcp client library (stdio/SSE/HTTP) | Connects to MCP server for tool invocation |
| Location Detector | Rule-based (regex + keywords) | Identifies location queries for MCP enrichment |
| Web Search | DuckDuckGo (via httpx + BeautifulSoup) | Live web data for location augmentation |
| Persistence | FAISS index + pickle files | Survive app restarts |

---

## 6. Configuration (config.py)

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

## 7. Flow Diagram

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
│  Greeting? → YES → return friendly message                      │
│       │ NO                                                      │
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
│  Query expansion (topic keywords)                               │
│       │                                                         │
│       ▼                                                         │
│  FAISS semantic search (15 docs)                                │
│       +                                                         │
│  BM25 keyword search (15 docs)                                  │
│       │                                                         │
│       ▼                                                         │
│  Merge + deduplicate + header chunk injection                   │
│       │                                                         │
│       ▼                                                         │
│  Cross-encoder rerank + section affinity boost                  │
│       │                                                         │
│       ▼                                                         │
│  Top 5 chunks → Gemini LLM (with conversation context)          │
│       │                                                         │
│       ▼                                                         │
│  Location query? → YES → MCP generate_maps_link                 │
│       │                                                         │
│       ▼                                                         │
│  Return answer + sources → Cache result                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 8. Key v3 Improvements

1. **MCP Integration** — Web search and Google Maps link generation via Model Context Protocol
2. **Header chunk injection** — Identity/contact queries always find the header, even without metadata
3. **Content-based section inference** — Detects header by all-caps pattern when metadata is absent
4. **Section affinity for header** — +2.5 boost for name/contact queries on header chunks
5. **Query expansion** — Short queries expanded with synonyms for better retrieval
6. **Synonym-aware cache validation** — "person" / "candidate" / "applicant" treated as equivalent
7. **Single-document replacement** — Uploading a new doc fully clears the old one (no stale data)
8. **Conversation memory** — Follow-up questions understand context from prior Q&A
9. **Smart cache bypass** — Conversation-dependent questions skip cache to avoid wrong matches
10. **No caching of "not found"** — Prevents negative answers from being served to rephrased questions
11. **Current employment indicator** — Answers include "(currently working)" when applicable
12. **Broad rewrite strategy** — Follow-up queries rewritten to find ALL items, preventing retrieval bias
13. **Keep-alive system** — Background health pings prevent Streamlit Cloud from sleeping
14. **Location detection** — Auto-detects location queries and enriches answers with Maps links

## 9. Deployment Options

### Local Development

```bash
# Start Streamlit (unified mode — no separate backend needed)
streamlit run streamlit_app.py --server.port 8502

# Or separate backend + frontend
uvicorn backend.main:app --reload          # Backend on :8000
streamlit run frontend/app.py              # Frontend on :8501
```

### Streamlit Community Cloud

- Push to GitHub → connect repository in Streamlit Cloud
- Set `GOOGLE_API_KEY` in Streamlit secrets
- Optional: set `STREAMLIT_APP_URL` for keep-alive pings

### MCP Server on Render

- Deploy `mcp_server.py` as a web service on Render
- Set transport to SSE, disable DNS rebinding protection for public access
- Set `MCP_SERVER_URL` in the RAG app to point to the Render URL
