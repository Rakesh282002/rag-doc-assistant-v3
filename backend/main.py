import os
import uuid
import json

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

from backend.document_processor import load_document, split_documents, SUPPORTED_EXTENSIONS
from backend.rag_pipeline import add_to_vector_store, query_documents
from backend.semantic_cache import get_cache
from backend.config import UPLOAD_DIR, DOCUMENTS_REGISTRY

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="RAG Document Assistant",
    description="Upload documents, ask questions — answers powered by Gemini + FAISS.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Registry helpers  (simple JSON file, no DB dependency)
# ---------------------------------------------------------------------------

def _load_registry() -> list:
    if os.path.exists(DOCUMENTS_REGISTRY):
        with open(DOCUMENTS_REGISTRY, "r") as f:
            return json.load(f)
    return []


def _save_registry(docs: list) -> None:
    with open(DOCUMENTS_REGISTRY, "w") as f:
        json.dump(docs, f, indent=2)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str


class SourceItem(BaseModel):
    content: str
    source: str
    page: str | int


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceItem]
    cached: bool = False
    cache_similarity: float | None = None
    cached_question: str | None = None


class DocumentItem(BaseModel):
    id: str
    filename: str
    chunks: int


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", tags=["Health"])
def root():
    return {"message": "RAG Document Assistant API is running."}


@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Cache routes
# ---------------------------------------------------------------------------

@app.get("/cache/stats", tags=["Cache"])
def cache_stats():
    """Return current semantic cache statistics and stored questions."""
    return get_cache().stats()


@app.delete("/cache", tags=["Cache"])
def clear_cache():
    """Wipe the entire semantic cache."""
    count = get_cache().clear()
    return {"message": f"Cleared {count} cached entries."}


@app.post("/upload", tags=["Documents"], response_model=DocumentItem)
async def upload_document(file: UploadFile = File(...)):
    """Upload a PDF / DOCX / TXT file, chunk it, embed it into FAISS."""
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(SUPPORTED_EXTENSIONS)}",
        )

    doc_id = str(uuid.uuid4())
    save_path = os.path.join(UPLOAD_DIR, f"{doc_id}{ext}")

    content = await file.read()
    with open(save_path, "wb") as f:
        f.write(content)

    try:
        documents = load_document(save_path)
        chunks = split_documents(documents)
        add_to_vector_store(chunks)
    except Exception as exc:
        os.remove(save_path)
        raise HTTPException(status_code=500, detail=f"Processing failed: {exc}") from exc

    registry = _load_registry()
    registry.append(
        {"id": doc_id, "filename": file.filename, "path": save_path, "chunks": len(chunks)}
    )
    _save_registry(registry)

    return {"id": doc_id, "filename": file.filename, "chunks": len(chunks)}


@app.get("/documents", tags=["Documents"], response_model=List[DocumentItem])
def list_documents():
    """Return all indexed documents."""
    return _load_registry()


@app.delete("/documents/{doc_id}", tags=["Documents"])
def delete_document(doc_id: str):
    """Remove a document record (file + registry entry). Vector store is NOT rebuilt."""
    registry = _load_registry()
    doc = next((d for d in registry if d["id"] == doc_id), None)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    if os.path.exists(doc["path"]):
        os.remove(doc["path"])

    _save_registry([d for d in registry if d["id"] != doc_id])
    return {"message": f"'{doc['filename']}' deleted successfully."}


@app.post("/ask", tags=["QA"], response_model=QueryResponse)
def ask_question(request: QueryRequest):
    """Ask a question against all indexed documents."""
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        result = query_documents(question)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}") from exc
    return result
