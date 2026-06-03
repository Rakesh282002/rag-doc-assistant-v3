import os
from dotenv import load_dotenv

load_dotenv()

# --- LLM / Embeddings ---
# Try Streamlit secrets first (for Streamlit Cloud), fall back to .env
def _get_secret(key: str, default: str = "") -> str:
    try:
        import streamlit as st
        return st.secrets.get(key, os.getenv(key, default))
    except Exception:
        return os.getenv(key, default)

GOOGLE_API_KEY: str = _get_secret("GOOGLE_API_KEY")
EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"   # local, no API calls
CROSS_ENCODER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # local reranker

# Gemini 3.1 Flash-Lite — stable, cost-efficient multimodal model.
LLM_MODEL: str = "gemini-3.1-flash-lite"
LLM_TEMPERATURE: float = 0.3

# --- Chunking ---
CHUNK_SIZE: int = 1000
CHUNK_OVERLAP: int = 200

# --- Retrieval ---
INITIAL_RETRIEVAL_K: int = 10    # candidates from each retriever before reranking
MAX_RETRIEVAL_DOCS: int = 5      # top chunks after reranking sent to LLM
SCORE_GAP: float = 5.0           # only keep chunks within this many points of the best score

# --- Semantic Cache ---
# Any candidate with cosine similarity >= floor is passed to the LLM validator.
# The LLM decides whether the cached question truly covers the new question.
# Anything below the floor is a definite miss (no LLM call).
CACHE_CANDIDATE_FLOOR: float = 0.55      # min cosine sim to be a validation candidate
CACHE_TTL_DAYS: int = 7                   # days before a cache entry is expired
CACHE_MAX_SIZE: int = 500                 # max entries kept (LRU eviction beyond this)

# --- Paths ---
BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_DIR: str = os.path.join(BASE_DIR, "uploads")
VECTOR_STORE_DIR: str = os.path.join(BASE_DIR, "vector_store")
DOCUMENTS_REGISTRY: str = os.path.join(UPLOAD_DIR, "documents.json")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
