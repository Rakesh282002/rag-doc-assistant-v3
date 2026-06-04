"""
Unified Streamlit app for deployment on Streamlit Community Cloud.
Directly imports backend logic — no separate FastAPI server needed.
"""

import os
import sys
import uuid
import json
import tempfile

import streamlit as st

# Ensure project root is on path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.config import UPLOAD_DIR, DOCUMENTS_REGISTRY, GOOGLE_API_KEY
from backend.document_processor import load_document, split_documents, SUPPORTED_EXTENSIONS
from backend.rag_pipeline import add_to_vector_store, query_documents
from backend.semantic_cache import get_cache

# ---------------------------------------------------------------------------
# Registry helpers
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
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .source-card {
        background: #f8f9fa;
        border-left: 4px solid #4285f4;
        padding: 10px 14px;
        border-radius: 4px;
        margin-bottom: 8px;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar — document management
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("📁 Documents")
    st.caption("Upload PDFs, DOCX, or TXT files to build your knowledge base.")

    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["pdf", "docx", "txt"],
        label_visibility="collapsed",
    )

    if uploaded_file:
        if st.button("⬆️  Process & Index", use_container_width=True, type="primary"):
            with st.spinner(f"Processing **{uploaded_file.name}**…"):
                try:
                    ext = os.path.splitext(uploaded_file.name)[1].lower()
                    if ext not in SUPPORTED_EXTENSIONS:
                        st.error(f"Unsupported file type: {ext}")
                    else:
                        # Save uploaded file
                        doc_id = str(uuid.uuid4())
                        filename = f"{doc_id}{ext}"
                        filepath = os.path.join(UPLOAD_DIR, filename)
                        with open(filepath, "wb") as f:
                            f.write(uploaded_file.getvalue())

                        # Process document
                        raw_docs = load_document(filepath)
                        chunks = split_documents(raw_docs)
                        add_to_vector_store(chunks)

                        # Update registry
                        registry = _load_registry()
                        registry.append({
                            "id": doc_id,
                            "filename": uploaded_file.name,
                            "chunks": len(chunks),
                        })
                        _save_registry(registry)

                        st.success(f"✅ Indexed **{uploaded_file.name}** — {len(chunks)} chunks")
                        st.rerun()
                except Exception as e:
                    st.error(f"Error processing document: {e}")

    st.divider()
    st.subheader("Indexed Documents")

    docs = _load_registry()
    if docs:
        for doc in docs:
            col1, col2 = st.columns([5, 1])
            col1.markdown(f"📄 **{doc['filename']}**")
            col1.caption(f"{doc['chunks']} chunks · id: {doc['id'][:8]}…")
            if col2.button("🗑️", key=f"del_{doc['id']}", help="Delete document"):
                # Remove file
                registry = _load_registry()
                registry = [d for d in registry if d["id"] != doc["id"]]
                _save_registry(registry)
                # Try to remove the actual file
                for ext in SUPPORTED_EXTENSIONS:
                    fpath = os.path.join(UPLOAD_DIR, f"{doc['id']}{ext}")
                    if os.path.exists(fpath):
                        os.remove(fpath)
                        break
                st.toast("Document deleted.", icon="🗑️")
                st.rerun()
    else:
        st.info("No documents indexed yet.")

    st.divider()
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.subheader("⚡ Semantic Cache")
    try:
        cache = get_cache()
        stats = cache.stats()
        st.metric("Cached questions", stats["size"], help=f"Max {stats['max_size']} · TTL {stats['ttl_days']} days")
        st.caption(f"Candidate floor: {stats['candidate_floor']}")
        if stats.get("entries"):
            with st.expander("View cached questions"):
                for e in stats["entries"]:
                    st.markdown(f"- `{e['cached_at']}` — {e['question']}")
        if st.button("🗑️ Clear Cache", use_container_width=True):
            cache.clear()
            st.toast("Cache cleared.", icon="🗑️")
            st.rerun()
    except Exception:
        st.caption("Cache unavailable.")

# ---------------------------------------------------------------------------
# Main — chat interface
# ---------------------------------------------------------------------------
st.title("🤖 AI Document Assistant")
st.caption("Powered by **Google Gemini 3.1 Flash-Lite** · **LangChain** · **FAISS**")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("cached"):
            st.caption(f"⚡ From cache · similarity {msg.get('cache_similarity', '')}")
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"📚 {len(msg['sources'])} source(s) used"):
                for i, src in enumerate(msg["sources"], 1):
                    fname = os.path.basename(src["source"])
                    st.markdown(
                        f'<div class="source-card">'
                        f"<b>Source {i}</b> — <code>{fname}</code> &nbsp;|&nbsp; Page: {src['page']}<br>"
                        f"<span style='color:#555'>{src['content']}</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

# Chat input
if question := st.chat_input("Ask a question about your documents…"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating answer…"):
            try:
                data = query_documents(question)
                answer = data["answer"]
                sources = data["sources"]
                is_cached = data.get("cached", False)
                cache_sim = data.get("cache_similarity")

                if is_cached:
                    st.caption(f"⚡ From cache · similarity {cache_sim}")
                st.markdown(answer)

                if sources:
                    with st.expander(f"📚 {len(sources)} source(s) used"):
                        for i, src in enumerate(sources, 1):
                            fname = os.path.basename(src["source"])
                            st.markdown(
                                f'<div class="source-card">'
                                f"<b>Source {i}</b> — <code>{fname}</code> &nbsp;|&nbsp; Page: {src['page']}<br>"
                                f"<span style='color:#555'>{src['content']}</span>"
                                f"</div>",
                                unsafe_allow_html=True,
                            )

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "sources": sources,
                     "cached": is_cached, "cache_similarity": cache_sim}
                )
            except Exception as e:
                st.error(f"Error: {e}")
