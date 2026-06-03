import os
import requests
import streamlit as st

API_BASE = "http://localhost:8002"

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
                    resp = requests.post(
                        f"{API_BASE}/upload",
                        files={"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)},
                        timeout=300,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        st.success(f"✅ Indexed **{data['filename']}** — {data['chunks']} chunks")
                        st.rerun()
                    else:
                        st.error(resp.json().get("detail", "Upload failed."))
                except requests.ConnectionError:
                    st.error("Cannot reach backend. Is FastAPI running?")

    st.divider()
    st.subheader("Indexed Documents")

    try:
        docs_resp = requests.get(f"{API_BASE}/documents", timeout=15)
        if docs_resp.status_code == 200:
            docs = docs_resp.json()
            if docs:
                for doc in docs:
                    col1, col2 = st.columns([5, 1])
                    col1.markdown(f"📄 **{doc['filename']}**")
                    col1.caption(f"{doc['chunks']} chunks · id: {doc['id'][:8]}…")
                    if col2.button("🗑️", key=f"del_{doc['id']}", help="Delete document"):
                        del_resp = requests.delete(f"{API_BASE}/documents/{doc['id']}")
                        if del_resp.status_code == 200:
                            st.toast("Document deleted.", icon="🗑️")
                            st.rerun()
            else:
                st.info("No documents indexed yet.")
    except requests.ConnectionError:
        st.warning("⚠️ Backend not reachable.")

    st.divider()
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.subheader("⚡ Semantic Cache")
    try:
        stats_resp = requests.get(f"{API_BASE}/cache/stats", timeout=5)
        if stats_resp.status_code == 200:
            stats = stats_resp.json()
            st.metric("Cached questions", stats["size"], help=f"Max {stats['max_size']} · TTL {stats['ttl_days']} days")
            st.caption(f"Candidate floor: {stats['candidate_floor']}")
            if stats["entries"]:
                with st.expander("View cached questions"):
                    for e in stats["entries"]:
                        st.markdown(f"- `{e['cached_at']}` — {e['question']}")
            if st.button("🗑️ Clear Cache", use_container_width=True):
                requests.delete(f"{API_BASE}/cache", timeout=5)
                st.toast("Cache cleared.", icon="🗑️")
                st.rerun()
    except requests.ConnectionError:
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
    # Show user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Get assistant response
    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating answer…"):
            try:
                resp = requests.post(
                    f"{API_BASE}/ask",
                    json={"question": question},
                    timeout=120,
                )
                if resp.status_code == 200:
                    data = resp.json()
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
                else:
                    try:
                        err = resp.json().get("detail", "Unknown error.")
                    except Exception:
                        err = resp.text or f"HTTP {resp.status_code}"
                    st.error(f"Error: {err}")
            except requests.ConnectionError:
                st.error("Cannot connect to backend. Start it with: `uvicorn backend.main:app --reload`")
            except requests.Timeout:
                st.error("Request timed out. The LLM may be slow — try again.")
