import streamlit as st
import requests
from typing import List, Dict, Any, Optional

# Page configuration
st.set_page_config(
    page_title="DocuMind - AI-Powered PDF Q&A",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .source-box {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 10px 14px;
        margin-top: 8px;
        font-size: 0.88rem;
    }
    .badge {
        display: inline-block;
        padding: 2px 8px;
        font-size: 0.78rem;
        font-weight: 600;
        border-radius: 4px;
        background-color: #E0E7FF;
        color: #3730A3;
        margin-right: 6px;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []

if "active_document_id" not in st.session_state:
    st.session_state.active_document_id = None

if "uploaded_documents" not in st.session_state:
    st.session_state.uploaded_documents = []


# Sidebar Configuration
with st.sidebar:
    st.title("🧠 DocuMind")
    st.caption("AI-Powered PDF Question Answering System (Hybrid RAG + Gemini)")
    
    st.divider()

    backend_url = st.text_input(
        "Backend API URL",
        value="http://localhost:8000",
        help="FastAPI backend address"
    ).rstrip("/")

    # Function to fetch document list from backend
    def fetch_documents() -> List[Dict[str, Any]]:
        try:
            res = requests.get(f"{backend_url}/documents", timeout=5)
            if res.status_code == 200:
                return res.json().get("documents", [])
            return []
        except Exception:
            return []

    # Upload PDF Section
    st.subheader("📄 Upload Document")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Upload a selectable text PDF file"
    )

    if uploaded_file is not None:
        if st.button("Process & Ingest PDF", type="primary", use_container_width=True):
            with st.spinner("Processing PDF, extracting chunks and generating embeddings..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    response = requests.post(f"{backend_url}/documents/upload", files=files, timeout=60)
                    
                    if response.status_code == 201:
                        data = response.json()
                        st.success(f"✅ Processed successfully!")
                        st.info(f"**Doc ID:** `{data['document_id']}`\n\n"
                                f"**Pages:** {data['total_pages']} | **Chunks:** {data['total_chunks']}")
                        st.session_state.active_document_id = data["document_id"]
                    else:
                        detail = response.json().get("detail", response.text)
                        st.error(f"Upload failed: {detail}")
                except Exception as e:
                    st.error(f"Connection error: {str(e)}")

    st.divider()

    # Document Management & Selection
    st.subheader("📚 Active Document")
    docs = fetch_documents()
    st.session_state.uploaded_documents = docs

    if docs:
        doc_options = {"All Documents": None}
        for d in docs:
            label = f"{d['filename']} ({d['total_pages']} pgs, {d['total_chunks']} chunks)"
            doc_options[label] = d["document_id"]

        selected_label = st.selectbox(
            "Select context document:",
            options=list(doc_options.keys()),
            index=0
        )
        st.session_state.active_document_id = doc_options[selected_label]

        # Document Deletion
        if st.session_state.active_document_id:
            if st.button("🗑️ Delete Selected Document", type="secondary", use_container_width=True):
                with st.spinner("Deleting document..."):
                    try:
                        del_res = requests.delete(
                            f"{backend_url}/documents/{st.session_state.active_document_id}",
                            timeout=10
                        )
                        if del_res.status_code == 200:
                            st.success("Document deleted successfully.")
                            st.session_state.active_document_id = None
                            st.rerun()
                        else:
                            st.error(del_res.json().get("detail", "Failed to delete"))
                    except Exception as e:
                        st.error(f"Error: {e}")
    else:
        st.info("No documents uploaded yet. Upload a PDF above to get started.")

    st.divider()
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# Main Chat Area
st.markdown('<div class="main-header">DocuMind: Document Q&A</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">'
    'Ask questions grounded strictly in your uploaded PDF documents. '
    'Powered by Hybrid Retrieval (BM25 + ChromaDB Cosine Vector Search with Reciprocal Rank Fusion) and Google Gemini.'
    '</div>',
    unsafe_allow_html=True
)

# Display existing conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander(f"📍 View Sources ({len(msg['sources'])})", expanded=False):
                for src in msg["sources"]:
                    st.markdown(
                        f"""
                        <div class="source-box">
                            <span class="badge">Page {src.get('page', 'N/A')}</span>
                            <strong>{src.get('filename', 'Unknown')}</strong>
                            <p style="margin-top:6px; color:#475569;"><em>"{src.get('snippet', '')}"</em></p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )


# User Question Input
if user_query := st.chat_input("Ask a question about your document(s)..."):
    # Append and render user message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Generate assistant answer
    with st.chat_message("assistant"):
        with st.spinner("Retrieving relevant context and formulating answer with Gemini..."):
            try:
                payload = {
                    "question": user_query,
                    "document_id": st.session_state.active_document_id
                }
                res = requests.post(f"{backend_url}/chat/ask", json=payload, timeout=60)
                
                if res.status_code == 200:
                    data = res.json()
                    answer = data.get("answer", "No answer received.")
                    sources = data.get("sources", [])

                    st.markdown(answer)

                    if sources:
                        with st.expander(f"📍 View Sources ({len(sources)})", expanded=True):
                            for src in sources:
                                st.markdown(
                                    f"""
                                    <div class="source-box">
                                        <span class="badge">Page {src.get('page', 'N/A')}</span>
                                        <strong>{src.get('filename', 'Unknown')}</strong>
                                        <p style="margin-top:6px; color:#475569;"><em>"{src.get('snippet', '')}"</em></p>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                else:
                    error_detail = res.json().get("detail", res.text)
                    st.error(f"Error from server ({res.status_code}): {error_detail}")
            except Exception as e:
                st.error(f"Failed to communicate with backend: {str(e)}")

