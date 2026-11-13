import os
import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import streamlit as st
from app.config import settings
from app.core.pdf_loader import PDFDocumentLoader
from app.core.text_splitter import RecursiveTextSplitter
from app.core.vector_store import VectorStoreManager
from app.core.rag_engine import RAGEngine
from app.core.memory import memory_manager

# Page configuration
st.set_page_config(
    page_title="Multi-Doc Conversational RAG",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
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
    .citation-card {
        background-color: #F8FAFC;
        border-left: 4px solid #3B82F6;
        padding: 0.8rem 1rem;
        border-radius: 4px;
        margin-top: 0.6rem;
        font-size: 0.88rem;
    }
    .citation-badge {
        background-color: #DBEAFE;
        color: #1D4ED8;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_rag_components():
    """Initializes and caches RAG pipeline components."""
    pdf_loader = PDFDocumentLoader()
    text_splitter = RecursiveTextSplitter(
        chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP
    )
    vector_store = VectorStoreManager()
    rag_engine = RAGEngine(vector_store=vector_store, memory=memory_manager)
    return pdf_loader, text_splitter, vector_store, rag_engine


pdf_loader, text_splitter, vector_store, rag_engine = get_rag_components()

# Session state initialization
if "session_id" not in st.session_state:
    st.session_state.session_id = f"session_{os.urandom(4).hex()}"
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar: Document Management
with st.sidebar:
    st.header("📄 Document Ingestion")

    # Sample documents fast-loader
    st.markdown("### Pre-Bundled Samples")
    sample_files = list(settings.SAMPLE_DOCS_DIR.glob("*.pdf"))
    if sample_files:
        if st.button("📥 Index All Sample Reports", use_container_width=True):
            with st.spinner("Ingesting and embedding sample documents..."):
                pages = pdf_loader.load_multiple_pdfs(sample_files)
                chunks = text_splitter.split_pages(pages)
                vector_store.build_from_chunks(chunks)
                st.success(f"Indexed {len(sample_files)} sample documents ({len(chunks)} chunks)!")
                st.rerun()

    st.markdown("---")
    st.markdown("### Upload Custom PDFs")
    uploaded_files = st.file_uploader(
        "Upload one or more PDF reports",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload company policies, financial reports, or technical manuals.",
    )

    if uploaded_files:
        if st.button("🚀 Process & Ingest Files", type="primary", use_container_width=True):
            saved_paths = []
            with st.spinner("Parsing and indexing PDFs..."):
                for uploaded_file in uploaded_files:
                    save_path = settings.UPLOAD_DIR / uploaded_file.name
                    with open(save_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    saved_paths.append(save_path)

                pages = pdf_loader.load_multiple_pdfs(saved_paths)
                chunks = text_splitter.split_pages(pages)
                vector_store.add_chunks(chunks)
                st.success(f"Added {len(chunks)} chunks to vector index!")
                st.rerun()

    st.markdown("---")
    st.markdown("### 📊 Indexed Inventory")
    indexed_docs = vector_store.get_indexed_documents()
    if indexed_docs:
        st.write(f"**Total Documents:** {len(indexed_docs)}")
        st.write(f"**Total Chunks in FAISS:** {len(vector_store.chunks)}")
        for doc in indexed_docs:
            st.caption(f"• **{doc['document_name']}** ({doc['total_pages']} pgs, {doc['total_chunks']} chunks)")
    else:
        st.info("No documents currently indexed. Click above to index samples.")

    st.markdown("---")
    if st.button("🔄 Reset Conversation History", use_container_width=True):
        memory_manager.clear_session(st.session_state.session_id)
        st.session_state.messages = []
        st.rerun()


# Main Chat Interface
st.markdown('<div class="main-header">📚 Multi-Document Conversational RAG</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">'
    'Enterprise QA assistant with multi-document retrieval, conversational memory, and precise source citations.'
    '</div>',
    unsafe_allow_html=True,
)

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("citations"):
            with st.expander(f"📌 Source References ({len(msg['citations'])})"):
                for c in msg["citations"]:
                    st.markdown(
                        f"""
                        <div class="citation-card">
                            <span class="citation-badge">{c['document_name']}</span>
                            <strong>Page {c['page_number']}</strong> (Relevance: {c['relevance_score']:.2f})<br/>
                            <em>"{c['snippet']}"</em>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

# Chat Input
if prompt := st.chat_input("Ask a question about the uploaded documents..."):
    # Append user turn
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process query through RAG engine
    with st.chat_message("assistant"):
        with st.spinner("Searching documents & generating grounded response..."):
            query_response = rag_engine.query(
                question=prompt,
                session_id=st.session_state.session_id,
                top_k=settings.TOP_K_RETRIEVAL,
            )

            st.markdown(query_response.answer)

            # Display citations if grounded
            citations_data = [c.dict() for c in query_response.citations]
            if citations_data:
                with st.expander(f"📌 Source References ({len(citations_data)})"):
                    for c in citations_data:
                        st.markdown(
                            f"""
                            <div class="citation-card">
                                <span class="citation-badge">{c['document_name']}</span>
                                <strong>Page {c['page_number']}</strong> (Relevance: {c['relevance_score']:.2f})<br/>
                                <em>"{c['snippet']}"</em>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

            # Record assistant turn in state
            st.session_state.messages.append({
                "role": "assistant",
                "content": query_response.answer,
                "citations": citations_data,
            })
