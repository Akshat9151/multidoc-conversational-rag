import shutil
import logging
from pathlib import Path
from typing import List
from datetime import datetime
from fastapi import APIRouter, UploadFile, File, HTTPException, status, Query

from app.models.schemas import (
    QueryRequest,
    QueryResponse,
    UploadResponse,
    DocumentSummary,
    ChatHistoryResponse,
    ClearSessionResponse,
)
from app.core.pdf_loader import PDFDocumentLoader
from app.core.text_splitter import RecursiveTextSplitter
from app.core.vector_store import VectorStoreManager
from app.core.rag_engine import RAGEngine
from app.core.memory import memory_manager
from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Conversational RAG"])

# Singleton components
pdf_loader = PDFDocumentLoader()
text_splitter = RecursiveTextSplitter(
    chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP
)
vector_store = VectorStoreManager()
rag_engine = RAGEngine(vector_store=vector_store, memory=memory_manager)


@router.post(
    "/documents/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and index multiple PDF documents",
)
async def upload_documents(files: List[UploadFile] = File(...)):
    """
    Accepts multiple PDF files, parses text per page, chunks with overlap,
    generates vector embeddings, and updates the persistent FAISS index.
    """
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="No files uploaded."
        )

    saved_paths: List[Path] = []
    summaries: List[DocumentSummary] = []

    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid file '{file.filename}'. Only PDF format is supported.",
            )

        destination = settings.UPLOAD_DIR / file.filename
        try:
            with open(destination, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            saved_paths.append(destination)
        except Exception as e:
            logger.error(f"Failed to write uploaded file '{file.filename}': {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Could not save file '{file.filename}'.",
            )

    try:
        # Extract pages
        all_pages = pdf_loader.load_multiple_pdfs(saved_paths)
        if not all_pages:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Uploaded PDFs did not contain any readable text.",
            )

        # Chunk pages
        chunks = text_splitter.split_pages(all_pages)

        # Add to vector store
        vector_store.add_chunks(chunks)

        # Build document summaries
        for path in saved_paths:
            file_chunks = [c for c in chunks if c.metadata.get("document_name") == path.name]
            file_pages = {c.metadata.get("page_number") for c in file_chunks}
            size_kb = round(path.stat().st_size / 1024.0, 2)

            summaries.append(
                DocumentSummary(
                    document_name=path.name,
                    total_pages=len(file_pages) or 1,
                    total_chunks=len(file_chunks),
                    file_size_kb=size_kb,
                    indexed_at=datetime.utcnow().isoformat(),
                )
            )

        return UploadResponse(
            message="Successfully processed and indexed uploaded documents.",
            documents_processed=len(saved_paths),
            total_chunks_added=len(chunks),
            documents=summaries,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during document ingestion: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process documents: {str(e)}",
        )


@router.post(
    "/documents/index-samples",
    response_model=UploadResponse,
    summary="Index pre-bundled sample PDF documents",
)
def index_sample_documents():
    """
    Scans the data/sample_docs directory and indexes all included sample PDFs.
    Allows instant testing without manual file uploading.
    """
    sample_files = list(settings.SAMPLE_DOCS_DIR.glob("*.pdf"))
    if not sample_files:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No sample PDFs found in data/sample_docs/.",
        )

    pages = pdf_loader.load_multiple_pdfs(sample_files)
    chunks = text_splitter.split_pages(pages)
    vector_store.build_from_chunks(chunks)

    summaries = []
    for path in sample_files:
        file_chunks = [c for c in chunks if c.metadata.get("document_name") == path.name]
        file_pages = {c.metadata.get("page_number") for c in file_chunks}
        summaries.append(
            DocumentSummary(
                document_name=path.name,
                total_pages=len(file_pages) or 1,
                total_chunks=len(file_chunks),
                file_size_kb=round(path.stat().st_size / 1024.0, 2),
                indexed_at=datetime.utcnow().isoformat(),
            )
        )

    return UploadResponse(
        message=f"Successfully indexed {len(sample_files)} sample documents.",
        documents_processed=len(sample_files),
        total_chunks_added=len(chunks),
        documents=summaries,
    )


@router.post(
    "/chat",
    response_model=QueryResponse,
    summary="Ask a question with conversational memory & citations",
)
def chat_with_documents(payload: QueryRequest):
    """
    Processes natural language queries against indexed documents.
    Preserves multi-turn conversational memory, executes contextual query rewriting,
    retrieves grounded passages, and includes explicit document & page citations.
    """
    try:
        response = rag_engine.query(
            question=payload.question,
            session_id=payload.session_id,
            top_k=payload.top_k,
        )
        return response
    except Exception as e:
        logger.error(f"Error handling query: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query execution failed: {str(e)}",
        )


@router.get(
    "/chat/history/{session_id}",
    response_model=ChatHistoryResponse,
    summary="Get multi-turn conversational history for a session",
)
def get_chat_history(session_id: str):
    """Returns the full chronological message history for a given session."""
    history = memory_manager.get_history(session_id)
    return ChatHistoryResponse(
        session_id=session_id,
        total_turns=len(history) // 2,
        history=history,
    )


@router.delete(
    "/chat/session/{session_id}",
    response_model=ClearSessionResponse,
    summary="Reset conversational memory for a session",
)
def clear_chat_session(session_id: str):
    """Clears history for the specified session ID."""
    cleared = memory_manager.clear_session(session_id)
    return ClearSessionResponse(
        session_id=session_id,
        cleared=cleared,
        message="Session memory cleared successfully." if cleared else "Session ID not found.",
    )


@router.get(
    "/documents",
    summary="List all currently indexed documents in vector store",
)
def list_indexed_documents():
    """Returns metadata for all documents currently indexed in FAISS."""
    return {
        "status": "success",
        "total_documents": len(vector_store.get_indexed_documents()),
        "total_indexed_chunks": len(vector_store.chunks),
        "documents": vector_store.get_indexed_documents(),
    }


@router.get("/health", summary="Service health check")
def health_check():
    """Returns operational status of the RAG backend."""
    return {
        "status": "healthy",
        "project": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "llm_provider": settings.LLM_PROVIDER,
        "indexed_chunks": len(vector_store.chunks),
        "embedding_model": settings.EMBEDDING_MODEL_NAME,
    }
