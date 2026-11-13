from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import datetime

class Citation(BaseModel):
    """Citation referencing a specific chunk from an ingested document."""
    document_name: str = Field(..., description="Name of the source document")
    page_number: int = Field(..., description="1-indexed page number in the source PDF")
    snippet: str = Field(..., description="Relevant text excerpt retrieved from the document")
    relevance_score: float = Field(..., description="Cosine similarity score (0.0 to 1.0)")

class QueryRequest(BaseModel):
    """Request payload for querying the RAG system."""
    question: str = Field(..., min_length=2, description="Natural language question")
    session_id: Optional[str] = Field(
        None, description="Conversational session ID for multi-turn memory. Auto-created if omitted."
    )
    top_k: Optional[int] = Field(
        None, ge=1, le=10, description="Override default number of chunks retrieved"
    )

class QueryResponse(BaseModel):
    """Grounded response returned by the RAG system."""
    answer: str = Field(..., description="Generated answer grounded in retrieved documents")
    session_id: str = Field(..., description="Active session ID")
    grounded: bool = Field(
        ..., description="True if answer is directly derived from documents; False if info was unavailable"
    )
    citations: List[Citation] = Field(default_factory=list, description="List of source citations")
    rewritten_query: Optional[str] = Field(
        None, description="Contextually rewritten query for follow-up questions"
    )
    latency_ms: float = Field(..., description="Total processing time in milliseconds")

class DocumentSummary(BaseModel):
    """Metadata summary for an ingested document."""
    document_name: str
    total_pages: int
    total_chunks: int
    file_size_kb: float
    indexed_at: str

class UploadResponse(BaseModel):
    """Response returned after processing uploaded documents."""
    message: str
    documents_processed: int
    total_chunks_added: int
    documents: List[DocumentSummary]

class ChatMessage(BaseModel):
    """Single chat turn in conversation history."""
    role: str = Field(..., description="'user' or 'assistant'")
    content: str
    timestamp: str
    citations: Optional[List[Citation]] = None

class ChatHistoryResponse(BaseModel):
    """Full conversational memory transcript for a session."""
    session_id: str
    total_turns: int
    history: List[ChatMessage]

class ClearSessionResponse(BaseModel):
    """Acknowledgment for clearing a conversation session."""
    session_id: str
    cleared: bool
    message: str
