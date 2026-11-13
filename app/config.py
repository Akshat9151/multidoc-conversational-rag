import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Base directory of the repository
BASE_DIR = Path(__file__).resolve().parent.parent

# Load .env file if available
load_dotenv(BASE_DIR / ".env")

class Settings:
    """Application configuration settings."""
    PROJECT_NAME: str = "Intelligent Multi-Document Conversational RAG"
    VERSION: str = "1.0.0"
    DESCRIPTION: str = (
        "Enterprise-ready RAG system supporting multi-PDF ingestion, "
        "conversational memory, grounded answer generation, and precise source citations."
    )

    # LLM Settings
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "gemini").lower()
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")
    GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gemini-1.5-flash")
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.2"))
    MAX_OUTPUT_TOKENS: int = int(os.getenv("MAX_OUTPUT_TOKENS", "1024"))

    # Embeddings
    EMBEDDING_MODEL_NAME: str = os.getenv(
        "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
    )

    # Retrieval & Chunking
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "800"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    TOP_K_RETRIEVAL: int = int(os.getenv("TOP_K_RETRIEVAL", "4"))
    SIMILARITY_SCORE_THRESHOLD: float = float(
        os.getenv("SIMILARITY_SCORE_THRESHOLD", "0.30")
    )

    # Storage Paths
    VECTOR_STORE_DIR: Path = BASE_DIR / os.getenv("VECTOR_STORE_DIR", "data/faiss_index")
    UPLOAD_DIR: Path = BASE_DIR / os.getenv("UPLOAD_DIR", "data/uploaded_docs")
    SAMPLE_DOCS_DIR: Path = BASE_DIR / os.getenv("SAMPLE_DOCS_DIR", "data/sample_docs")

    # Server
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    DEBUG: bool = os.getenv("DEBUG", "False").lower() in ("true", "1", "yes")

    def ensure_directories(self):
        """Ensure necessary operational directories exist."""
        self.VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
        self.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        self.SAMPLE_DOCS_DIR.mkdir(parents=True, exist_ok=True)

settings = Settings()
settings.ensure_directories()
