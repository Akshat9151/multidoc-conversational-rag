import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from app.config import settings
from app.api.routes import router as api_router, vector_store, index_sample_documents

# Configure root logger
logging.basicConfig(
    level=logging.INFO if not settings.DEBUG else logging.DEBUG,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("rag-app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup & shutdown events."""
    logger.info(f"Starting {settings.PROJECT_NAME} v{settings.VERSION}")
    settings.ensure_directories()

    # Auto-index sample documents on first startup if vector store is empty
    if not vector_store.chunks:
        sample_files = list(settings.SAMPLE_DOCS_DIR.glob("*.pdf"))
        if sample_files:
            logger.info("Vector index is empty. Ingesting pre-bundled sample PDF documents...")
            try:
                index_sample_documents()
                logger.info(f"Auto-indexed {len(sample_files)} sample documents successfully.")
            except Exception as e:
                logger.warning(f"Could not auto-index sample documents at startup: {e}")

    yield
    logger.info("Shutting down RAG application.")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=settings.DESCRIPTION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for local Streamlit / React frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API Router
app.include_router(api_router)


@app.get("/", include_in_schema=False)
def root():
    """Redirect root path to interactive Swagger documentation."""
    return RedirectResponse(url="/docs")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)
