import os
import pytest
from pathlib import Path

from app.core.pdf_loader import PDFDocumentLoader
from app.core.text_splitter import RecursiveTextSplitter
from app.core.vector_store import VectorStoreManager
from app.core.memory import ConversationMemoryManager
from app.core.rag_engine import RAGEngine
from app.config import settings

SAMPLE_DOCS_DIR = Path(__file__).resolve().parent.parent / "data" / "sample_docs"

@pytest.fixture(scope="module")
def sample_pdf_path():
    pdf = SAMPLE_DOCS_DIR / "acme_financial_annual_report_2025.pdf"
    if not pdf.exists():
        pytest.skip(f"Sample PDF not generated at {pdf}")
    return pdf

@pytest.fixture(scope="module")
def loaded_pages(sample_pdf_path):
    loader = PDFDocumentLoader()
    return loader.load_pdf(sample_pdf_path)

@pytest.fixture(scope="module")
def chunked_docs(loaded_pages):
    splitter = RecursiveTextSplitter(chunk_size=500, chunk_overlap=100)
    return splitter.split_pages(loaded_pages)


class TestPDFIngestion:
    def test_pdf_extraction(self, loaded_pages):
        assert len(loaded_pages) >= 1
        page1 = loaded_pages[0]
        assert "Acme Global Technologies" in page1.text
        assert page1.metadata["document_name"] == "acme_financial_annual_report_2025.pdf"
        assert page1.metadata["page_number"] == 1

    def test_recursive_chunking(self, chunked_docs):
        assert len(chunked_docs) > 0
        for chunk in chunked_docs:
            assert len(chunk.text) > 20
            assert "document_name" in chunk.metadata
            assert "page_number" in chunk.metadata
            assert "chunk_id" in chunk.metadata


class TestVectorStoreAndRetrieval:
    def test_indexing_and_similarity_search(self, chunked_docs, tmp_path):
        vstore = VectorStoreManager(storage_dir=tmp_path)
        vstore.build_from_chunks(chunked_docs)

        # Search for financial revenue
        results = vstore.similarity_search_with_score("What was Acme's consolidated revenue?", top_k=2)
        assert len(results) > 0
        top_chunk, score = results[0]
        assert score > 0.0
        assert "revenue" in top_chunk.text.lower() or "485" in top_chunk.text


class TestConversationalMemory:
    def test_multi_turn_history(self):
        mem = ConversationMemoryManager(max_history_turns=3)
        sid = mem.get_or_create_session("test_session_1")

        mem.add_user_message(sid, "What was the EBITDA margin?")
        mem.add_assistant_message(sid, "The EBITDA margin was 24.5%.")
        mem.add_user_message(sid, "And what about Net Income?")
        mem.add_assistant_message(sid, "Net Income was $72.8 million.")

        history = mem.get_history(sid)
        assert len(history) == 4
        assert history[0].role == "user"
        assert history[1].role == "assistant"

        prompt_str = mem.format_history_for_prompt(sid)
        assert "User: What was the EBITDA margin?" in prompt_str
        assert "Assistant: Net Income was $72.8 million." in prompt_str

        mem.clear_session(sid)
        assert len(mem.get_history(sid)) == 0


class TestRAGEnginePipeline:
    def test_grounded_vs_unavailable_query(self, chunked_docs, tmp_path):
        vstore = VectorStoreManager(storage_dir=tmp_path)
        vstore.build_from_chunks(chunked_docs)
        engine = RAGEngine(vector_store=vstore)

        # Test 1: Grounded question
        resp1 = engine.query("What was full-year revenue?", session_id="test_rag_session")
        assert resp1.grounded is True
        assert len(resp1.citations) > 0
        assert resp1.citations[0].document_name == "acme_financial_annual_report_2025.pdf"

        # Test 2: Completely unrelated question (unavailable info guardrail)
        resp2 = engine.query(
            "What is the average temperature on Neptune during winter?",
            session_id="test_rag_session",
        )
        assert resp2.grounded is False
        assert "could not find information" in resp2.answer.lower()
