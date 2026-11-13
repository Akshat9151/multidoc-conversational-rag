import logging
from pathlib import Path
from typing import List, Dict, Any, Union
from pypdf import PdfReader

logger = logging.getLogger(__name__)

class DocumentPage:
    """Represents a single extracted page from a PDF with attached metadata."""
    def __init__(self, text: str, metadata: Dict[str, Any]):
        self.text = text
        self.metadata = metadata

    def __repr__(self):
        return f"<DocumentPage doc='{self.metadata.get('document_name')}' page={self.metadata.get('page_number')}>"

class PDFDocumentLoader:
    """
    Robust multi-document PDF extractor.
    Extracts text per page while preserving document origin and 1-indexed page numbering.
    """

    @staticmethod
    def clean_text(text: str) -> str:
        """Normalize line breaks and excessive whitespace while preserving paragraph structure."""
        if not text:
            return ""
        # Remove null characters and standard normalize
        text = text.replace("\x00", "")
        lines = [line.strip() for line in text.splitlines()]
        # Filter out empty blank lines but preserve paragraph separation
        cleaned = "\n".join([l for l in lines if l])
        return cleaned

    def load_pdf(self, file_path: Union[str, Path]) -> List[DocumentPage]:
        """
        Loads a single PDF and extracts pages as DocumentPage objects.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {path}")

        pages: List[DocumentPage] = []
        doc_name = path.name

        try:
            reader = PdfReader(str(path))
            total_pages = len(reader.pages)

            for idx, page in enumerate(reader.pages):
                page_num = idx + 1
                try:
                    raw_text = page.extract_text() or ""
                    cleaned = self.clean_text(raw_text)

                    if not cleaned.strip():
                        logger.warning(f"Page {page_num} of '{doc_name}' has no extractable text.")
                        continue

                    metadata = {
                        "document_name": doc_name,
                        "file_path": str(path.resolve()),
                        "page_number": page_num,
                        "total_pages": total_pages,
                    }
                    pages.append(DocumentPage(text=cleaned, metadata=metadata))

                except Exception as page_err:
                    logger.error(f"Error extracting page {page_num} of {doc_name}: {page_err}")
                    continue

            logger.info(f"Loaded '{doc_name}' with {len(pages)}/{total_pages} text-containing pages.")
            return pages

        except Exception as e:
            logger.error(f"Failed to read PDF '{doc_name}': {e}")
            raise RuntimeError(f"Could not parse PDF '{doc_name}': {str(e)}")

    def load_multiple_pdfs(self, file_paths: List[Union[str, Path]]) -> List[DocumentPage]:
        """Loads and combines extracted pages from multiple PDF files."""
        all_pages: List[DocumentPage] = []
        for path in file_paths:
            all_pages.extend(self.load_pdf(path))
        return all_pages
