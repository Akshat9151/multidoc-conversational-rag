import logging
from typing import List, Dict, Any
from app.core.pdf_loader import DocumentPage

logger = logging.getLogger(__name__)

class DocumentChunk:
    """Represents a text chunk ready for vector embedding and retrieval."""
    def __init__(self, text: str, metadata: Dict[str, Any]):
        self.text = text
        self.metadata = metadata

    def __repr__(self):
        return (
            f"<DocumentChunk doc='{self.metadata.get('document_name')}' "
            f"page={self.metadata.get('page_number')} chunk={self.metadata.get('chunk_index')}>"
        )

class RecursiveTextSplitter:
    """
    Recursively splits text on natural boundaries:
    paragraphs (\\n\\n), sentences (\\n, . ), and words ( ),
    ensuring chunk overlap without breaking semantic flow.
    """

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 150,
        separators: List[str] = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", "? ", "! ", " ", ""]

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        final_chunks: List[str] = []
        separator = separators[-1]
        new_separators = []

        for i, sep in enumerate(separators):
            if sep == "" or sep in text:
                separator = sep
                new_separators = separators[i + 1:]
                break

        splits = text.split(separator) if separator else list(text)

        good_splits: List[str] = []
        for s in splits:
            if len(s) < self.chunk_size:
                good_splits.append(s)
            else:
                if good_splits:
                    merged = self._merge_splits(good_splits, separator)
                    final_chunks.extend(merged)
                    good_splits = []
                if not new_separators:
                    final_chunks.append(s)
                else:
                    recursive_splits = self._split_text(s, new_separators)
                    final_chunks.extend(recursive_splits)

        if good_splits:
            merged = self._merge_splits(good_splits, separator)
            final_chunks.extend(merged)

        return final_chunks

    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        docs: List[str] = []
        current_doc: List[str] = []
        total_len = 0

        for d in splits:
            d_len = len(d)
            if total_len + d_len + (len(separator) if current_doc else 0) > self.chunk_size:
                if total_len > 0:
                    doc = separator.join(current_doc).strip()
                    if doc:
                        docs.append(doc)

                    # Handle overlap by keeping the trailing end of splits
                    while total_len > self.chunk_overlap or (
                        total_len + d_len + len(separator) > self.chunk_size and total_len > 0
                    ):
                        removed = current_doc.pop(0)
                        total_len -= len(removed) + (len(separator) if current_doc else 0)

            current_doc.append(d)
            total_len += d_len + (len(separator) if len(current_doc) > 1 else 0)

        if current_doc:
            doc = separator.join(current_doc).strip()
            if doc:
                docs.append(doc)

        return docs

    def split_pages(self, pages: List[DocumentPage]) -> List[DocumentChunk]:
        """
        Splits a list of DocumentPages into DocumentChunks,
        propagating and enriching metadata for each chunk.
        """
        all_chunks: List[DocumentChunk] = []

        for page in pages:
            raw_text = page.text
            if not raw_text.strip():
                continue

            text_splits = self._split_text(raw_text, self.separators)

            for chunk_idx, chunk_text in enumerate(text_splits):
                cleaned_chunk = chunk_text.strip()
                if len(cleaned_chunk) < 30:  # Skip tiny fragments
                    continue

                chunk_meta = dict(page.metadata)
                chunk_meta["chunk_index"] = chunk_idx + 1
                chunk_meta["chunk_id"] = (
                    f"{chunk_meta['document_name']}_p{chunk_meta['page_number']}_c{chunk_idx+1}"
                )
                chunk_meta["char_count"] = len(cleaned_chunk)

                all_chunks.append(DocumentChunk(text=cleaned_chunk, metadata=chunk_meta))

        logger.info(
            f"Split {len(pages)} pages into {len(all_chunks)} chunks "
            f"(chunk_size={self.chunk_size}, overlap={self.chunk_overlap})."
        )
        return all_chunks
