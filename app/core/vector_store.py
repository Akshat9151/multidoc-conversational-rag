import os
import json
import pickle
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

from app.core.text_splitter import DocumentChunk
from app.config import settings

logger = logging.getLogger(__name__)

class EmbeddingEngine:
    """Handles text embeddings with SentenceTransformers, with graceful CPU fallback."""
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self._fallback_mode = False

    def _load_model(self):
        if self._model is None and not self._fallback_mode:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Loading embedding model: {self.model_name}")
                self._model = SentenceTransformer(self.model_name)
            except Exception as e:
                logger.warning(
                    f"SentenceTransformer not available or failed to load ({e}). "
                    f"Engaging fallback TF-IDF/Hashing embedding engine."
                )
                self._fallback_mode = True

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embeds a batch of texts into normalized 2D numpy array."""
        self._load_model()

        if self._model is not None:
            embeddings = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            # L2 normalize for cosine similarity
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1e-10
            return (embeddings / norms).astype(np.float32)

        # Fallback pseudo-embedding (Bag of words / hash vectorizer)
        dim = 384
        vectors = []
        for t in texts:
            vec = np.zeros(dim, dtype=np.float32)
            words = t.lower().split()
            for w in words:
                idx = hash(w) % dim
                vec[idx] += 1.0
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            vectors.append(vec)
        return np.array(vectors, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Embeds a single query string into normalized 1D vector."""
        return self.embed_texts([query])[0]


class VectorStoreManager:
    """
    Manages local FAISS index persistence, document chunk storage,
    and cosine similarity search with score thresholding.
    """

    def __init__(self, storage_dir: Optional[Path] = None, embedding_engine: Optional[EmbeddingEngine] = None):
        self.storage_dir = storage_dir or settings.VECTOR_STORE_DIR
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_engine = embedding_engine or EmbeddingEngine(settings.EMBEDDING_MODEL_NAME)

        self.index_file = self.storage_dir / "index.faiss"
        self.metadata_file = self.storage_dir / "chunks.pkl"
        self.summary_file = self.storage_dir / "docs_summary.json"

        self.chunks: List[DocumentChunk] = []
        self.faiss_index = None
        self._use_faiss = True

        self._init_backend()
        self.load()

    def _init_backend(self):
        try:
            import faiss
            self.faiss = faiss
            self._use_faiss = True
        except ImportError:
            logger.warning("FAISS not installed. Falling back to native NumPy cosine similarity index.")
            self._use_faiss = False

    def build_from_chunks(self, chunks: List[DocumentChunk]):
        """Builds a new index from scratch with the provided chunks."""
        self.chunks = chunks
        if not chunks:
            logger.warning("No chunks provided to build vector index.")
            return

        texts = [c.text for c in chunks]
        embeddings = self.embedding_engine.embed_texts(texts)
        dim = embeddings.shape[1]

        if self._use_faiss:
            # Inner Product with normalized vectors == Cosine Similarity
            self.faiss_index = self.faiss.IndexFlatIP(dim)
            self.faiss_index.add(embeddings)
        else:
            self._numpy_embeddings = embeddings

        self.save()
        logger.info(f"Vector store successfully built and indexed {len(chunks)} chunks.")

    def add_chunks(self, new_chunks: List[DocumentChunk]):
        """Appends new chunks to the existing index."""
        if not new_chunks:
            return

        new_texts = [c.text for c in new_chunks]
        new_embeddings = self.embedding_engine.embed_texts(new_texts)

        if self.chunks:
            self.chunks.extend(new_chunks)
            if self._use_faiss and self.faiss_index is not None:
                self.faiss_index.add(new_embeddings)
            else:
                if hasattr(self, "_numpy_embeddings"):
                    self._numpy_embeddings = np.vstack([self._numpy_embeddings, new_embeddings])
                else:
                    self._numpy_embeddings = new_embeddings
        else:
            self.build_from_chunks(new_chunks)
            return

        self.save()

    def similarity_search_with_score(
        self, query: str, top_k: int = 4, filter_doc: Optional[str] = None
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Retrieves top_k chunks most relevant to query.
        Returns list of (DocumentChunk, relevance_score) where score in [0.0, 1.0].
        """
        if not self.chunks:
            logger.warning("Similarity search called on an empty vector store.")
            return []

        query_vec = self.embedding_engine.embed_query(query)

        results: List[Tuple[DocumentChunk, float]] = []

        if self._use_faiss and self.faiss_index is not None:
            # Query FAISS
            k = min(len(self.chunks), top_k * 3 if filter_doc else top_k)
            scores, indices = self.faiss_index.search(
                np.expand_dims(query_vec, axis=0), k
            )
            scores = scores[0]
            indices = indices[0]

            for idx, score in zip(indices, scores):
                if idx < 0 or idx >= len(self.chunks):
                    continue
                chunk = self.chunks[idx]
                if filter_doc and chunk.metadata.get("document_name") != filter_doc:
                    continue

                # Normalizing inner product score to [0, 1] range
                norm_score = float(max(0.0, min(1.0, (score + 1.0) / 2.0)))
                results.append((chunk, norm_score))

                if len(results) >= top_k:
                    break
        else:
            # Native NumPy cosine similarity
            if hasattr(self, "_numpy_embeddings"):
                scores = np.dot(self._numpy_embeddings, query_vec)
                top_indices = np.argsort(scores)[::-1]
                for idx in top_indices:
                    chunk = self.chunks[idx]
                    if filter_doc and chunk.metadata.get("document_name") != filter_doc:
                        continue
                    norm_score = float(max(0.0, min(1.0, (scores[idx] + 1.0) / 2.0)))
                    results.append((chunk, norm_score))
                    if len(results) >= top_k:
                        break

        return results

    def save(self):
        """Persists the FAISS index and chunk metadata to disk."""
        try:
            if self._use_faiss and self.faiss_index is not None:
                self.faiss.write_index(self.faiss_index, str(self.index_file))
            elif hasattr(self, "_numpy_embeddings"):
                np.save(str(self.storage_dir / "embeddings.npy"), self._numpy_embeddings)

            with open(self.metadata_file, "wb") as f:
                pickle.dump(self.chunks, f)

            # Save summary JSON for fast document introspection
            docs_summary = self._compute_docs_summary()
            with open(self.summary_file, "w", encoding="utf-8") as f:
                json.dump(docs_summary, f, indent=2)

            logger.info(f"Vector store saved to {self.storage_dir}")
        except Exception as e:
            logger.error(f"Failed to persist vector store: {e}")

    def load(self):
        """Loads index and metadata from disk if present."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, "rb") as f:
                    self.chunks = pickle.load(f)

                if self._use_faiss and self.index_file.exists():
                    self.faiss_index = self.faiss.read_index(str(self.index_file))
                    logger.info(f"Loaded FAISS index with {len(self.chunks)} chunks.")
                else:
                    npy_path = self.storage_dir / "embeddings.npy"
                    if npy_path.exists():
                        self._numpy_embeddings = np.load(str(npy_path))
                        logger.info(f"Loaded NumPy embeddings index with {len(self.chunks)} chunks.")
            except Exception as e:
                logger.error(f"Failed to load vector store from disk: {e}")

    def _compute_docs_summary(self) -> Dict[str, Any]:
        doc_map = {}
        for c in self.chunks:
            d_name = c.metadata.get("document_name", "unknown")
            page = c.metadata.get("page_number", 1)
            if d_name not in doc_map:
                doc_map[d_name] = {"pages": set(), "chunks_count": 0}
            doc_map[d_name]["pages"].add(page)
            doc_map[d_name]["chunks_count"] += 1

        summary = []
        for name, data in doc_map.items():
            summary.append({
                "document_name": name,
                "total_pages": len(data["pages"]),
                "total_chunks": data["chunks_count"]
            })
        return {"documents": summary, "total_chunks": len(self.chunks)}

    def get_indexed_documents(self) -> List[Dict[str, Any]]:
        """Returns metadata list of all indexed documents."""
        if self.summary_file.exists():
            try:
                with open(self.summary_file, "r", encoding="utf-8") as f:
                    return json.load(f).get("documents", [])
            except Exception:
                pass
        return self._compute_docs_summary().get("documents", [])
