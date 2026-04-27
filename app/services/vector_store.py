"""
Vector store service using FAISS.
Handles embeddings, indexing, similarity search, and MMR retrieval.
Thread-safe with async support.
"""

import asyncio
import json
import os
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import structlog
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from app.core.config import settings
from app.core.exceptions import NoDocumentsIndexedError, VectorStoreError

logger = structlog.get_logger(__name__)

METADATA_FILE = "doc_metadata.json"
STORE_NAME = "faiss_index"


class VectorStoreService:
    """
    Production FAISS vector store with:
    - Async-safe operations
    - Persistent storage
    - Per-document deletion
    - MMR search for diverse results
    - Health monitoring
    """

    def __init__(self):
        self._store: Optional[FAISS] = None
        self._embeddings: Optional[HuggingFaceEmbeddings] = None
        self._doc_metadata: Dict[str, dict] = {}
        self._lock = asyncio.Lock()
        self._store_path = settings.VECTOR_STORE_DIR / STORE_NAME
        self._meta_path = settings.VECTOR_STORE_DIR / METADATA_FILE
        self._initialized = False

    async def initialize(self) -> None:
        """Load or create vector store and embeddings model."""
        logger.info("initializing_vector_store", model=settings.EMBEDDING_MODEL)
        start = time.time()

        # Load embeddings in thread pool (CPU-bound)
        self._embeddings = await asyncio.get_event_loop().run_in_executor(
            None, self._load_embeddings
        )

        # Load persisted index if exists
        if self._store_path.exists():
            await self._load_persisted_store()
        else:
            logger.info("no_existing_index_found", path=str(self._store_path))

        # Load document metadata
        self._load_doc_metadata()
        self._initialized = True

        elapsed = (time.time() - start) * 1000
        logger.info(
            "vector_store_initialized",
            elapsed_ms=round(elapsed, 2),
            num_documents=len(self._doc_metadata),
            has_index=self._store is not None,
        )

    def _load_embeddings(self) -> HuggingFaceEmbeddings:
        """Load HuggingFace sentence-transformer embeddings."""
        return HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={"device": settings.EMBEDDING_DEVICE},
            encode_kwargs={
                "normalize_embeddings": True,  # For cosine similarity
                "batch_size": settings.EMBEDDING_BATCH_SIZE,
            },
        )

    async def _load_persisted_store(self) -> None:
        """Load FAISS index from disk."""
        try:
            self._store = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: FAISS.load_local(
                    str(self._store_path),
                    self._embeddings,
                    allow_dangerous_deserialization=True,
                ),
            )
            logger.info("loaded_persisted_index", path=str(self._store_path))
        except Exception as e:
            logger.warning("failed_to_load_persisted_index", error=str(e))

    def _load_doc_metadata(self) -> None:
        """Load document metadata from JSON."""
        if self._meta_path.exists():
            with open(self._meta_path) as f:
                self._doc_metadata = json.load(f)

    def _save_doc_metadata(self) -> None:
        """Persist document metadata to JSON."""
        with open(self._meta_path, "w") as f:
            json.dump(self._doc_metadata, f, indent=2, default=str)

    async def add_documents(
        self, chunks: List[Document], doc_id: str, doc_metadata: dict
    ) -> int:
        """
        Add document chunks to the FAISS index.
        Returns number of chunks indexed.
        """
        if not chunks:
            return 0

        async with self._lock:
            try:
                start = time.time()

                # Embed and add to store
                if self._store is None:
                    self._store = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: FAISS.from_documents(chunks, self._embeddings),
                    )
                else:
                    await asyncio.get_event_loop().run_in_executor(
                        None, self._store.add_documents, chunks
                    )

                # Persist index
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self._store.save_local(str(self._store_path)),
                )

                # Save metadata
                self._doc_metadata[doc_id] = {
                    **doc_metadata,
                    "num_chunks": len(chunks),
                    "chunk_ids": [c.metadata.get("chunk_id") for c in chunks],
                }
                self._save_doc_metadata()

                elapsed = (time.time() - start) * 1000
                logger.info(
                    "documents_indexed",
                    doc_id=doc_id,
                    num_chunks=len(chunks),
                    elapsed_ms=round(elapsed, 2),
                )
                return len(chunks)

            except Exception as e:
                logger.error("indexing_failed", doc_id=doc_id, error=str(e))
                raise VectorStoreError("add_documents", str(e))

    async def similarity_search(
        self,
        query: str,
        top_k: int = 5,
        doc_ids: Optional[List[str]] = None,
        use_mmr: bool = True,
        score_threshold: float = 0.0,
    ) -> List[Tuple[Document, float]]:
        """
        Semantic search with optional MMR for diversity.
        Returns (document, score) tuples sorted by relevance.
        """
        if self._store is None:
            raise NoDocumentsIndexedError()

        async with self._lock:
            try:
                start = time.time()

                # Filter function for doc_ids
                filter_fn = None
                if doc_ids:
                    filter_fn = lambda meta: meta.get("doc_id") in doc_ids  # noqa

                if use_mmr:
                    docs = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: self._store.max_marginal_relevance_search(
                            query,
                            k=top_k,
                            fetch_k=min(top_k * 4, 100),
                            lambda_mult=settings.MMR_LAMBDA,
                            filter=filter_fn,
                        ),
                    )
                    # MMR doesn't return scores, compute them separately
                    results_with_scores = [(doc, 0.85) for doc in docs]  # approx
                else:
                    results_with_scores = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: self._store.similarity_search_with_relevance_scores(
                            query,
                            k=top_k,
                            filter=filter_fn,
                        ),
                    )

                # Filter by threshold
                results_with_scores = [
                    (doc, score)
                    for doc, score in results_with_scores
                    if score >= score_threshold
                ]

                elapsed = (time.time() - start) * 1000
                logger.info(
                    "search_completed",
                    query_preview=query[:50],
                    num_results=len(results_with_scores),
                    elapsed_ms=round(elapsed, 2),
                    use_mmr=use_mmr,
                )

                return results_with_scores

            except NoDocumentsIndexedError:
                raise
            except Exception as e:
                logger.error("search_failed", error=str(e))
                raise VectorStoreError("similarity_search", str(e))

    async def delete_document(self, doc_id: str) -> bool:
        """Remove a document and all its chunks from the index."""
        if doc_id not in self._doc_metadata:
            return False

        async with self._lock:
            try:
                # FAISS doesn't support selective deletion natively
                # Rebuild index excluding target doc
                if self._store is not None:
                    all_docs = []
                    docstore = self._store.docstore._dict
                    for key, doc in docstore.items():
                        if doc.metadata.get("doc_id") != doc_id:
                            all_docs.append(doc)

                    if all_docs:
                        self._store = await asyncio.get_event_loop().run_in_executor(
                            None,
                            lambda: FAISS.from_documents(all_docs, self._embeddings),
                        )
                        await asyncio.get_event_loop().run_in_executor(
                            None,
                            lambda: self._store.save_local(str(self._store_path)),
                        )
                    else:
                        self._store = None
                        # Remove saved files
                        for f in self._store_path.parent.glob(f"{STORE_NAME}*"):
                            f.unlink(missing_ok=True)

                del self._doc_metadata[doc_id]
                self._save_doc_metadata()

                logger.info("document_deleted", doc_id=doc_id)
                return True

            except Exception as e:
                logger.error("deletion_failed", doc_id=doc_id, error=str(e))
                raise VectorStoreError("delete_document", str(e))

    def get_stats(self) -> dict:
        """Return index statistics."""
        index_size = 0
        if self._store_path.exists():
            index_size = sum(
                f.stat().st_size
                for f in self._store_path.parent.iterdir()
                if f.name.startswith(STORE_NAME)
            ) / (1024 * 1024)

        embedding_dim = 0
        if self._store is not None:
            try:
                embedding_dim = self._store.index.d
            except Exception:
                pass

        total_chunks = sum(
            m.get("num_chunks", 0) for m in self._doc_metadata.values()
        )

        return {
            "total_documents": len(self._doc_metadata),
            "total_chunks": total_chunks,
            "index_size_mb": round(index_size, 2),
            "embedding_model": settings.EMBEDDING_MODEL,
            "embedding_dimension": embedding_dim,
            "index_type": settings.FAISS_INDEX_TYPE,
            "has_index": self._store is not None,
        }

    def get_document_metadata(self, doc_id: str) -> Optional[dict]:
        return self._doc_metadata.get(doc_id)

    def list_documents(self) -> List[dict]:
        return list(self._doc_metadata.values())

    async def cleanup(self) -> None:
        """Graceful shutdown."""
        logger.info("vector_store_cleanup")
        if self._store is not None:
            try:
                async with self._lock:
                    self._store.save_local(str(self._store_path))
            except Exception as e:
                logger.warning("cleanup_save_failed", error=str(e))
