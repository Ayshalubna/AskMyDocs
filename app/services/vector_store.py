from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

import asyncio
import json
import time
from typing import List, Optional, Tuple
import structlog

from app.core.config import settings
from app.core.exceptions import NoDocumentsIndexedError, VectorStoreError

logger = structlog.get_logger(__name__)


class VectorStoreService:
    def __init__(self):
        self._store = None
        self._embeddings = None
        self._doc_metadata = {}
        self._lock = asyncio.Lock()
        self._persist_dir = str(settings.VECTOR_STORE_DIR)
        self._initialized = False

    async def initialize(self):
        logger.info("initializing_vector_store")

        self._embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL
        )

        # Load existing DB if exists
        try:
            self._store = Chroma(
                persist_directory=self._persist_dir,
                embedding_function=self._embeddings
            )
            logger.info("chroma_loaded")
        except Exception:
            self._store = None

        self._initialized = True

    async def add_documents(self, chunks: List[Document], doc_id: str, metadata: dict):
        if not chunks:
            return 0

        async with self._lock:
            try:
                if self._store is None:
                    self._store = Chroma.from_documents(
                        documents=chunks,
                        embedding=self._embeddings,
                        persist_directory=self._persist_dir
                    )
                else:
                    self._store.add_documents(chunks)

                self._store.persist()

                self._doc_metadata[doc_id] = {
                    "num_chunks": len(chunks),
                    **metadata
                }

                return len(chunks)

            except Exception as e:
                raise VectorStoreError("add_documents", str(e))

    async def similarity_search(
        self,
        query: str,
        top_k: int = 3,
        doc_ids: Optional[List[str]] = None,
        use_mmr: bool = False,
        score_threshold: float = 0.0,
    ) -> List[Tuple[Document, float]]:

        if self._store is None:
            raise NoDocumentsIndexedError()

        async with self._lock:
            try:
                results = self._store.similarity_search_with_relevance_scores(
                    query,
                    k=top_k
                )

                return results

            except Exception as e:
                raise VectorStoreError("search", str(e))

    async def delete_document(self, doc_id: str):
        # Simple version (Chroma handles deletion differently)
        if doc_id in self._doc_metadata:
            del self._doc_metadata[doc_id]
        return True

    def get_stats(self):
        return {
            "total_documents": len(self._doc_metadata),
            "has_index": self._store is not None
        }

    async def cleanup(self):
        if self._store:
            self._store.persist()
