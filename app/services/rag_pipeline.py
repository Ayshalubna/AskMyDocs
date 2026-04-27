"""
RAG pipeline orchestrator.
Wires together: vector search → LLM generation → response formatting.
"""

import time
from typing import AsyncGenerator, List, Optional, Tuple

import structlog
from langchain.schema import Document

from app.core.config import settings
from app.core.exceptions import NoDocumentsIndexedError
from app.models.schemas import QueryRequest, QueryResponse, SourceChunk
from app.services.llm_service import LLMService
from app.services.session_service import get_session_service
from app.services.vector_store import VectorStoreService

logger = structlog.get_logger(__name__)


class RAGPipeline:
    """
    End-to-end RAG pipeline:
    1. Retrieve relevant chunks via FAISS
    2. Rerank / filter by score threshold
    3. Generate answer via LLM with context
    4. Persist to session history
    """

    def __init__(self, vector_service: VectorStoreService):
        self.vector_service = vector_service
        self.llm_service = LLMService()
        self.session_service = get_session_service()

    async def initialize(self) -> None:
        await self.llm_service.initialize()
        await self.session_service.start()

    async def query(self, request: QueryRequest) -> QueryResponse:
        """Full RAG query pipeline."""
        start = time.time()

        # Session management
        session_id = self.session_service.get_or_create(request.session_id)
        history = self.session_service.get_history(session_id)

        # Retrieval
        docs_with_scores = await self.vector_service.similarity_search(
            query=request.question,
            top_k=request.top_k,
            doc_ids=request.doc_ids,
            use_mmr=request.use_mmr,
            score_threshold=settings.SIMILARITY_THRESHOLD,
        )

        # Generate
        answer, llm_meta = await self.llm_service.generate_answer(
            question=request.question,
            docs_with_scores=docs_with_scores,
            history=history,
        )

        # Format sources
        sources = []
        if request.include_sources:
            sources = [
                SourceChunk(
                    content=doc.page_content[:500],
                    score=round(score, 4),
                    doc_id=doc.metadata.get("doc_id", ""),
                    filename=doc.metadata.get("filename", ""),
                    chunk_index=doc.metadata.get("chunk_index", 0),
                    page_number=doc.metadata.get("page_number"),
                    metadata={
                        k: v
                        for k, v in doc.metadata.items()
                        if k not in ("doc_id", "filename", "chunk_index", "page_number")
                    },
                )
                for doc, score in docs_with_scores
            ]

        # Save to session
        self.session_service.add_turn(
            session_id=session_id,
            question=request.question,
            answer=answer,
            sources=[s.dict() for s in sources],
        )

        elapsed = (time.time() - start) * 1000

        return QueryResponse(
            answer=answer,
            session_id=session_id,
            question=request.question,
            sources=sources,
            model_used=self.llm_service.model_name,
            processing_time_ms=round(elapsed, 2),
        )

    async def stream_query(
        self, request: QueryRequest
    ) -> AsyncGenerator[str, None]:
        """Streaming RAG pipeline."""
        session_id = self.session_service.get_or_create(request.session_id)
        history = self.session_service.get_history(session_id)

        docs_with_scores = await self.vector_service.similarity_search(
            query=request.question,
            top_k=request.top_k,
            doc_ids=request.doc_ids,
            use_mmr=request.use_mmr,
        )

        full_answer = ""
        async for token in self.llm_service.stream_answer(
            question=request.question,
            docs_with_scores=docs_with_scores,
            history=history,
        ):
            full_answer += token
            yield token

        self.session_service.add_turn(session_id, request.question, full_answer)
