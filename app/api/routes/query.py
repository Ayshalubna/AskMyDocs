"""Query API - RAG Q&A and semantic search endpoints."""

import json
import time
from typing import Optional

import structlog
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from app.models.schemas import QueryRequest, QueryResponse, SearchRequest, SearchResponse, SourceChunk
from app.services.rag_pipeline import RAGPipeline
from app.services.vector_store import VectorStoreService

router = APIRouter()
logger = structlog.get_logger(__name__)

_pipeline: Optional[RAGPipeline] = None


async def get_pipeline(request: Request) -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline(request.app.state.vector_service)
        await _pipeline.initialize()
    return _pipeline


@router.post("/ask", response_model=QueryResponse)
async def ask_question(
    request: QueryRequest,
    pipeline: RAGPipeline = Depends(get_pipeline),
):
    """
    Ask a question against your indexed documents.
    
    Uses RAG: semantic retrieval + LLM generation.
    Supports conversation history via session_id.
    """
    if request.stream:
        return StreamingResponse(
            _stream_response(request, pipeline),
            media_type="text/event-stream",
        )
    return await pipeline.query(request)


async def _stream_response(request: QueryRequest, pipeline: RAGPipeline):
    """SSE streaming generator."""
    try:
        async for token in pipeline.stream_query(request):
            data = json.dumps({"token": token, "done": False})
            yield f"data: {data}\n\n"
        yield f"data: {json.dumps({'token': '', 'done': True})}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"


@router.post("/search", response_model=SearchResponse)
async def semantic_search(
    request: SearchRequest,
    vec_service: VectorStoreService = Depends(
        lambda req: req.app.state.vector_service
    ),
):
    """
    Pure semantic search — returns relevant chunks without LLM generation.
    Useful for exploring document content.
    """
    start = time.time()
    results = await vec_service.similarity_search(
        query=request.query,
        top_k=request.top_k,
        doc_ids=request.doc_ids,
        use_mmr=request.use_mmr,
    )
    elapsed = (time.time() - start) * 1000

    sources = [
        SourceChunk(
            content=doc.page_content,
            score=round(score, 4),
            doc_id=doc.metadata.get("doc_id", ""),
            filename=doc.metadata.get("filename", ""),
            chunk_index=doc.metadata.get("chunk_index", 0),
            page_number=doc.metadata.get("page_number"),
            metadata=doc.metadata,
        )
        for doc, score in results
    ]

    return SearchResponse(
        query=request.query,
        results=sources,
        total_found=len(sources),
        processing_time_ms=round(elapsed, 2),
    )
