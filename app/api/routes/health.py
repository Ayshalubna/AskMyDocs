"""Health check endpoint."""

import time
from datetime import datetime

import structlog
from fastapi import APIRouter, Request

from app.core.config import settings
from app.models.schemas import HealthResponse

router = APIRouter()
logger = structlog.get_logger(__name__)
_start_time = time.time()


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request):
    """Full system health check."""
    vector_service = getattr(request.app.state, "vector_service", None)
    stats = vector_service.get_stats() if vector_service else {}

    components = {
        "vector_store": {
            "status": "healthy" if stats.get("has_index") is not None else "initializing",
            "total_documents": stats.get("total_documents", 0),
            "total_chunks": stats.get("total_chunks", 0),
        },
        "storage": {
            "status": "healthy",
            "upload_dir": str(settings.UPLOAD_DIR),
            "vector_store_dir": str(settings.VECTOR_STORE_DIR),
        },
        "embeddings": {
            "status": "healthy",
            "model": settings.EMBEDDING_MODEL,
        },
    }

    return HealthResponse(
        status="healthy",
        version=settings.APP_VERSION,
        environment=settings.ENVIRONMENT,
        components=components,
        uptime_seconds=round(time.time() - _start_time, 2),
    )


@router.get("/")
async def root():
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "health": "/health",
    }
