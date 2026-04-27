"""
RAG-Powered Document Q&A API
Production-grade FastAPI application with LangChain + FAISS + HuggingFace
"""

import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator

from app.api.routes import documents, health, query, sessions
from app.core.config import settings
from app.core.exceptions import AppException
from app.core.logging import setup_logging
from app.services.vector_store import VectorStoreService

setup_logging()
logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Application lifespan manager - startup and shutdown."""
    logger.info("🚀 Starting RAG Document Q&A API", version=settings.APP_VERSION)
    
    # Initialize vector store on startup
    vector_service = VectorStoreService()
    await vector_service.initialize()
    app.state.vector_service = vector_service

    from app.services.llm_service import LLMService

    llm_service = LLMService()
    await llm_service.initialize()
    app.state.llm_service = llm_service
    
    logger.info("✅ Application ready", environment=settings.ENVIRONMENT)
    yield
    
    # Graceful shutdown
    logger.info("🔻 Shutting down application")
    await vector_service.cleanup()


def create_application() -> FastAPI:
    """Application factory pattern."""
    app = FastAPI(
        title=settings.APP_NAME,
        description="""
## RAG-Powered Document Q&A System

A production-grade Retrieval-Augmented Generation system that lets you:

- 📄 **Upload documents** (PDF, DOCX, TXT, MD, CSV)
- 🔍 **Semantic search** across your document corpus
- 💬 **Ask questions** and get AI-powered answers with source citations
- 📊 **Track sessions** and conversation history

### Architecture
- **LangChain** for RAG orchestration
- **FAISS** for fast vector similarity search  
- **HuggingFace** sentence-transformers for embeddings
- **Ollama** for local LLM inference (no API costs!)
- **FastAPI** for async REST API
        """,
        version=settings.APP_VERSION,
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # Request ID middleware
    @app.middleware("http")
    async def add_request_id(request: Request, call_next):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        start_time = time.time()
        
        response = await call_next(request)
        
        process_time = (time.time() - start_time) * 1000
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = f"{process_time:.2f}ms"
        
        logger.info(
            "request_completed",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            process_time_ms=round(process_time, 2),
            request_id=request_id,
        )
        return response

    # Exception handlers
    @app.exception_handler(AppException)
    async def app_exception_handler(request: Request, exc: AppException):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.error_code,
                "message": exc.message,
                "request_id": getattr(request.state, "request_id", None),
            },
        )

    # Routes
    app.include_router(health.router, tags=["Health"])
    app.include_router(documents.router, prefix="/api/v1/documents", tags=["Documents"])
    app.include_router(query.router, prefix="/api/v1/query", tags=["Query"])
    app.include_router(sessions.router, prefix="/api/v1/sessions", tags=["Sessions"])

    # Prometheus metrics
    Instrumentator(
        should_group_status_codes=False,
        should_ignore_untemplated=True,
        excluded_handlers=["/health", "/metrics"],
    ).instrument(app).expose(app, endpoint="/metrics")

    return app


app = create_application()
