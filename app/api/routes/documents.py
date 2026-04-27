"""Documents API - upload, list, delete, stats."""

import time
import uuid
from pathlib import Path
from typing import List, Optional

import aiofiles
import structlog
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.exceptions import FileTooLargeError, UnsupportedFileTypeError
from app.models.schemas import (
    DocumentListResponse,
    DocumentMetadata,
    DocumentStatus,
    DocumentUploadResponse,
    IndexStatsResponse,
)
from app.services.document_processor import DocumentProcessor
from app.services.vector_store import VectorStoreService

router = APIRouter()
logger = structlog.get_logger(__name__)
processor = DocumentProcessor()


def get_vector_service(request: Request) -> VectorStoreService:
    return request.app.state.vector_service


@router.post("/upload", response_model=DocumentUploadResponse, status_code=201)
async def upload_document(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    vector_service: VectorStoreService = Depends(get_vector_service),
):
    """
    Upload and index a document.
    
    Supports: PDF, DOCX, TXT, MD, CSV, HTML (max 50MB).
    """
    # Validate extension
    ext = Path(file.filename).suffix.lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise UnsupportedFileTypeError(ext, settings.ALLOWED_EXTENSIONS)

    # Validate size
    content = await file.read()
    if len(content) > settings.max_upload_bytes:
        raise FileTooLargeError(len(content) / (1024 * 1024), settings.MAX_UPLOAD_SIZE_MB)

    doc_id = str(uuid.uuid4())
    safe_filename = f"{doc_id}{ext}"
    file_path = settings.UPLOAD_DIR / safe_filename

    start = time.time()

    try:
        # Save to disk
        async with aiofiles.open(file_path, "wb") as f:
            await f.write(content)

        # Process and chunk
        extra_meta = {
            "title": title or file.filename,
            "description": description or "",
        }
        chunks, file_meta = await processor.process_file(
            file_path=file_path,
            doc_id=doc_id,
            filename=file.filename,
            extra_metadata=extra_meta,
        )

        # Index in FAISS
        doc_metadata = {
            "doc_id": doc_id,
            "filename": file.filename,
            "original_filename": file.filename,
            "file_size_bytes": len(content),
            "file_type": ext,
            "status": DocumentStatus.INDEXED,
            "title": title or file.filename,
            "description": description or "",
            **file_meta,
        }
        await vector_service.add_documents(chunks, doc_id, doc_metadata)

        elapsed = (time.time() - start) * 1000
        logger.info("document_uploaded", doc_id=doc_id, filename=file.filename)

        return DocumentUploadResponse(
            doc_id=doc_id,
            filename=file.filename,
            status="indexed",
            message=f"Successfully indexed {len(chunks)} chunks",
            num_chunks=len(chunks),
            processing_time_ms=round(elapsed, 2),
        )

    except (UnsupportedFileTypeError, FileTooLargeError):
        raise
    except Exception as e:
        # Clean up file on failure
        file_path.unlink(missing_ok=True)
        logger.error("upload_failed", filename=file.filename, error=str(e))
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@router.get("/", response_model=DocumentListResponse)
async def list_documents(
    page: int = 1,
    page_size: int = 20,
    vector_service: VectorStoreService = Depends(get_vector_service),
):
    """List all indexed documents with pagination."""
    all_docs = vector_service.list_documents()
    total = len(all_docs)
    start = (page - 1) * page_size
    paginated = all_docs[start : start + page_size]

    return DocumentListResponse(
        documents=[DocumentMetadata(**d) for d in paginated],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/stats", response_model=IndexStatsResponse)
async def get_index_stats(
    vector_service: VectorStoreService = Depends(get_vector_service),
):
    """Get vector index statistics."""
    return IndexStatsResponse(**vector_service.get_stats())


@router.get("/{doc_id}", response_model=DocumentMetadata)
async def get_document(
    doc_id: str,
    vector_service: VectorStoreService = Depends(get_vector_service),
):
    """Get metadata for a specific document."""
    meta = vector_service.get_document_metadata(doc_id)
    if not meta:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found")
    return DocumentMetadata(**meta)


@router.delete("/{doc_id}", status_code=200)
async def delete_document(
    doc_id: str,
    vector_service: VectorStoreService = Depends(get_vector_service),
):
    """Delete a document and remove it from the index."""
    meta = vector_service.get_document_metadata(doc_id)
    deleted = await vector_service.delete_document(doc_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found")

    # Remove file from disk
    if meta:
        ext = meta.get("file_type", "")
        file_path = settings.UPLOAD_DIR / f"{doc_id}{ext}"
        file_path.unlink(missing_ok=True)

    return {"message": f"Document '{doc_id}' deleted successfully", "doc_id": doc_id}
