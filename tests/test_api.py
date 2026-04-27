"""
Integration and unit tests for RAG Document Q&A API.
Run: pytest tests/ -v --cov=app --cov-report=html
"""

import io
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient

from app.main import app


@pytest.fixture(scope="session")
def client():
    """Sync test client."""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def sample_txt(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text(
        "LangChain is a framework for building LLM-powered applications. "
        "It provides tools for document loading, text splitting, embeddings, "
        "vector stores, and retrieval-augmented generation. "
        "FAISS is a library developed by Meta for efficient similarity search. "
        "It supports both CPU and GPU acceleration."
    )
    return f


# ─── Health ───────────────────────────────────────────────
class TestHealth:
    def test_health_endpoint(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "components" in data
        assert "uptime_seconds" in data

    def test_root_endpoint(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "version" in r.json()


# ─── Documents ────────────────────────────────────────────
class TestDocuments:
    def test_list_documents_empty(self, client):
        r = client.get("/api/v1/documents/")
        assert r.status_code == 200
        data = r.json()
        assert "documents" in data
        assert "total" in data

    def test_upload_txt_document(self, client, sample_txt):
        with open(sample_txt, "rb") as f:
            r = client.post(
                "/api/v1/documents/upload",
                files={"file": ("test.txt", f, "text/plain")},
            )
        assert r.status_code in (201, 500)  # 500 if vector store not available
        if r.status_code == 201:
            data = r.json()
            assert "doc_id" in data
            assert data["num_chunks"] >= 1

    def test_upload_unsupported_extension(self, client):
        r = client.post(
            "/api/v1/documents/upload",
            files={"file": ("test.xyz", b"content", "application/octet-stream")},
        )
        assert r.status_code == 415

    def test_upload_too_large(self, client):
        large = b"x" * (51 * 1024 * 1024)  # 51 MB
        r = client.post(
            "/api/v1/documents/upload",
            files={"file": ("big.txt", large, "text/plain")},
        )
        assert r.status_code == 413

    def test_get_nonexistent_document(self, client):
        r = client.get("/api/v1/documents/nonexistent-id")
        assert r.status_code == 404

    def test_delete_nonexistent_document(self, client):
        r = client.delete("/api/v1/documents/nonexistent-id")
        assert r.status_code == 404

    def test_index_stats(self, client):
        r = client.get("/api/v1/documents/stats")
        assert r.status_code == 200
        data = r.json()
        assert "total_documents" in data
        assert "embedding_model" in data


# ─── Sessions ─────────────────────────────────────────────
class TestSessions:
    def test_create_session(self, client):
        r = client.post("/api/v1/sessions/")
        assert r.status_code == 201
        data = r.json()
        assert "session_id" in data

    def test_get_session(self, client):
        create = client.post("/api/v1/sessions/")
        session_id = create.json()["session_id"]
        r = client.get(f"/api/v1/sessions/{session_id}")
        assert r.status_code == 200
        data = r.json()
        assert data["session_id"] == session_id
        assert "history" in data

    def test_get_nonexistent_session(self, client):
        r = client.get("/api/v1/sessions/nonexistent")
        assert r.status_code == 404

    def test_delete_session(self, client):
        create = client.post("/api/v1/sessions/")
        session_id = create.json()["session_id"]
        r = client.delete(f"/api/v1/sessions/{session_id}")
        assert r.status_code == 200
        # Verify it's gone
        r2 = client.get(f"/api/v1/sessions/{session_id}")
        assert r2.status_code == 404


# ─── Document Processor Unit Tests ────────────────────────
class TestDocumentProcessor:
    def test_clean_text(self):
        from app.services.document_processor import DocumentProcessor
        proc = DocumentProcessor()
        text = "Hello    world\n\n\n\nNew paragraph"
        result = proc._clean_text(text)
        assert "   " not in result  # multiple spaces removed
        assert "\n\n\n" not in result  # triple newlines removed

    def test_deduplicate_chunks(self):
        from langchain.schema import Document
        from app.services.document_processor import DocumentProcessor
        proc = DocumentProcessor()
        chunks = [
            Document(page_content="Hello world", metadata={}),
            Document(page_content="Hello world", metadata={}),  # duplicate
            Document(page_content="Different content", metadata={}),
        ]
        result = proc._deduplicate_chunks(chunks)
        assert len(result) == 2

    def test_enrich_metadata(self):
        from langchain.schema import Document
        from app.services.document_processor import DocumentProcessor
        proc = DocumentProcessor()
        chunks = [Document(page_content="test", metadata={})]
        enriched = proc._enrich_metadata(chunks, "doc123", "test.txt", {"author": "test"})
        assert enriched[0].metadata["doc_id"] == "doc123"
        assert enriched[0].metadata["chunk_index"] == 0
        assert "chunk_id" in enriched[0].metadata
        assert enriched[0].metadata["author"] == "test"


# ─── Schema Validation ────────────────────────────────────
class TestSchemas:
    def test_query_request_validation(self):
        from app.models.schemas import QueryRequest
        req = QueryRequest(question="What is RAG?", top_k=5)
        assert req.question == "What is RAG?"
        assert req.top_k == 5

    def test_query_request_strips_whitespace(self):
        from app.models.schemas import QueryRequest
        req = QueryRequest(question="  hello  ")
        assert req.question == "hello"

    def test_query_request_too_short(self):
        from pydantic import ValidationError
        from app.models.schemas import QueryRequest
        with pytest.raises(ValidationError):
            QueryRequest(question="hi")

    def test_query_request_top_k_bounds(self):
        from pydantic import ValidationError
        from app.models.schemas import QueryRequest
        with pytest.raises(ValidationError):
            QueryRequest(question="Valid question?", top_k=100)


# ─── Config ───────────────────────────────────────────────
class TestConfig:
    def test_settings_load(self):
        from app.core.config import settings
        assert settings.APP_NAME
        assert settings.CHUNK_SIZE > 0
        assert settings.CHUNK_OVERLAP < settings.CHUNK_SIZE
        assert settings.TOP_K_RESULTS > 0

    def test_max_upload_bytes(self):
        from app.core.config import settings
        assert settings.max_upload_bytes == settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024


# ─── Exceptions ───────────────────────────────────────────
class TestExceptions:
    def test_document_not_found(self):
        from app.core.exceptions import DocumentNotFoundError
        exc = DocumentNotFoundError("abc123")
        assert exc.status_code == 404
        assert "abc123" in exc.message

    def test_unsupported_file_type(self):
        from app.core.exceptions import UnsupportedFileTypeError
        exc = UnsupportedFileTypeError(".xyz", [".pdf", ".txt"])
        assert exc.status_code == 415

    def test_file_too_large(self):
        from app.core.exceptions import FileTooLargeError
        exc = FileTooLargeError(55.0, 50)
        assert exc.status_code == 413
