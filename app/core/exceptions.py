"""Custom application exceptions with HTTP status codes."""

from typing import Any, Dict, Optional


class AppException(Exception):
    """Base application exception."""

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        error_code: str = "INTERNAL_ERROR",
        details: Optional[Dict[str, Any]] = None,
    ):
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        self.details = details or {}
        super().__init__(message)


class DocumentNotFoundError(AppException):
    def __init__(self, doc_id: str):
        super().__init__(
            message=f"Document '{doc_id}' not found",
            status_code=404,
            error_code="DOCUMENT_NOT_FOUND",
        )


class DocumentProcessingError(AppException):
    def __init__(self, filename: str, reason: str):
        super().__init__(
            message=f"Failed to process '{filename}': {reason}",
            status_code=422,
            error_code="DOCUMENT_PROCESSING_ERROR",
        )


class UnsupportedFileTypeError(AppException):
    def __init__(self, extension: str, allowed: list):
        super().__init__(
            message=f"File type '{extension}' not supported. Allowed: {', '.join(allowed)}",
            status_code=415,
            error_code="UNSUPPORTED_FILE_TYPE",
        )


class FileTooLargeError(AppException):
    def __init__(self, size_mb: float, max_mb: int):
        super().__init__(
            message=f"File size {size_mb:.1f}MB exceeds maximum {max_mb}MB",
            status_code=413,
            error_code="FILE_TOO_LARGE",
        )


class VectorStoreError(AppException):
    def __init__(self, operation: str, reason: str):
        super().__init__(
            message=f"Vector store error during '{operation}': {reason}",
            status_code=500,
            error_code="VECTOR_STORE_ERROR",
        )


class LLMError(AppException):
    def __init__(self, reason: str):
        super().__init__(
            message=f"LLM inference error: {reason}",
            status_code=503,
            error_code="LLM_ERROR",
        )


class SessionNotFoundError(AppException):
    def __init__(self, session_id: str):
        super().__init__(
            message=f"Session '{session_id}' not found or expired",
            status_code=404,
            error_code="SESSION_NOT_FOUND",
        )


class NoDocumentsIndexedError(AppException):
    def __init__(self):
        super().__init__(
            message="No documents have been indexed yet. Please upload documents first.",
            status_code=400,
            error_code="NO_DOCUMENTS_INDEXED",
        )


class RateLimitExceededError(AppException):
    def __init__(self, retry_after: int):
        super().__init__(
            message=f"Rate limit exceeded. Retry after {retry_after} seconds.",
            status_code=429,
            error_code="RATE_LIMIT_EXCEEDED",
            details={"retry_after": retry_after},
        )
