"""Pydantic models for API request/response schemas."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator


class DocumentStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"


class ChunkMetadata(BaseModel):
    doc_id: str
    filename: str
    chunk_index: int
    page_number: Optional[int] = None
    section: Optional[str] = None
    char_start: Optional[int] = None
    char_end: Optional[int] = None


class DocumentMetadata(BaseModel):
    doc_id: str = Field(default_factory=lambda: str(uuid4()))
    filename: str
    original_filename: str
    file_size_bytes: int
    file_type: str
    status: DocumentStatus = DocumentStatus.PENDING
    num_chunks: int = 0
    num_pages: Optional[int] = None
    uploaded_at: datetime = Field(default_factory=datetime.utcnow)
    indexed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        use_enum_values = True


class DocumentUploadResponse(BaseModel):
    doc_id: str
    filename: str
    status: str
    message: str
    num_chunks: int
    processing_time_ms: float


class DocumentListResponse(BaseModel):
    documents: List[DocumentMetadata]
    total: int
    page: int
    page_size: int


class SourceChunk(BaseModel):
    content: str
    score: float
    doc_id: str
    filename: str
    chunk_index: int
    page_number: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000, description="Question to ask")
    session_id: Optional[str] = Field(None, description="Session ID for conversation history")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve")
    doc_ids: Optional[List[str]] = Field(None, description="Filter to specific documents")
    use_mmr: bool = Field(default=True, description="Use Maximal Marginal Relevance for diversity")
    temperature: float = Field(default=0.1, ge=0.0, le=1.0)
    include_sources: bool = Field(default=True, description="Include source chunks in response")
    stream: bool = Field(default=False, description="Stream response tokens")

    @validator("question")
    def clean_question(cls, v):
        return v.strip()


class QueryResponse(BaseModel):
    answer: str
    session_id: str
    question: str
    sources: List[SourceChunk] = []
    model_used: str
    processing_time_ms: float
    tokens_used: Optional[int] = None
    confidence_score: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ConversationTurn(BaseModel):
    role: str  # "human" or "assistant"
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    sources: Optional[List[SourceChunk]] = None


class SessionResponse(BaseModel):
    session_id: str
    created_at: datetime
    last_active: datetime
    num_turns: int
    history: List[ConversationTurn] = []


class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str
    components: Dict[str, Dict[str, Any]]
    uptime_seconds: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=500)
    top_k: int = Field(default=5, ge=1, le=20)
    doc_ids: Optional[List[str]] = None
    use_mmr: bool = True


class SearchResponse(BaseModel):
    query: str
    results: List[SourceChunk]
    total_found: int
    processing_time_ms: float


class IndexStatsResponse(BaseModel):
    total_documents: int
    total_chunks: int
    index_size_mb: float
    embedding_model: str
    embedding_dimension: int
    index_type: str
