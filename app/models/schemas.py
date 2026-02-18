"""
Pydantic models for API request/response validation.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


# =============================================================================
# Enums
# =============================================================================

class DocumentType(str, Enum):
    """Supported document types for ingestion."""
    PDF = "pdf"
    DOCX = "docx"
    PPTX = "pptx"
    TXT = "txt"


class DifficultyLevel(str, Enum):
    """Study plan difficulty levels."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class ProcessingStatus(str, Enum):
    """File processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# =============================================================================
# Ingestion Models
# =============================================================================

class IngestRequest(BaseModel):
    """Request model for document ingestion."""
    room_id: str = Field(..., description="ID of the study room")
    file_id: str = Field(..., description="Unique identifier for the file")
    document_type: DocumentType = Field(..., description="Type of document")


class S3IngestRequest(BaseModel):
    """Request model for document ingestion from S3."""
    s3_url: str = Field(..., description="S3 URL of the file to process")
    room_id: str = Field(..., description="ID of the study room")
    file_id: str = Field(..., description="Unique identifier for the file")
    document_type: DocumentType = Field(..., description="Type of document")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class ChunkMetadata(BaseModel):
    """Metadata stored with each vector chunk."""
    room_id: str
    file_id: str
    document_type: str
    chunk_index: int
    total_chunks: int
    section_title: Optional[str] = None
    char_count: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class IngestResponse(BaseModel):
    """Response model for document ingestion."""
    success: bool
    file_id: str
    room_id: str
    chunks_created: int
    processing_time_ms: float
    message: str


class ProcessingStatusResponse(BaseModel):
    """Response model for processing status check."""
    file_id: str
    status: ProcessingStatus
    chunks_created: Optional[int] = None
    error: Optional[str] = None
    processed_at: Optional[datetime] = None


# =============================================================================
# Query Models
# =============================================================================

class ConversationMessage(BaseModel):
    """A message in the conversation history."""
    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class QueryRequest(BaseModel):
    """Request model for RAG query."""
    query: str = Field(..., min_length=1, max_length=2000, description="User's question")
    user_id: str = Field(..., description="ID of the user making the query")
    room_ids: List[str] = Field(..., min_length=1, description="List of room IDs to search")
    context_file_id: Optional[str] = Field(None, description="Optional specific file to focus on")
    top_k: Optional[int] = Field(None, ge=1, le=20, description="Number of chunks to retrieve")
    conversation_history: Optional[List[ConversationMessage]] = Field(
        None, 
        description="Recent conversation history (last 8 messages)"
    )
    conversation_summary: Optional[str] = Field(
        None, 
        description="Summary of older conversation messages"
    )
    # File attachment fields (for AI conversation attachments)
    attachment_s3_url: Optional[str] = Field(
        None, 
        description="S3 URL of attached file (for first message with attachment)"
    )
    attachment_type: Optional[str] = Field(
        None, 
        description="Type of attachment: IMAGE, PDF, DOCX, PPTX, TXT"
    )
    attachment_context: Optional[str] = Field(
        None, 
        description="Extracted text/base64 content for Flow 1 (small files)"
    )
    has_vector_attachment: Optional[bool] = Field(
        False, 
        description="Indicates attachment was stored in vector DB (Flow 2)"
    )
    conversation_id: Optional[str] = Field(
        None, 
        description="Conversation ID for filtering temporary attachment vectors"
    )


class SourceChunk(BaseModel):
    """A source chunk returned with the answer."""
    file_id: str
    chunk_index: int
    content: str
    relevance_score: float
    section_title: Optional[str] = None


class QueryResponse(BaseModel):
    """Response model for RAG query."""
    answer: str
    sources: List[SourceChunk]
    query: str
    processing_time_ms: float
    chunks_retrieved: int
    # Attachment processing results
    attachment_stored: Optional[bool] = Field(
        None, 
        description="Whether attachment was successfully processed"
    )
    extracted_content_length: Optional[int] = Field(
        None, 
        description="Character count of extracted content (Flow 1)"
    )
    chunks_created_for_attachment: Optional[int] = Field(
        None, 
        description="Number of chunks created from attachment (Flow 2)"
    )
    flow_used: Optional[str] = Field(
        None, 
        description="Which processing flow was used: 'direct_injection' or 'vector_storage'"
    )


# =============================================================================
# Study Plan Models
# =============================================================================

class StudyPreferences(BaseModel):
    """User preferences for study plan generation."""
    preferred_study_hours: Optional[List[int]] = Field(
        None, 
        description="Preferred hours of the day for studying (0-23)"
    )
    session_duration_minutes: int = Field(
        default=60, 
        ge=15, 
        le=180,
        description="Preferred study session duration"
    )
    break_duration_minutes: int = Field(
        default=15, 
        ge=5, 
        le=60,
        description="Break duration between sessions"
    )
    days_per_week: int = Field(
        default=5, 
        ge=1, 
        le=7,
        description="Number of study days per week"
    )


class StudyPlanRequest(BaseModel):
    """Request model for study plan generation."""
    user_id: str = Field(..., description="ID of the user")
    goal: str = Field(..., min_length=10, max_length=1000, description="Study goal description")
    topics: List[str] = Field(..., min_length=1, description="Topics to cover")
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")
    difficulty_level: DifficultyLevel = Field(default=DifficultyLevel.INTERMEDIATE)
    preferences: Optional[StudyPreferences] = None
    auth_token: str = Field(..., description="Auth token for calendar API")


class StudySession(BaseModel):
    """A single study session in the plan."""
    date: str
    start_time: str
    end_time: str
    topic: str
    learning_objectives: List[str]
    resources: Optional[List[str]] = None
    notes: Optional[str] = None


class WeeklySchedule(BaseModel):
    """Weekly breakdown of study sessions."""
    week_number: int
    start_date: str
    end_date: str
    sessions: List[StudySession]
    weekly_goals: List[str]
    estimated_hours: float


class CalendarConflict(BaseModel):
    """A detected calendar conflict."""
    date: str
    event_title: str
    conflict_type: str
    suggestion: str


class StudyPlanResponse(BaseModel):
    """Response model for study plan generation."""
    success: bool
    plan_id: str
    goal: str
    total_weeks: int
    total_sessions: int
    total_hours: float
    weekly_schedule: List[WeeklySchedule]
    milestones: List[Dict[str, Any]]
    calendar_conflicts: List[CalendarConflict]
    adjustment_tips: List[str]
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class CalendarPreviewRequest(BaseModel):
    """Request for calendar preview (debug endpoint)."""
    user_id: str
    start_date: str
    end_date: str
    auth_token: str


class CalendarPreviewResponse(BaseModel):
    """Response for calendar preview."""
    events: List[Dict[str, Any]]
    total_events: int
    available_slots: List[Dict[str, Any]]


# =============================================================================
# Health Check Models
# =============================================================================

class ServiceStatus(BaseModel):
    """Status of an individual service."""
    name: str
    status: str  # "healthy", "unhealthy", "degraded"
    latency_ms: Optional[float] = None
    message: Optional[str] = None


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str  # "healthy", "unhealthy", "degraded"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str = "1.0.0"
    services: List[ServiceStatus]


# =============================================================================
# Error Models
# =============================================================================

class ErrorResponse(BaseModel):
    """Standard error response model."""
    error: str
    detail: Optional[str] = None
    code: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
