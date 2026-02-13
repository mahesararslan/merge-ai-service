"""Models package initialization."""

from app.models.schemas import (
    # Enums
    DocumentType,
    DifficultyLevel,
    ProcessingStatus,
    # Ingestion
    IngestRequest,
    S3IngestRequest,
    IngestResponse,
    ProcessingStatusResponse,
    ChunkMetadata,
    # Query
    QueryRequest,
    QueryResponse,
    SourceChunk,
    # Study Plan
    StudyPlanRequest,
    StudyPlanResponse,
    StudyPreferences,
    StudySession,
    WeeklySchedule,
    CalendarConflict,
    CalendarPreviewRequest,
    CalendarPreviewResponse,
    # Health
    HealthResponse,
    ServiceStatus,
    # Error
    ErrorResponse,
)

__all__ = [
    "DocumentType",
    "DifficultyLevel",
    "ProcessingStatus",
    "IngestRequest",
    "S3IngestRequest",
    "IngestResponse",
    "ProcessingStatusResponse",
    "ChunkMetadata",
    "QueryRequest",
    "QueryResponse",
    "SourceChunk",
    "StudyPlanRequest",
    "StudyPlanResponse",
    "StudyPreferences",
    "StudySession",
    "WeeklySchedule",
    "CalendarConflict",
    "CalendarPreviewRequest",
    "CalendarPreviewResponse",
    "HealthResponse",
    "ServiceStatus",
    "ErrorResponse",
]
