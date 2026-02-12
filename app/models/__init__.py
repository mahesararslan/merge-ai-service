"""Models package initialization."""

from app.models.schemas import (
    # Enums
    DocumentType,
    DifficultyLevel,
    # Ingestion
    IngestRequest,
    IngestResponse,
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
    "IngestRequest",
    "IngestResponse",
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
