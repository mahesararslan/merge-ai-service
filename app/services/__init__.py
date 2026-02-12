"""Services package initialization."""

from app.services.document_processor import document_processor, DocumentProcessor
from app.services.chunking_service import chunking_service, ChunkingService, TextChunk
from app.services.embedding_service import embedding_service, EmbeddingService
from app.services.vector_store import vector_store_service, VectorStoreService
from app.services.llm_service import llm_service, LLMService
from app.services.retrieval_service import retrieval_service, RetrievalService
from app.services.calendar_service import calendar_service, CalendarService
from app.services.study_plan_service import study_plan_service, StudyPlanService

__all__ = [
    # Document processing
    "document_processor",
    "DocumentProcessor",
    "chunking_service",
    "ChunkingService",
    "TextChunk",
    # Embeddings & Vector store
    "embedding_service",
    "EmbeddingService",
    "vector_store_service",
    "VectorStoreService",
    # LLM & Retrieval
    "llm_service",
    "LLMService",
    "retrieval_service",
    "RetrievalService",
    # Study planning
    "calendar_service",
    "CalendarService",
    "study_plan_service",
    "StudyPlanService",
]
