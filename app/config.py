"""
Application configuration using Pydantic settings management.
All configuration is loaded from environment variables.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Qdrant Vector Database
    qdrant_url: str = Field(default="http://localhost:6333", alias="QDRANT_URL")
    qdrant_api_key: Optional[str] = Field(default=None, alias="QDRANT_API_KEY")
    collection_name: str = Field(default="study_materials", alias="COLLECTION_NAME")
    
    # Cohere Embeddings API
    cohere_api_key: str = Field(alias="COHERE_API_KEY")
    embedding_model: str = Field(default="embed-english-v3.0", alias="EMBEDDING_MODEL")
    embedding_dimension: int = Field(default=1024, alias="EMBEDDING_DIMENSION")
    
    # Google Gemini LLM
    gemini_api_key: str = Field(alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-1.5-pro", alias="GEMINI_MODEL")
    
    # Backend API Server (NestJS)
    api_server_url: str = Field(default="https://api.mergeedu.app", alias="API_SERVER_URL")
    
    # Chunking Configuration
    chunk_size: int = Field(default=512, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=100, alias="CHUNK_OVERLAP")
    
    # Retrieval Configuration
    top_k_results: int = Field(default=5, alias="TOP_K_RESULTS")
    min_relevance_score: float = Field(default=0.3, alias="MIN_RELEVANCE_SCORE")
    
    # File Processing
    max_file_size_mb: int = Field(default=10, alias="MAX_FILE_SIZE_MB")
    
    # File Attachment Configuration (for AI conversation attachments)
    attachment_text_size_threshold: int = Field(default=80000, alias="ATTACHMENT_TEXT_SIZE_THRESHOLD")
    attachment_file_size_threshold: int = Field(default=8388608, alias="ATTACHMENT_FILE_SIZE_THRESHOLD")
    max_document_size: int = Field(default=15728640, alias="MAX_DOCUMENT_SIZE")
    max_text_size: int = Field(default=5242880, alias="MAX_TEXT_SIZE")
    max_image_size: int = Field(default=5242880, alias="MAX_IMAGE_SIZE")
    temp_vector_ttl_days: int = Field(default=7, alias="TEMP_VECTOR_TTL_DAYS")
    
    # Server Configuration
    port: int = Field(default=8001, alias="PORT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    environment: str = Field(default="development", alias="ENVIRONMENT")
    
    # Timeouts (in seconds)
    calendar_api_timeout: int = Field(default=10, alias="CALENDAR_API_TIMEOUT")
    embedding_api_timeout: int = Field(default=30, alias="EMBEDDING_API_TIMEOUT")
    llm_api_timeout: int = Field(default=60, alias="LLM_API_TIMEOUT")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"
    
    @property
    def max_file_size_bytes(self) -> int:
        """Convert max file size from MB to bytes."""
        return self.max_file_size_mb * 1024 * 1024
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment.lower() == "production"


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    Uses lru_cache to avoid reading .env file on every request.
    """
    return Settings()
