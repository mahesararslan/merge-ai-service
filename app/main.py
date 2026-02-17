"""
AI Study Assistant - Main FastAPI Application
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.routes import health_router, ingest_router, query_router, study_plan_router, utils_router

# Configure logging
settings = get_settings()
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.
    Handles startup and shutdown events.
    """
    # Startup
    logger.info("Starting AI Study Assistant service...")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Qdrant URL: {settings.qdrant_url}")
    logger.info(f"Collection: {settings.collection_name}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down AI Study Assistant service...")


# Create FastAPI application
app = FastAPI(
    title="AI Study Assistant",
    description="""
    AI-powered study assistant with RAG (Retrieval-Augmented Generation) capabilities.
    
    ## Features
    
    * **RAG-Based Q&A**: Answer questions using course materials with source citations
    * **Document Ingestion**: Process PDF, DOCX, PPTX, and TXT files
    * **Study Plan Generation**: Create personalized study schedules with calendar integration
    * **Streaming Responses**: Real-time answer generation via SSE
    
    ## Tech Stack
    
    * FastAPI with async/await
    * Qdrant Cloud for vector storage
    * Cohere API for embeddings
    * Google Gemini for LLM with function calling
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle uncaught exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if not settings.is_production else None,
            "code": "INTERNAL_ERROR"
        }
    )


# Register routers
app.include_router(health_router)
app.include_router(ingest_router)
app.include_router(query_router)
app.include_router(study_plan_router)
app.include_router(utils_router)


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with service information."""
    return {
        "service": "AI Study Assistant",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=settings.port,
        reload=not settings.is_production,
        log_level=settings.log_level.lower()
    )
