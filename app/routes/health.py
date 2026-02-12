"""
Health check endpoint for monitoring service status.
"""

import logging
import time
from datetime import datetime

from fastapi import APIRouter

from app.models import HealthResponse, ServiceStatus
from app.services import embedding_service, vector_store_service, llm_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Check the health status of all service dependencies.
    
    Returns status of:
    - Qdrant vector database
    - Cohere embedding API
    - Gemini LLM API
    """
    services = []
    overall_status = "healthy"
    
    # Check Qdrant
    qdrant_status = await check_service(
        "Qdrant Vector DB",
        vector_store_service.health_check
    )
    services.append(qdrant_status)
    if qdrant_status.status != "healthy":
        overall_status = "degraded"
    
    # Check Cohere
    cohere_status = await check_service(
        "Cohere Embeddings",
        embedding_service.health_check
    )
    services.append(cohere_status)
    if cohere_status.status != "healthy":
        overall_status = "degraded"
    
    # Check Gemini
    gemini_status = await check_service(
        "Gemini LLM",
        llm_service.health_check
    )
    services.append(gemini_status)
    if gemini_status.status != "healthy":
        overall_status = "degraded"
    
    # If all services are unhealthy, mark as unhealthy
    if all(s.status == "unhealthy" for s in services):
        overall_status = "unhealthy"
    
    return HealthResponse(
        status=overall_status,
        timestamp=datetime.utcnow(),
        version="1.0.0",
        services=services
    )


async def check_service(name: str, check_func) -> ServiceStatus:
    """
    Helper to check a service and measure latency.
    
    Args:
        name: Service name
        check_func: Async function that returns bool
        
    Returns:
        ServiceStatus object
    """
    try:
        start = time.time()
        is_healthy = await check_func()
        latency = (time.time() - start) * 1000
        
        return ServiceStatus(
            name=name,
            status="healthy" if is_healthy else "unhealthy",
            latency_ms=latency,
            message="OK" if is_healthy else "Service check failed"
        )
    except Exception as e:
        logger.error(f"Health check failed for {name}: {str(e)}")
        return ServiceStatus(
            name=name,
            status="unhealthy",
            latency_ms=None,
            message=str(e)
        )
