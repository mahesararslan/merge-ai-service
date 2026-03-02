"""
API Key Authentication for AI Service

This module provides security middleware to restrict access to the AI service,
allowing only authorized clients (NestJS API server) to make requests.
"""

from fastapi import Header, HTTPException, status
from typing import Annotated

from app.config import get_settings


async def verify_api_key(
    x_api_key: Annotated[str | None, Header()] = None
) -> str:
    """
    Dependency that validates the API key from the X-API-Key header.
    
    Args:
        x_api_key: The API key from the request header
        
    Returns:
        The validated API key
        
    Raises:
        HTTPException: 401 if API key is missing or invalid
    """
    settings = get_settings()
    
    # Check if API key is missing
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Include X-API-Key header in your request.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    
    # Validate API key
    if x_api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key. Access denied.",
        )
    
    return x_api_key
