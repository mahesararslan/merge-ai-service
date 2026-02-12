"""
Query endpoints for RAG-based Q&A.
Supports both standard and streaming responses.
"""

import logging
import json
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse

from app.models import QueryRequest, QueryResponse, ErrorResponse
from app.services import retrieval_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/query", tags=["Query"])


@router.post(
    "",
    response_model=QueryResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        404: {"model": ErrorResponse, "description": "No relevant content found"},
        500: {"model": ErrorResponse, "description": "Processing error"}
    }
)
async def query(request: QueryRequest):
    """
    Execute a RAG query and return a complete response.
    
    Process:
    1. Generate query embedding via Cohere
    2. Search Qdrant for relevant chunks
    3. Generate answer using Gemini with context
    4. Return answer with source citations
    
    Returns:
        QueryResponse with answer, sources, and metadata
    """
    logger.info(
        f"Query received from user {request.user_id}: "
        f"{request.query[:50]}..."
    )
    
    if not request.room_ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one room_id is required"
        )
    
    try:
        response = await retrieval_service.query(
            query=request.query,
            user_id=request.user_id,
            room_ids=request.room_ids,
            context_file_id=request.context_file_id,
            top_k=request.top_k
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Query failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query processing failed: {str(e)}"
        )


@router.post("/stream")
async def query_stream(request: QueryRequest):
    """
    Execute a RAG query with streaming response using Server-Sent Events.
    
    Events emitted:
    - status: Processing status updates
    - sources: Retrieved source documents
    - chunk: Answer text chunks as they're generated
    - complete: Final metadata when done
    - error: Error information if something fails
    
    Returns:
        EventSourceResponse with SSE stream
    """
    logger.info(
        f"Streaming query from user {request.user_id}: "
        f"{request.query[:50]}..."
    )
    
    if not request.room_ids:
        # Return error as SSE event
        async def error_stream():
            yield {
                "event": "error",
                "data": json.dumps({"error": "At least one room_id is required"})
            }
        return EventSourceResponse(error_stream())
    
    async def event_generator() -> AsyncGenerator[dict, None]:
        try:
            async for event in retrieval_service.query_stream(
                query=request.query,
                user_id=request.user_id,
                room_ids=request.room_ids,
                context_file_id=request.context_file_id,
                top_k=request.top_k
            ):
                yield {
                    "event": event.get("event", "message"),
                    "data": json.dumps(event.get("data", {}))
                }
                
        except Exception as e:
            logger.error(f"Streaming query failed: {str(e)}")
            yield {
                "event": "error",
                "data": json.dumps({"error": str(e)})
            }
    
    return EventSourceResponse(event_generator())
