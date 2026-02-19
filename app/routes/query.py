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
from app.services.document_processor import document_processor
from app.config import get_settings

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
    1. Process attachment if provided (Flow 1 or Flow 2)
    2. Generate query embedding via Cohere
    3. Search Qdrant for relevant chunks (dual retrieval if attachment)
    4. Generate answer using Gemini with context
    5. Return answer with source citations
    
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
    
    settings = get_settings()
    attachment_result = None
    
    try:
        # Process attachment if provided (only on first message with attachment)
        if request.attachment_s3_url and request.attachment_type:
            logger.info(
                f"Processing attachment: type={request.attachment_type}, "
                f"S3={request.attachment_s3_url[:50]}..."
            )
            
            # Get file size from request
            file_size = request.attachment_file_size or 0
            
            attachment_result = await document_processor.process_attachment(
                s3_url=request.attachment_s3_url,
                attachment_type=request.attachment_type,
                file_size=file_size,
                text_threshold=settings.attachment_text_size_threshold,
                file_size_threshold=settings.attachment_file_size_threshold
            )
            
            if not attachment_result['success']:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Failed to process attachment: {attachment_result['error']}"
                )
        
        # Execute query with attachment context
        response = await retrieval_service.query(
            query=request.query,
            user_id=request.user_id,
            room_ids=request.room_ids,
            context_file_id=request.context_file_id,
            top_k=request.top_k,
            conversation_history=[msg.dict() for msg in request.conversation_history] if request.conversation_history else None,
            conversation_summary=request.conversation_summary,
            conversation_id=request.conversation_id,
            attachment_context=request.attachment_context,
            has_vector_attachment=request.has_vector_attachment,
            attachment_result=attachment_result
        )
        
        # Add attachment processing info to response
        if attachment_result:
            response.attachment_stored = True
            response.flow_used = attachment_result['flow']
            
            if attachment_result['flow'] == 'direct_injection':
                response.extracted_content = attachment_result['extracted_content']
                response.extracted_content_length = attachment_result['char_count']
            elif attachment_result['flow'] == 'vector_storage':
                # chunks_created will be set by retrieval service
                pass
        
        return response
        
    except HTTPException:
        raise
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
                top_k=request.top_k,
                conversation_history=[msg.dict() for msg in request.conversation_history] if request.conversation_history else None,
                conversation_summary=request.conversation_summary,
                conversation_id=request.conversation_id,
                attachment_context=request.attachment_context,
                has_vector_attachment=request.has_vector_attachment
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
