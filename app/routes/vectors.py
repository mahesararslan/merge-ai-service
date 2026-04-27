"""
Vector management endpoints for cleanup and maintenance.
"""

import logging
from fastapi import APIRouter, HTTPException, status, BackgroundTasks
from pydantic import BaseModel

from app.services.vector_store import vector_store_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/vectors", tags=["Vectors"])


class DeleteConversationVectorsResponse(BaseModel):
    """Response for conversation vector deletion."""
    success: bool
    conversation_id: str
    vectors_deleted: int


class FileChunksResponse(BaseModel):
    """Response model for fetching all chunks of a file."""
    file_id: str
    chunk_count: int
    char_count: int
    content: str  # Concatenated text of all chunks, ordered by chunk_index


@router.get(
    "/file/{file_id}/all-chunks",
    response_model=FileChunksResponse,
    responses={
        404: {"description": "No chunks found for file"},
        500: {"description": "Fetch failed"},
    },
)
async def get_file_all_chunks(file_id: str):
    """
    Fetch and concatenate all chunks of a single file_id.

    Used when a room file is "attached" to an AI conversation. The caller
    (NestJS) takes the returned `content` and stores it as a Flow 1
    attachment context on the conversation, the same shape used for
    personal-file attachments. Chunks are ordered by chunk_index.
    """
    logger.info(f"Fetching all chunks for file {file_id}")
    try:
        chunks = await vector_store_service.get_all_chunks_by_file(file_id)
    except Exception as e:
        logger.error(f"Failed to fetch chunks for {file_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch chunks: {e}",
        )

    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No chunks found for file {file_id}",
        )

    content = "\n\n".join(c["content"] for c in chunks if c.get("content"))
    return FileChunksResponse(
        file_id=file_id,
        chunk_count=len(chunks),
        char_count=len(content),
        content=content,
    )


@router.delete(
    "/conversation/{conversation_id}",
    response_model=DeleteConversationVectorsResponse,
    responses={
        404: {"description": "Conversation not found"},
        500: {"description": "Deletion failed"}
    }
)
async def delete_conversation_vectors(conversation_id: str, background_tasks: BackgroundTasks):
    """
    Delete all temporary attachment vectors for a conversation.
    
    Called when:
    - A conversation is deleted by the user
    - Cleanup job removes expired vectors
    
    Args:
        conversation_id: ID of the conversation
        
    Returns:
        Count of vectors deleted
    """
    logger.info(f"Deleting vectors for conversation {conversation_id}")
    
    try:
        vectors_deleted = await vector_store_service.delete_conversation_vectors(conversation_id)
        
        logger.info(f"Deleted {vectors_deleted} vectors for conversation {conversation_id}")
        
        return DeleteConversationVectorsResponse(
            success=True,
            conversation_id=conversation_id,
            vectors_deleted=vectors_deleted
        )
        
    except Exception as e:
        logger.error(f"Failed to delete conversation vectors: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete conversation vectors: {str(e)}"
        )
