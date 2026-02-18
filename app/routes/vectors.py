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
