"""
Utility endpoints for conversation management.
"""

import logging
from typing import List

from fastapi import APIRouter, HTTPException, status

from app.models.schemas import ConversationMessage
from app.services.llm_service import llm_service
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/utils", tags=["Utils"])


class SummarizeConversationRequest(BaseModel):
    """Request model for conversation summarization."""
    messages: List[ConversationMessage] = Field(
        ..., 
        description="Messages to summarize (old messages, not recent ones)"
    )
    existing_summary: str | None = Field(
        None, 
        description="Previous summary to update"
    )


class SummarizeConversationResponse(BaseModel):
    """Response model for conversation summarization."""
    summary: str = Field(..., description="Concise conversation summary")


@router.post(
    "/summarize-conversation",
    response_model=SummarizeConversationResponse,
    responses={
        400: {"description": "Invalid request"},
        500: {"description": "Summarization failed"}
    }
)
async def summarize_conversation(request: SummarizeConversationRequest):
    """
    Summarize a conversation to compress context for LLM.
    
    Takes older conversation messages and creates a concise 3-4 sentence
    summary capturing topics discussed, questions asked, and key concepts.
    
    This is used to save tokens in the LLM context window by replacing
    old messages with a summary while keeping recent messages intact.
    
    Returns:
        SummarizeConversationResponse with the summary text
    """
    logger.info(f"Summarizing {len(request.messages)} conversation messages")
    
    if not request.messages:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one message is required for summarization"
        )
    
    try:
        # Build conversation text
        conversation_text = ""
        for msg in request.messages:
            role = "Student" if msg.role == "user" else "Assistant"
            conversation_text += f"{role}: {msg.content}\n\n"
        
        # Build summarization prompt
        prompt = """You are helping to summarize a conversation between a student and an AI study assistant.

Your task: Create a concise 3-4 sentence summary capturing:
1. Main topics discussed
2. Key questions asked by the student
3. Important concepts explained
4. Any recurring themes or focus areas

Keep it factual and comprehensive but brief."""

        # Include existing summary if present
        if request.existing_summary:
            prompt += f"\n\nPREVIOUS SUMMARY:\n{request.existing_summary}\n\n"
            prompt += "Update this summary to include the new conversation below.\n\n"
        
        prompt += f"\n\nCONVERSATION TO SUMMARIZE:\n{conversation_text}\n\n"
        prompt += "Provide the summary (3-4 sentences):"
        
        # Call Gemini to generate summary
        summary = await llm_service.generate_summary(prompt)
        
        logger.info(f"Successfully generated conversation summary")
        
        return SummarizeConversationResponse(summary=summary)
        
    except Exception as e:
        logger.error(f"Conversation summarization failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to summarize conversation: {str(e)}"
        )
