"""
Study plan generation endpoints.
Provides personalized study schedules with calendar integration.
"""

import logging
from datetime import datetime

from fastapi import APIRouter, HTTPException, status

from app.models import (
    StudyPlanRequest,
    StudyPlanResponse,
    CalendarPreviewRequest,
    CalendarPreviewResponse,
    ErrorResponse
)
from app.services import study_plan_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/study-plan", tags=["Study Plan"])


@router.post(
    "/generate",
    response_model=StudyPlanResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Generation error"}
    }
)
async def generate_study_plan(request: StudyPlanRequest):
    """
    Generate a personalized study plan with calendar integration.
    
    Process:
    1. Initialize Gemini with calendar tool
    2. Send study goal and preferences to LLM
    3. LLM calls get_user_calendar function
    4. Backend fetches calendar from NestJS API
    5. LLM analyzes availability and creates plan
    6. Return structured study schedule
    
    The plan includes:
    - Weekly breakdown with sessions
    - Learning objectives per session
    - Calendar conflict detection
    - Adjustment tips
    - Progress milestones
    
    Returns:
        StudyPlanResponse with complete schedule
    """
    logger.info(
        f"Generating study plan for user {request.user_id}: "
        f"{request.goal[:50]}..."
    )
    
    # Validate dates
    try:
        start = datetime.strptime(request.start_date, "%Y-%m-%d")
        end = datetime.strptime(request.end_date, "%Y-%m-%d")
        
        if end <= start:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="End date must be after start date"
            )
        
        # Check if date range is reasonable (max 6 months)
        days_diff = (end - start).days
        if days_diff > 180:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Study plan duration cannot exceed 6 months"
            )
            
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid date format. Use YYYY-MM-DD. Error: {str(e)}"
        )
    
    # Validate topics
    if not request.topics:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one topic is required"
        )
    
    try:
        response = await study_plan_service.generate_plan(request)
        
        logger.info(
            f"Study plan generated: {response.total_weeks} weeks, "
            f"{response.total_sessions} sessions, "
            f"{response.total_hours} hours"
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Study plan generation failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate study plan: {str(e)}"
        )


@router.post(
    "/preview",
    response_model=CalendarPreviewResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Calendar fetch error"}
    }
)
async def preview_calendar(request: CalendarPreviewRequest):
    """
    Preview calendar data without generating a plan.
    
    Debug endpoint to view:
    - User's calendar events
    - Calculated available time slots
    - Deadline detection
    
    This bypasses the LLM and returns raw calendar analysis.
    
    Returns:
        CalendarPreviewResponse with events and availability
    """
    logger.info(
        f"Calendar preview for user {request.user_id}: "
        f"{request.start_date} to {request.end_date}"
    )
    
    # Validate dates
    try:
        start = datetime.strptime(request.start_date, "%Y-%m-%d")
        end = datetime.strptime(request.end_date, "%Y-%m-%d")
        
        if end <= start:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="End date must be after start date"
            )
            
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid date format. Use YYYY-MM-DD. Error: {str(e)}"
        )
    
    try:
        calendar_data = await study_plan_service.preview_calendar(
            user_id=request.user_id,
            start_date=request.start_date,
            end_date=request.end_date,
            auth_token=request.auth_token
        )
        
        return CalendarPreviewResponse(
            events=calendar_data.get('events', []),
            total_events=calendar_data.get('total_events', 0),
            available_slots=calendar_data.get('available_slots', [])
        )
        
    except Exception as e:
        logger.error(f"Calendar preview failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch calendar: {str(e)}"
        )
