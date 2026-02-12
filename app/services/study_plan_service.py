"""
Study plan generation service.
Orchestrates calendar fetching and LLM-based plan generation.
"""

import logging
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime

from app.config import get_settings
from app.services.calendar_service import calendar_service
from app.services.llm_service import llm_service
from app.models import (
    StudyPlanRequest,
    StudyPlanResponse,
    WeeklySchedule,
    StudySession,
    CalendarConflict,
)

logger = logging.getLogger(__name__)


class StudyPlanService:
    """
    Orchestrates study plan generation using:
    1. Calendar service for user availability
    2. LLM service with function calling for intelligent planning
    """
    
    def __init__(self):
        self.settings = get_settings()
    
    async def generate_plan(
        self,
        request: StudyPlanRequest
    ) -> StudyPlanResponse:
        """
        Generate a personalized study plan.
        
        Args:
            request: Study plan request with user details and preferences
            
        Returns:
            Complete study plan response
        """
        logger.info(
            f"Generating study plan for user {request.user_id}: "
            f"{request.goal[:50]}..."
        )
        
        # Create calendar fetcher closure with auth token
        async def calendar_fetcher(user_id: str, start_date: str, end_date: str):
            return await calendar_service.fetch_calendar(
                user_id=user_id,
                start_date=start_date,
                end_date=end_date,
                auth_token=request.auth_token
            )
        
        try:
            # Generate plan using LLM with function calling
            raw_plan = await llm_service.generate_study_plan_with_function_calling(
                user_id=request.user_id,
                goal=request.goal,
                topics=request.topics,
                start_date=request.start_date,
                end_date=request.end_date,
                difficulty_level=request.difficulty_level.value,
                preferences=request.preferences.model_dump() if request.preferences else None,
                calendar_fetcher=calendar_fetcher
            )
            
            # Parse and validate the response
            return self._build_response(raw_plan, request)
            
        except Exception as e:
            logger.error(f"Study plan generation failed: {str(e)}")
            raise
    
    def _build_response(
        self,
        raw_plan: Dict[str, Any],
        request: StudyPlanRequest
    ) -> StudyPlanResponse:
        """
        Build StudyPlanResponse from raw LLM output.
        
        Args:
            raw_plan: Raw plan from LLM
            request: Original request for context
            
        Returns:
            Validated StudyPlanResponse
        """
        # Parse weekly schedule
        weekly_schedule = []
        raw_weeks = raw_plan.get('weekly_schedule', [])
        
        for week_data in raw_weeks:
            sessions = []
            for session_data in week_data.get('sessions', []):
                sessions.append(StudySession(
                    date=session_data.get('date', ''),
                    start_time=session_data.get('start_time', ''),
                    end_time=session_data.get('end_time', ''),
                    topic=session_data.get('topic', ''),
                    learning_objectives=session_data.get('learning_objectives', []),
                    resources=session_data.get('resources'),
                    notes=session_data.get('notes')
                ))
            
            weekly_schedule.append(WeeklySchedule(
                week_number=week_data.get('week_number', 1),
                start_date=week_data.get('start_date', ''),
                end_date=week_data.get('end_date', ''),
                sessions=sessions,
                weekly_goals=week_data.get('weekly_goals', []),
                estimated_hours=week_data.get('estimated_hours', 0.0)
            ))
        
        # Parse calendar conflicts
        conflicts = []
        for conflict_data in raw_plan.get('calendar_conflicts', []):
            conflicts.append(CalendarConflict(
                date=conflict_data.get('date', ''),
                event_title=conflict_data.get('event_title', ''),
                conflict_type=conflict_data.get('conflict_type', ''),
                suggestion=conflict_data.get('suggestion', '')
            ))
        
        # Calculate totals
        total_sessions = sum(len(week.sessions) for week in weekly_schedule)
        total_hours = sum(week.estimated_hours for week in weekly_schedule)
        
        return StudyPlanResponse(
            success=True,
            plan_id=str(uuid.uuid4()),
            goal=request.goal,
            total_weeks=len(weekly_schedule),
            total_sessions=total_sessions,
            total_hours=total_hours or raw_plan.get('total_hours', 0),
            weekly_schedule=weekly_schedule,
            milestones=raw_plan.get('milestones', []),
            calendar_conflicts=conflicts,
            adjustment_tips=raw_plan.get('adjustment_tips', []),
            generated_at=datetime.utcnow()
        )
    
    async def preview_calendar(
        self,
        user_id: str,
        start_date: str,
        end_date: str,
        auth_token: str
    ) -> Dict[str, Any]:
        """
        Preview calendar data without generating a plan.
        Useful for debugging and validation.
        
        Args:
            user_id: User ID
            start_date: Start date
            end_date: End date
            auth_token: Auth token for API
            
        Returns:
            Raw calendar data
        """
        return await calendar_service.fetch_calendar(
            user_id=user_id,
            start_date=start_date,
            end_date=end_date,
            auth_token=auth_token
        )


# Singleton instance
study_plan_service = StudyPlanService()
