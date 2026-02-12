"""
Calendar service for fetching user calendar data from the backend API.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)


class CalendarService:
    """
    Handles calendar data fetching from the NestJS backend API.
    Used by the study plan generator to understand user availability.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.base_url = self.settings.api_server_url
        self.timeout = self.settings.calendar_api_timeout
    
    async def fetch_calendar(
        self,
        user_id: str,
        start_date: str,
        end_date: str,
        auth_token: str
    ) -> Dict[str, Any]:
        """
        Fetch calendar events for a user within a date range.
        
        Args:
            user_id: User ID
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            auth_token: Bearer token for authentication
            
        Returns:
            Calendar data including events and available slots
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    f"{self.base_url}/calendar",
                    params={
                        "userId": user_id,
                        "startDate": start_date,
                        "endDate": end_date
                    },
                    headers={
                        "Authorization": f"Bearer {auth_token}",
                        "Content-Type": "application/json"
                    }
                )
                
                response.raise_for_status()
                data = response.json()
                
                logger.info(
                    f"Fetched calendar for user {user_id}: "
                    f"{len(data.get('events', []))} events"
                )
                
                # Process and return calendar data
                return self._process_calendar_data(data, start_date, end_date)
                
        except httpx.TimeoutException:
            logger.error(f"Calendar API timeout for user {user_id}")
            return self._get_fallback_calendar(start_date, end_date)
            
        except httpx.HTTPStatusError as e:
            logger.error(f"Calendar API error: {e.response.status_code}")
            return self._get_fallback_calendar(start_date, end_date)
            
        except Exception as e:
            logger.error(f"Failed to fetch calendar: {str(e)}")
            return self._get_fallback_calendar(start_date, end_date)
    
    def _process_calendar_data(
        self,
        raw_data: Dict[str, Any],
        start_date: str,
        end_date: str
    ) -> Dict[str, Any]:
        """
        Process raw calendar data into a useful format.
        
        Args:
            raw_data: Raw API response
            start_date: Start date
            end_date: End date
            
        Returns:
            Processed calendar data
        """
        events = raw_data.get('events', [])
        
        # Extract events with relevant info
        processed_events = []
        for event in events:
            processed_events.append({
                "id": event.get('id'),
                "title": event.get('title', 'Untitled Event'),
                "description": event.get('description'),
                "start": event.get('startTime') or event.get('start'),
                "end": event.get('endTime') or event.get('end'),
                "all_day": event.get('allDay', False),
                "type": event.get('type', 'event'),
                "room_id": event.get('roomId')
            })
        
        # Calculate available slots (simplified)
        available_slots = self._calculate_available_slots(
            processed_events, start_date, end_date
        )
        
        # Extract deadlines/assignments
        deadlines = [
            e for e in processed_events 
            if e.get('type') in ['assignment', 'deadline', 'quiz']
        ]
        
        return {
            "events": processed_events,
            "total_events": len(processed_events),
            "deadlines": deadlines,
            "available_slots": available_slots,
            "busy_hours_per_day": self._estimate_busy_hours(processed_events),
            "start_date": start_date,
            "end_date": end_date
        }
    
    def _calculate_available_slots(
        self,
        events: List[Dict[str, Any]],
        start_date: str,
        end_date: str
    ) -> List[Dict[str, Any]]:
        """
        Calculate available time slots for studying.
        Simplified implementation - assumes 8am-10pm as potential study hours.
        """
        available_slots = []
        
        try:
            start = datetime.strptime(start_date, "%Y-%m-%d")
            end = datetime.strptime(end_date, "%Y-%m-%d")
            
            current = start
            while current <= end:
                date_str = current.strftime("%Y-%m-%d")
                
                # Get events for this day
                day_events = [
                    e for e in events
                    if e.get('start', '').startswith(date_str)
                ]
                
                # Default available hours (8am-10pm)
                # Subtract busy hours
                busy_hours = sum(
                    self._get_event_duration(e) for e in day_events
                )
                available_hours = max(0, 14 - busy_hours)  # 14 hours (8am-10pm)
                
                if available_hours >= 1:
                    available_slots.append({
                        "date": date_str,
                        "available_hours": available_hours,
                        "suggested_slots": self._suggest_time_slots(day_events)
                    })
                
                current += timedelta(days=1)
                
        except Exception as e:
            logger.error(f"Error calculating available slots: {str(e)}")
        
        return available_slots
    
    def _get_event_duration(self, event: Dict[str, Any]) -> float:
        """Get duration of an event in hours."""
        try:
            if event.get('all_day'):
                return 8.0  # Assume 8 hours for all-day events
            
            start = event.get('start', '')
            end = event.get('end', '')
            
            if 'T' in start and 'T' in end:
                start_dt = datetime.fromisoformat(start.replace('Z', '+00:00'))
                end_dt = datetime.fromisoformat(end.replace('Z', '+00:00'))
                return (end_dt - start_dt).total_seconds() / 3600
            
            return 1.0  # Default 1 hour
        except Exception:
            return 1.0
    
    def _suggest_time_slots(
        self,
        day_events: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """Suggest available time slots around existing events."""
        # Simplified implementation
        if not day_events:
            return [
                {"start": "09:00", "end": "12:00"},
                {"start": "14:00", "end": "17:00"},
                {"start": "19:00", "end": "21:00"}
            ]
        
        # Basic morning/evening slots when events exist
        return [
            {"start": "09:00", "end": "11:00"},
            {"start": "19:00", "end": "21:00"}
        ]
    
    def _estimate_busy_hours(
        self,
        events: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Estimate busy hours per day."""
        busy_hours = {}
        
        for event in events:
            start = event.get('start', '')
            if start:
                date = start.split('T')[0]
                duration = self._get_event_duration(event)
                busy_hours[date] = busy_hours.get(date, 0) + duration
        
        return busy_hours
    
    def _get_fallback_calendar(
        self,
        start_date: str,
        end_date: str
    ) -> Dict[str, Any]:
        """
        Return fallback calendar data when API is unavailable.
        Assumes generic availability.
        """
        logger.info("Using fallback calendar data")
        
        available_slots = []
        try:
            start = datetime.strptime(start_date, "%Y-%m-%d")
            end = datetime.strptime(end_date, "%Y-%m-%d")
            
            current = start
            while current <= end:
                date_str = current.strftime("%Y-%m-%d")
                weekday = current.weekday()
                
                # Less availability on weekends
                if weekday < 5:  # Weekday
                    available_hours = 4.0
                else:  # Weekend
                    available_hours = 6.0
                
                available_slots.append({
                    "date": date_str,
                    "available_hours": available_hours,
                    "suggested_slots": [
                        {"start": "09:00", "end": "12:00"},
                        {"start": "14:00", "end": "17:00"},
                        {"start": "19:00", "end": "21:00"}
                    ]
                })
                
                current += timedelta(days=1)
                
        except Exception as e:
            logger.error(f"Error creating fallback calendar: {str(e)}")
        
        return {
            "events": [],
            "total_events": 0,
            "deadlines": [],
            "available_slots": available_slots,
            "busy_hours_per_day": {},
            "start_date": start_date,
            "end_date": end_date,
            "is_fallback": True,
            "message": "Calendar data unavailable. Using estimated availability."
        }


# Singleton instance
calendar_service = CalendarService()
