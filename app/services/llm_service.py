"""
LLM service using Google Gemini with function calling support.
Handles RAG answer generation and study plan creation.
"""

import logging
import json
from typing import List, Dict, Any, Optional, AsyncGenerator

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from app.config import get_settings

logger = logging.getLogger(__name__)


# System prompts
RAG_SYSTEM_PROMPT = """You are an AI study assistant helping students understand their course materials. 

INSTRUCTIONS:
1. Answer questions ONLY based on the provided context from course materials
2. Always cite your sources by mentioning which document/section the information comes from
3. If the context doesn't contain enough information to answer, clearly state: "I don't have enough information in the course materials to answer this question fully."
4. Be concise but thorough - aim for 2-4 paragraphs
5. Use bullet points or numbered lists when appropriate for clarity
6. If you notice related topics in the context that might be helpful, briefly mention them

FORMAT:
- Start with a direct answer to the question
- Support with evidence from the context
- End with source citations

Remember: You are helping students learn, so explain concepts clearly."""

STUDY_PLAN_SYSTEM_PROMPT = """You are an AI study planning assistant. Your job is to create personalized, realistic study schedules.

INSTRUCTIONS:
1. First, fetch the user's calendar to understand their existing commitments
2. Analyze their available time slots based on calendar data
3. Create a study plan that:
   - Respects existing calendar events and deadlines
   - Distributes topics evenly across available time
   - Includes regular breaks (15-30 min every 1-2 hours)
   - Builds in review sessions
   - Accounts for topic difficulty (harder topics = more time)
   - Leaves buffer time for unexpected events

4. If calendar fetch fails, create a generic plan with reasonable assumptions

OUTPUT FORMAT:
Return a valid JSON object with this structure:
{
    "weekly_schedule": [
        {
            "week_number": 1,
            "start_date": "YYYY-MM-DD",
            "end_date": "YYYY-MM-DD",
            "sessions": [
                {
                    "date": "YYYY-MM-DD",
                    "start_time": "HH:MM",
                    "end_time": "HH:MM",
                    "topic": "Topic name",
                    "learning_objectives": ["objective 1", "objective 2"],
                    "resources": ["resource suggestion"],
                    "notes": "optional notes"
                }
            ],
            "weekly_goals": ["goal 1", "goal 2"],
            "estimated_hours": 10.5
        }
    ],
    "milestones": [
        {"week": 2, "milestone": "Complete fundamentals", "verification": "Quiz or self-test"}
    ],
    "calendar_conflicts": [
        {"date": "YYYY-MM-DD", "event_title": "Event name", "conflict_type": "overlap", "suggestion": "Move session to morning"}
    ],
    "adjustment_tips": ["Tip 1", "Tip 2"],
    "total_hours": 42.5
}"""


class LLMService:
    """
    Handles LLM interactions using Google Gemini.
    Supports function calling for calendar integration.
    """
    
    def __init__(self):
        settings = get_settings()
        genai.configure(api_key=settings.gemini_api_key)
        
        self.model_name = settings.gemini_model
        
        # Safety settings - allow educational content
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        }
        
        # Generation config
        self.generation_config = genai.GenerationConfig(
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            max_output_tokens=2048,
        )
    
    async def generate_answer(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]]
    ) -> str:
        """
        Generate an answer using RAG with retrieved context.
        
        Args:
            query: User's question
            context_chunks: Retrieved relevant chunks
            
        Returns:
            Generated answer with citations
        """
        # Build context string from chunks
        context_parts = []
        for i, chunk in enumerate(context_chunks, 1):
            source = chunk.get('section_title') or f"Document {chunk.get('file_id', 'unknown')[:8]}"
            context_parts.append(f"[Source {i}: {source}]\n{chunk['content']}\n")
        
        context_text = "\n---\n".join(context_parts)
        
        # Build prompt
        prompt = f"""CONTEXT FROM COURSE MATERIALS:
{context_text}

STUDENT QUESTION:
{query}

Please provide a helpful answer based on the context above."""

        try:
            model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
                system_instruction=RAG_SYSTEM_PROMPT
            )
            
            response = model.generate_content(prompt)
            
            if response.text:
                return response.text
            else:
                logger.warning("Empty response from Gemini")
                return "I apologize, but I couldn't generate a response. Please try rephrasing your question."
                
        except Exception as e:
            logger.error(f"LLM generation failed: {str(e)}")
            raise
    
    async def generate_answer_stream(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]]
    ) -> AsyncGenerator[str, None]:
        """
        Generate answer with streaming for SSE support.
        
        Yields:
            Chunks of the generated answer
        """
        context_parts = []
        for i, chunk in enumerate(context_chunks, 1):
            source = chunk.get('section_title') or f"Document {chunk.get('file_id', 'unknown')[:8]}"
            context_parts.append(f"[Source {i}: {source}]\n{chunk['content']}\n")
        
        context_text = "\n---\n".join(context_parts)
        
        prompt = f"""CONTEXT FROM COURSE MATERIALS:
{context_text}

STUDENT QUESTION:
{query}

Please provide a helpful answer based on the context above."""

        try:
            model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
                system_instruction=RAG_SYSTEM_PROMPT
            )
            
            response = model.generate_content(prompt, stream=True)
            
            for chunk in response:
                if chunk.text:
                    yield chunk.text
                    
        except Exception as e:
            logger.error(f"Streaming generation failed: {str(e)}")
            yield f"Error generating response: {str(e)}"
    
    def get_calendar_tool_definition(self) -> Dict[str, Any]:
        """
        Get the function calling tool definition for calendar API.
        
        Returns:
            Tool definition dict for Gemini
        """
        return {
            "function_declarations": [
                {
                    "name": "get_user_calendar",
                    "description": "Fetch the user's calendar events and commitments for a date range. Use this to understand the user's schedule and find available time slots for studying.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "user_id": {
                                "type": "string",
                                "description": "The ID of the user whose calendar to fetch"
                            },
                            "start_date": {
                                "type": "string",
                                "description": "Start date in YYYY-MM-DD format"
                            },
                            "end_date": {
                                "type": "string",
                                "description": "End date in YYYY-MM-DD format"
                            }
                        },
                        "required": ["user_id", "start_date", "end_date"]
                    }
                }
            ]
        }
    
    async def generate_study_plan_with_function_calling(
        self,
        user_id: str,
        goal: str,
        topics: List[str],
        start_date: str,
        end_date: str,
        difficulty_level: str,
        preferences: Optional[Dict[str, Any]],
        calendar_fetcher: callable
    ) -> Dict[str, Any]:
        """
        Generate a study plan using function calling for calendar integration.
        
        Args:
            user_id: User ID
            goal: Study goal
            topics: List of topics to cover
            start_date: Plan start date
            end_date: Plan end date
            difficulty_level: beginner/intermediate/advanced
            preferences: User study preferences
            calendar_fetcher: Async function to fetch calendar data
            
        Returns:
            Study plan as structured dict
        """
        # Build the prompt
        pref_text = ""
        if preferences:
            pref_text = f"""
User Preferences:
- Preferred study session duration: {preferences.get('session_duration_minutes', 60)} minutes
- Break duration: {preferences.get('break_duration_minutes', 15)} minutes
- Study days per week: {preferences.get('days_per_week', 5)}
"""
        
        prompt = f"""Create a personalized study plan with these details:

USER ID: {user_id}
GOAL: {goal}
TOPICS TO COVER: {', '.join(topics)}
DATE RANGE: {start_date} to {end_date}
DIFFICULTY LEVEL: {difficulty_level}
{pref_text}

First, fetch the user's calendar to understand their existing commitments and availability.
Then create a realistic study schedule that works around their existing events.

Return the study plan as a valid JSON object following the specified format."""

        try:
            # Initialize model with tools
            tools = self.get_calendar_tool_definition()
            
            model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=genai.GenerationConfig(
                    temperature=0.5,
                    max_output_tokens=4096,
                ),
                safety_settings=self.safety_settings,
                system_instruction=STUDY_PLAN_SYSTEM_PROMPT,
                tools=[tools]
            )
            
            chat = model.start_chat()
            
            # Send initial message
            response = chat.send_message(prompt)
            
            # Handle function calls in a loop
            max_iterations = 3
            iteration = 0
            
            while iteration < max_iterations:
                iteration += 1
                
                # Check for function calls
                if response.candidates and response.candidates[0].content.parts:
                    has_function_call = False
                    
                    for part in response.candidates[0].content.parts:
                        if hasattr(part, 'function_call') and part.function_call:
                            has_function_call = True
                            func_call = part.function_call
                            
                            if func_call.name == "get_user_calendar":
                                # Extract arguments
                                args = dict(func_call.args)
                                logger.info(f"Function call: get_user_calendar with args: {args}")
                                
                                # Execute the calendar fetch
                                try:
                                    calendar_data = await calendar_fetcher(
                                        args.get('user_id', user_id),
                                        args.get('start_date', start_date),
                                        args.get('end_date', end_date)
                                    )
                                    function_result = json.dumps(calendar_data)
                                except Exception as e:
                                    logger.error(f"Calendar fetch failed: {str(e)}")
                                    function_result = json.dumps({
                                        "error": str(e),
                                        "events": [],
                                        "message": "Calendar unavailable, please create plan with generic availability"
                                    })
                                
                                # Send function response back
                                response = chat.send_message(
                                    genai.protos.Content(
                                        parts=[
                                            genai.protos.Part(
                                                function_response=genai.protos.FunctionResponse(
                                                    name="get_user_calendar",
                                                    response={"result": function_result}
                                                )
                                            )
                                        ]
                                    )
                                )
                                break
                    
                    if not has_function_call:
                        # No more function calls, we have the final response
                        break
                else:
                    break
            
            # Extract the final response
            if response.text:
                # Parse JSON from response
                return self._parse_study_plan_response(response.text)
            else:
                raise ValueError("Empty response from model")
                
        except Exception as e:
            logger.error(f"Study plan generation failed: {str(e)}")
            raise
    
    def _parse_study_plan_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse the study plan JSON from the model response.
        
        Args:
            response_text: Raw response text
            
        Returns:
            Parsed study plan dict
        """
        # Try to extract JSON from the response
        text = response_text.strip()
        
        # Remove markdown code blocks if present
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        
        text = text.strip()
        
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse study plan JSON: {str(e)}")
            logger.debug(f"Raw response: {response_text[:500]}")
            
            # Return a basic structure on parse failure
            return {
                "error": "Failed to parse study plan",
                "raw_response": response_text[:1000],
                "weekly_schedule": [],
                "milestones": [],
                "calendar_conflicts": [],
                "adjustment_tips": ["Please try generating the plan again"]
            }
    
    async def health_check(self) -> bool:
        """Check if Gemini API is accessible."""
        try:
            model = genai.GenerativeModel(model_name=self.model_name)
            response = model.generate_content("Say 'OK' if you can read this.")
            return bool(response.text)
        except Exception as e:
            logger.error(f"Gemini health check failed: {str(e)}")
            return False


# Singleton instance
llm_service = LLMService()
