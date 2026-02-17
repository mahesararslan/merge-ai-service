"""
Retrieval service that orchestrates the RAG pipeline.
Combines embedding, vector search, and LLM generation.
"""

import logging
import time
from typing import List, Dict, Any, Optional, AsyncGenerator

from app.config import get_settings
from app.services.embedding_service import embedding_service
from app.services.vector_store import vector_store_service
from app.services.llm_service import llm_service
from app.models import QueryResponse, SourceChunk

logger = logging.getLogger(__name__)


class RetrievalService:
    """
    Orchestrates the complete RAG pipeline:
    1. Embed query
    2. Search vector store
    3. Generate answer with context
    """
    
    def __init__(self):
        self.settings = get_settings()
    
    async def query(
        self,
        query: str,
        user_id: str,
        room_ids: List[str],
        context_file_id: Optional[str] = None,
        top_k: Optional[int] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        conversation_summary: Optional[str] = None
    ) -> QueryResponse:
        """
        Execute a RAG query and return a complete response.
        
        Args:
            query: User's question
            user_id: User ID for logging
            room_ids: Rooms to search
            context_file_id: Optional specific file to focus on
            top_k: Number of chunks to retrieve
            conversation_history: Recent conversation messages
            conversation_summary: Summary of older messages
            
        Returns:
            QueryResponse with answer and sources
        """
        start_time = time.time()
        
        k = top_k or self.settings.top_k_results
        min_score = self.settings.min_relevance_score
        
        logger.info(f"Processing query from user {user_id}: {query[:50]}...")
        
        try:
            # Step 1: Generate query embedding
            query_embedding = await embedding_service.embed_query(query)
            
            # Step 2: Search vector store
            search_results = await vector_store_service.search(
                query_embedding=query_embedding,
                room_ids=room_ids,
                file_id=context_file_id,
                top_k=k,
                min_score=min_score
            )
            
            if not search_results:
                logger.info(f"No relevant chunks found for query")
                return QueryResponse(
                    answer="I couldn't find any relevant information in your course materials to answer this question. Please make sure you've uploaded relevant documents to your study room.",
                    sources=[],
                    query=query,
                    processing_time_ms=(time.time() - start_time) * 1000,
                    chunks_retrieved=0
                )
            
            # Step 3: Generate answer with context and conversation history
            answer = await llm_service.generate_answer(
                query=query,
                search_results=search_results,
                conversation_history=conversation_history,
                conversation_summary=conversation_summary
            )
            
            # Format sources
            sources = [
                SourceChunk(
                    file_id=result['file_id'],
                    chunk_index=result['chunk_index'],
                    content=result['content'][:500],  # Truncate for response
                    relevance_score=result['score'],
                    section_title=result.get('section_title')
                )
                for result in search_results
            ]
            
            processing_time = (time.time() - start_time) * 1000
            
            logger.info(
                f"Query completed in {processing_time:.0f}ms, "
                f"retrieved {len(sources)} chunks"
            )
            
            return QueryResponse(
                answer=answer,
                sources=sources,
                query=query,
                processing_time_ms=processing_time,
                chunks_retrieved=len(sources)
            )
            
        except Exception as e:
            logger.error(f"Query failed: {str(e)}")
            raise
    
    async def query_stream(
        self,
        query: str,
        user_id: str,
        room_ids: List[str],
        context_file_id: Optional[str] = None,
        top_k: Optional[int] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        conversation_summary: Optional[str] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Execute a RAG query with streaming response.
        
        Yields:
            Dict events for SSE streaming
        """
        start_time = time.time()
        
        k = top_k or self.settings.top_k_results
        min_score = self.settings.min_relevance_score
        
        logger.info(f"Processing streaming query from user {user_id}")
        
        try:
            # Yield status event
            yield {
                "event": "status",
                "data": {"status": "searching", "message": "Searching course materials..."}
            }
            
            # Step 1: Generate query embedding
            query_embedding = await embedding_service.embed_query(query)
            
            # Step 2: Search vector store
            search_results = await vector_store_service.search(
                query_embedding=query_embedding,
                room_ids=room_ids,
                file_id=context_file_id,
                top_k=k,
                min_score=min_score
            )
            
            if not search_results:
                yield {
                    "event": "complete",
                    "data": {
                        "answer": "I couldn't find any relevant information in your course materials.",
                        "sources": [],
                        "chunks_retrieved": 0
                    }
                }
                return
            
            # Yield sources first
            sources = [
                {
                    "file_id": result['file_id'],
                    "chunk_index": result['chunk_index'],
                    "relevance_score": result['score'],
                    "section_title": result.get('section_title')
                }
                for result in search_results
            ]
            
            yield {
                "event": "sources",
                "data": {"sources": sources, "count": len(sources)}
            }
            
            yield {
                "event": "status",
                "data": {"status": "generating", "message": "Generating answer..."}
            }
            
            # Step 3: Stream the answer with conversation context
            full_answer = ""
            async for chunk in llm_service.generate_answer_stream(
                query=query,
                search_results=search_results,
                conversation_history=conversation_history,
                conversation_summary=conversation_summary
            ):
                full_answer += chunk
                yield {
                    "event": "chunk",
                    "data": {"text": chunk}
                }
            
            processing_time = (time.time() - start_time) * 1000
            
            # Final complete event
            yield {
                "event": "complete",
                "data": {
                    "processing_time_ms": processing_time,
                    "chunks_retrieved": len(sources)
                }
            }
            
            logger.info(f"Streaming query completed in {processing_time:.0f}ms")
            
        except Exception as e:
            logger.error(f"Streaming query failed: {str(e)}")
            yield {
                "event": "error",
                "data": {"error": str(e)}
            }


# Singleton instance
retrieval_service = RetrievalService()
