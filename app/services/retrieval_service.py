"""
Retrieval service that orchestrates the RAG pipeline.
Combines embedding, vector search, and LLM generation.
"""

import logging
import time
from typing import List, Dict, Any, Optional, AsyncGenerator
from datetime import datetime, timedelta

from app.config import get_settings
from app.services.embedding_service import embedding_service
from app.services.vector_store import vector_store_service
from app.services.llm_service import llm_service
from app.services.chunking_service import ChunkingService
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
        self.chunking_service = ChunkingService()
    
    async def query(
        self,
        query: str,
        user_id: str,
        room_ids: List[str],
        context_file_id: Optional[str] = None,
        top_k: Optional[int] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        conversation_summary: Optional[str] = None,
        conversation_id: Optional[str] = None,
        attachment_context: Optional[str] = None,
        has_vector_attachment: Optional[bool] = False,
        attachment_result: Optional[Dict[str, Any]] = None
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
            conversation_id: Conversation ID for filtering temp attachments
            attachment_context: Extracted content for Flow 1
            has_vector_attachment: Whether attachment uses Flow 2
            attachment_result: Processing result from document processor
            
        Returns:
            QueryResponse with answer and sources
        """
        start_time = time.time()
        
        k = top_k or self.settings.top_k_results
        min_score = self.settings.min_relevance_score
        
        logger.info(f"Processing query from user {user_id}: {query[:50]}...")
        
        chunks_created_for_attachment = 0
        
        try:
            # Handle Flow 2: Chunk and embed attachment for vector storage
            if attachment_result and attachment_result['flow'] == 'vector_storage':
                logger.info("Flow 2: Chunking and embedding attachment for vector storage")
                
                # Chunk the extracted text
                extracted_text = attachment_result.get('extracted_content') or ""
                if not extracted_text:
                    # Re-extract if not available (shouldn't happen but safety check)
                    logger.warning("Flow 2: No extracted text, cannot process attachment")
                else:
                    chunks = self.chunking_service.chunk_text(extracted_text)
                    chunks_created_for_attachment = len(chunks)
                    
                    logger.info(f"Created {chunks_created_for_attachment} chunks from attachment")
                    
                    # Generate embeddings for all chunks
                    chunk_texts = [chunk.content for chunk in chunks]
                    embeddings = await embedding_service.embed_documents(chunk_texts)
                    
                    # Calculate TTL expiration
                    ttl_expires_at = datetime.utcnow() + timedelta(days=self.settings.temp_vector_ttl_days)
                    
                    # Prepare points for Qdrant
                    points_to_insert = []
                    for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                        point = {
                            "vector": embedding,
                            "payload": {
                                "conversation_id": conversation_id,
                                "is_temporary": True,
                                "ttl_expires_at": ttl_expires_at.isoformat(),
                                "chunk_index": idx,
                                "total_chunks": chunks_created_for_attachment,
                                "content": chunk.content,
                                "char_count": chunk.char_count,
                                "section_title": chunk.section_title,
                                "timestamp": datetime.utcnow().isoformat()
                            }
                        }
                        points_to_insert.append(point)
                    
                    # Insert into vector store
                    await vector_store_service.upsert_temp_attachment_chunks(
                        conversation_id=conversation_id,
                        points=points_to_insert
                    )
                    
                    logger.info(
                        f"Stored {chunks_created_for_attachment} attachment chunks "
                        f"in vector DB for conversation {conversation_id}"
                    )
            
            # Step 1: Generate query embedding
            query_embedding = await embedding_service.embed_query(query)
            
            # Step 2: Search vector store (dual retrieval if attachment)
            if has_vector_attachment and conversation_id:
                # Dual retrieval: merge results from room_ids and conversation_id
                logger.info("Dual retrieval: Searching both room content and attachment vectors")
                
                # Search room content
                room_results = await vector_store_service.search(
                    query_embedding=query_embedding,
                    room_ids=room_ids,
                    file_id=context_file_id,
                    top_k=k,
                    min_score=min_score
                )
                
                # Search attachment vectors
                attachment_results = await vector_store_service.search_by_conversation(
                    query_embedding=query_embedding,
                    conversation_id=conversation_id,
                    top_k=k,
                    min_score=min_score
                )
                
                # Merge and sort by relevance score
                all_results = room_results + attachment_results
                all_results.sort(key=lambda x: x['score'], reverse=True)
                
                # Take top K from merged results
                search_results = all_results[:k]
                
                logger.info(
                    f"Merged {len(room_results)} room chunks + "
                    f"{len(attachment_results)} attachment chunks = "
                    f"{len(search_results)} total"
                )
            else:
                # Standard search (room_ids only)
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
                context_chunks=search_results,
                conversation_history=conversation_history,
                conversation_summary=conversation_summary,
                attachment_context=attachment_context
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
            
            response = QueryResponse(
                answer=answer,
                sources=sources,
                query=query,
                processing_time_ms=processing_time,
                chunks_retrieved=len(sources)
            )
            
            # Add attachment info if applicable
            if chunks_created_for_attachment > 0:
                response.chunks_created_for_attachment = chunks_created_for_attachment
            
            return response
            
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
        conversation_summary: Optional[str] = None,
        conversation_id: Optional[str] = None,
        attachment_context: Optional[str] = None,
        has_vector_attachment: Optional[bool] = False
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
            
            # Step 3: Stream the answer with conversation context and attachment
            full_answer = ""
            async for chunk in llm_service.generate_answer_stream(
                query=query,
                context_chunks=search_results,
                conversation_history=conversation_history,
                conversation_summary=conversation_summary,
                attachment_context=attachment_context
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
