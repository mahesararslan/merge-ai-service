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
        attachment_result: Optional[Dict[str, Any]] = None,
        attachment_original_name: Optional[str] = None
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
                logger.info(
                    f"[ATTACHMENT] Flow 2 Processing: Chunking and embedding attachment for vector storage"
                )
                
                # Chunk the extracted text
                extracted_text = attachment_result.get('extracted_content') or ""
                if not extracted_text:
                    # Re-extract if not available (shouldn't happen but safety check)
                    logger.warning("[ATTACHMENT] ⚠ Flow 2: No extracted text, cannot process attachment")
                else:
                    logger.info(f"[ATTACHMENT] Chunking {len(extracted_text)} chars of text...")
                    chunks = self.chunking_service.chunk_text(extracted_text)
                    chunks_created_for_attachment = len(chunks)
                    
                    logger.info(f"[ATTACHMENT] ✓ Created {chunks_created_for_attachment} chunks from attachment")
                    
                    # Generate embeddings for all chunks
                    chunk_texts = [chunk.content for chunk in chunks]
                    logger.info(f"[ATTACHMENT] Generating embeddings for {len(chunk_texts)} chunks...")
                    embeddings = await embedding_service.embed_documents(chunk_texts)
                    logger.info(f"[ATTACHMENT] ✓ Generated {len(embeddings)} embeddings")
                    
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
                    logger.info(f"[ATTACHMENT] Upserting {len(points_to_insert)} points to Qdrant...")
                    await vector_store_service.upsert_temp_attachment_chunks(
                        conversation_id=conversation_id,
                        points=points_to_insert
                    )
                    
                    logger.info(
                        f"[ATTACHMENT] ✓ Flow 2 Complete: Stored {chunks_created_for_attachment} chunks "
                        f"in vector DB for conversation {conversation_id}, TTL: {ttl_expires_at.isoformat()}"
                    )
            
            # ─── Strict priority: attachment > rooms > general knowledge ───
            #
            # The user has explicitly asked us NOT to mix sources. So:
            #   1) If anything is attached (fresh injection, stored Flow 1
            #      context, or Flow 2 vector chunks), answer ONLY from the
            #      attachment. Do not search room materials at all.
            #   2) Else, search the user's rooms in Qdrant.
            #   3) Else, fall through to general LLM knowledge.
            fresh_name = attachment_original_name or "most recently attached file"
            fresh_text: Optional[str] = None
            if attachment_result and attachment_result.get('flow') == 'direct_injection':
                fresh_text = attachment_result.get('extracted_content') or None
            has_fresh = bool(fresh_text)
            has_stored = bool(attachment_context)
            has_attachment = has_fresh or has_stored or bool(has_vector_attachment)

            search_results: List[Dict[str, Any]] = []

            if has_attachment:
                # Flow 2 still needs a vector search to pick relevant chunks
                # from the attached file's vectors. Flow 1 already has the
                # full extracted text in attachment_context — nothing to search.
                if has_vector_attachment and conversation_id:
                    logger.info(
                        f"Attachment-only retrieval for conversation {conversation_id}"
                    )
                    query_embedding = await embedding_service.embed_query(query)
                    search_results = await vector_store_service.search_by_conversation(
                        query_embedding=query_embedding,
                        conversation_id=conversation_id,
                        top_k=k,
                        min_score=0.1,
                    )
                    if not search_results:
                        logger.info("No attachment chunks above 0.1 — retrying without threshold")
                        search_results = await vector_store_service.search_by_conversation(
                            query_embedding=query_embedding,
                            conversation_id=conversation_id,
                            top_k=k,
                            min_score=0.0,
                        )
                    logger.info(f"Retrieved {len(search_results)} chunks from attached file")
                else:
                    logger.info("[LLM] Attachment-only mode (Flow 1) — skipping room search")
            else:
                # No attachment → search the user's rooms
                query_embedding = await embedding_service.embed_query(query)
                search_results = await vector_store_service.search(
                    query_embedding=query_embedding,
                    room_ids=room_ids,
                    file_id=context_file_id,
                    top_k=k,
                    min_score=min_score
                )
                if not search_results:
                    logger.info("[LLM] No room chunks found — falling back to general knowledge")

            if has_fresh and has_stored:
                final_attachment_context = (
                    f"[MOST RECENTLY ATTACHED FILE — if the user says \"this file\" they mean this one]\n"
                    f"## File: {fresh_name}\n\n{fresh_text}"
                    f"\n\n---\n\n"
                    f"[PREVIOUSLY ATTACHED FILES IN THIS CONVERSATION — only refer to these if the user's question is explicitly about them]\n"
                    f"{attachment_context}"
                )
            elif has_fresh:
                final_attachment_context = f"## File: {fresh_name}\n\n{fresh_text}"
            elif has_stored:
                final_attachment_context = attachment_context
            else:
                final_attachment_context = None

            if final_attachment_context:
                logger.info(
                    f"[LLM] Using combined attachment context ({len(final_attachment_context)} chars) + {len(search_results)} chunks"
                )
            elif not search_results:
                logger.info(f"[LLM] No relevant chunks or attachment found - using general LLM knowledge")
            else:
                logger.info(f"[LLM] Using {len(search_results)} chunks from vector store")
            
            # Generate answer with available context (chunks, attachment, or nothing)
            answer = await llm_service.generate_answer(
                query=query,
                context_chunks=search_results if search_results else [],
                conversation_history=conversation_history,
                conversation_summary=conversation_summary,
                attachment_context=final_attachment_context
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
                logger.info(
                    f"[ATTACHMENT] Adding chunks_created_for_attachment={chunks_created_for_attachment} to response"
                )
            
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
        has_vector_attachment: Optional[bool] = False,
        attachment_result: Optional[Dict[str, Any]] = None,
        attachment_original_name: Optional[str] = None
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

        chunks_created_for_attachment = 0

        try:
            # Yield status event
            yield {
                "event": "status",
                "data": {"status": "searching", "message": "Searching course materials..."}
            }

            # Handle Flow 2: Chunk and embed attachment for vector storage
            if attachment_result and attachment_result['flow'] == 'vector_storage':
                logger.info(
                    f"[ATTACHMENT] Stream Flow 2: Chunking and embedding attachment for vector storage"
                )

                extracted_text = attachment_result.get('extracted_content') or ""
                if not extracted_text:
                    logger.warning("[ATTACHMENT] ⚠ Stream Flow 2: No extracted text, cannot process attachment")
                else:
                    logger.info(f"[ATTACHMENT] Chunking {len(extracted_text)} chars of text...")
                    chunks = self.chunking_service.chunk_text(extracted_text)
                    chunks_created_for_attachment = len(chunks)

                    logger.info(f"[ATTACHMENT] ✓ Created {chunks_created_for_attachment} chunks from attachment")

                    chunk_texts = [chunk.content for chunk in chunks]
                    logger.info(f"[ATTACHMENT] Generating embeddings for {len(chunk_texts)} chunks...")
                    embeddings = await embedding_service.embed_documents(chunk_texts)
                    logger.info(f"[ATTACHMENT] ✓ Generated {len(embeddings)} embeddings")

                    ttl_expires_at = datetime.utcnow() + timedelta(days=self.settings.temp_vector_ttl_days)

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

                    logger.info(f"[ATTACHMENT] Upserting {len(points_to_insert)} points to Qdrant...")
                    await vector_store_service.upsert_temp_attachment_chunks(
                        conversation_id=conversation_id,
                        points=points_to_insert
                    )

                    logger.info(
                        f"[ATTACHMENT] ✓ Stream Flow 2 Complete: Stored {chunks_created_for_attachment} chunks "
                        f"in vector DB for conversation {conversation_id}"
                    )

                    # Mark for dual retrieval below
                    has_vector_attachment = True

            # ─── Strict priority: attachment > rooms > general knowledge ───
            # Mirrors the non-streaming path. Compute attachment state first,
            # then either attachment-only retrieval or rooms-only retrieval.
            fresh_text_check: Optional[str] = None
            if attachment_result and attachment_result.get('flow') == 'direct_injection':
                fresh_text_check = attachment_result.get('extracted_content') or None
            has_attachment = (
                bool(fresh_text_check)
                or bool(attachment_context)
                or bool(has_vector_attachment)
            )

            search_results: List[Dict[str, Any]] = []

            if has_attachment:
                if has_vector_attachment and conversation_id:
                    logger.info(
                        f"Stream: Attachment-only retrieval for conversation {conversation_id}"
                    )
                    query_embedding = await embedding_service.embed_query(query)
                    search_results = await vector_store_service.search_by_conversation(
                        query_embedding=query_embedding,
                        conversation_id=conversation_id,
                        top_k=k,
                        min_score=0.1,
                    )
                    if not search_results:
                        logger.info("Stream: No attachment chunks above 0.1 — retrying without threshold")
                        search_results = await vector_store_service.search_by_conversation(
                            query_embedding=query_embedding,
                            conversation_id=conversation_id,
                            top_k=k,
                            min_score=0.0,
                        )
                    logger.info(f"Stream: Retrieved {len(search_results)} chunks from attached file")
                else:
                    logger.info("[LLM] Stream: Attachment-only mode (Flow 1) — skipping room search")
            else:
                # No attachment → search the user's rooms
                query_embedding = await embedding_service.embed_query(query)
                search_results = await vector_store_service.search(
                    query_embedding=query_embedding,
                    room_ids=room_ids,
                    file_id=context_file_id,
                    top_k=k,
                    min_score=min_score
                )
                if not search_results:
                    logger.info("[LLM] Stream: No room chunks found — falling back to general knowledge")

            # Yield sources if found
            if search_results:
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
            else:
                logger.info(f"No relevant chunks found - using general LLM knowledge")

            yield {
                "event": "status",
                "data": {"status": "generating", "message": "Generating answer..."}
            }

            # Build the attachment block for the prompt.
            #
            # There are three possible states for any query:
            #   A) Only a newly attached file (single-file first message).
            #   B) Only stored context (follow-up on an existing conversation).
            #   C) Both (user just attached a second file to an ongoing chat).
            #
            # For state C the newly attached file goes FIRST and is tagged
            # as "most recently attached" — when the user says "this file",
            # they mean the one they just attached, and Gemini needs a
            # strong signal to prefer it over the older/larger prior file.
            # In A and B there is no ambiguity so we drop the scaffolding.
            fresh_name = attachment_original_name or "most recently attached file"
            fresh_text: Optional[str] = None
            if attachment_result and attachment_result.get('flow') == 'direct_injection':
                fresh_text = attachment_result.get('extracted_content') or None

            has_fresh = bool(fresh_text)
            has_stored = bool(attachment_context)

            if has_fresh and has_stored:
                final_attachment_context = (
                    f"[MOST RECENTLY ATTACHED FILE — if the user says \"this file\" they mean this one]\n"
                    f"## File: {fresh_name}\n\n{fresh_text}"
                    f"\n\n---\n\n"
                    f"[PREVIOUSLY ATTACHED FILES IN THIS CONVERSATION — only refer to these if the user's question is explicitly about them]\n"
                    f"{attachment_context}"
                )
                logger.info(
                    f"[LLM] Stream: Multi-file prompt — fresh '{fresh_name}' ({len(fresh_text or '')} chars) + stored ({len(attachment_context)} chars)"
                )
            elif has_fresh:
                final_attachment_context = f"## File: {fresh_name}\n\n{fresh_text}"
                logger.info(
                    f"[LLM] Stream: Single-file prompt — '{fresh_name}' ({len(fresh_text or '')} chars)"
                )
            elif has_stored:
                final_attachment_context = attachment_context
                logger.info(
                    f"[LLM] Stream: Stored context only ({len(attachment_context)} chars)"
                )
            else:
                final_attachment_context = None

            # Step 3: Stream the answer with conversation context and attachment
            full_answer = ""
            async for chunk in llm_service.generate_answer_stream(
                query=query,
                context_chunks=search_results if search_results else [],
                conversation_history=conversation_history,
                conversation_summary=conversation_summary,
                attachment_context=final_attachment_context
            ):
                full_answer += chunk
                yield {
                    "event": "chunk",
                    "data": {"text": chunk}
                }

            processing_time = (time.time() - start_time) * 1000

            # Build complete event data with attachment metadata
            complete_data = {
                "processing_time_ms": processing_time,
                "chunks_retrieved": len(search_results) if search_results else 0
            }

            if attachment_result:
                complete_data["flow_used"] = attachment_result['flow']
                if attachment_result['flow'] == 'direct_injection':
                    complete_data["extracted_content"] = attachment_result.get('extracted_content')
                    complete_data["extracted_content_length"] = attachment_result.get('char_count', 0)
                elif attachment_result['flow'] == 'vector_storage':
                    complete_data["chunks_created_for_attachment"] = chunks_created_for_attachment

            # Final complete event
            yield {
                "event": "complete",
                "data": complete_data
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
