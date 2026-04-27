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

    @staticmethod
    def _is_conversational_query(query: str) -> bool:
        """
        Heuristic: is this question pure chitchat that has no chance of
        being answered from course materials? (e.g. "hi", "thanks",
        "how are you")

        We skip vector search for these — running it always returns
        whatever the room has and surfaces those as "sources" on the
        UI even when the answer is just "Hello!". Cheap, defensive
        match: short messages + a small allowlist of greeting starters.
        """
        q = (query or "").strip().lower()
        if not q:
            return True
        # Strip trailing punctuation
        while q and q[-1] in "?!.,":
            q = q[:-1]
        if not q:
            return True

        # Anything 2 words or fewer that isn't an actual term lookup is
        # almost certainly chitchat.
        if len(q.split()) <= 2 and len(q) <= 12:
            return True

        chitchat_prefixes = (
            "hi", "hello", "hey", "yo", "sup",
            "thanks", "thank you", "thx",
            "ok", "okay", "cool", "nice", "great",
            "good morning", "good afternoon", "good evening", "good night",
            "how are you", "how's it going", "hows it going",
            "what's up", "whats up",
            "bye", "goodbye", "see you", "see ya",
            "yes", "yeah", "yep", "no", "nope",
        )
        if any(q == p or q.startswith(p + " ") for p in chitchat_prefixes):
            return True
        return False

    def _build_attachment_block(
        self,
        fresh_name: str,
        fresh_text: Optional[str],
        stored_list: List[Any],
        no_fresh_attachment: bool,
    ) -> Optional[str]:
        """
        Assemble the attachment portion of the prompt.

        Cases:
          - Fresh attachment + zero stored:
              just the fresh file, no scaffolding.
          - Fresh attachment + stored:
              fresh file is tagged "MOST RECENTLY ATTACHED — 'this file'
              means this one", stored files are tagged "previously
              attached, only use if the question is explicitly about
              them". This is the multi-file case where the user just
              dropped a new file into an ongoing conversation.
          - No fresh attachment + stored:
              the LATEST stored entry (first in the list) takes the
              "this file" role, so generic follow-ups like "summarise
              this file" resolve to the most recently uploaded file.
              Older stored entries are tagged "previously attached".
          - No fresh, no stored:
              returns None.
        """
        has_fresh = bool(fresh_text)
        if not has_fresh and not stored_list:
            return None

        if has_fresh:
            if not stored_list:
                return f"## File: {fresh_name}\n\n{fresh_text}"
            stored_block = "\n\n---\n\n".join(
                f"## File: {a.name if hasattr(a, 'name') else a['name']}\n\n"
                f"{a.content if hasattr(a, 'content') else a['content']}"
                for a in stored_list
            )
            return (
                "[MOST RECENTLY ATTACHED FILE — if the user says \"this file\" they mean this one]\n"
                f"## File: {fresh_name}\n\n{fresh_text}"
                "\n\n---\n\n"
                "[PREVIOUSLY ATTACHED FILES IN THIS CONVERSATION — only refer to these if the user's question is explicitly about them]\n"
                f"{stored_block}"
            )

        # No fresh attachment, stored entries only. List is latest-first.
        latest = stored_list[0]
        latest_name = latest.name if hasattr(latest, 'name') else latest['name']
        latest_content = latest.content if hasattr(latest, 'content') else latest['content']

        if len(stored_list) == 1:
            # Only one prior file — drop the scaffolding, just label it
            # so the model knows "this file" refers to it.
            return (
                "[ATTACHED FILE IN THIS CONVERSATION — if the user says \"this file\" they mean this one]\n"
                f"## File: {latest_name}\n\n{latest_content}"
            )

        older = stored_list[1:]
        older_block = "\n\n---\n\n".join(
            f"## File: {a.name if hasattr(a, 'name') else a['name']}\n\n"
            f"{a.content if hasattr(a, 'content') else a['content']}"
            for a in older
        )
        return (
            "[MOST RECENTLY ATTACHED FILE — if the user says \"this file\" they mean this one]\n"
            f"## File: {latest_name}\n\n{latest_content}"
            "\n\n---\n\n"
            "[PREVIOUSLY ATTACHED FILES IN THIS CONVERSATION — only refer to these if the user's question is explicitly about them]\n"
            f"{older_block}"
        )
    
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
        stored_attachments: Optional[List[Any]] = None,
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
            stored_attachments: Previously-extracted Flow 1 attachments,
                ordered LATEST-FIRST. The first entry is the referent
                for "this file" on follow-up turns.
            has_vector_attachment: Whether any attachment uses Flow 2
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
            
            # ─── Routing rules ───────────────────────────────────────────
            #
            #   1) Fresh attachment (or Flow 2 vectors) this turn → answer
            #      ONLY from the attached file(s). Do not search rooms.
            #   2) No fresh attachment → ALWAYS search the user's rooms.
            #      Stored context from prior turns still rides along in
            #      the prompt so the LLM can summarise/refer back when
            #      asked, but it does not gate room search.
            #   3) If neither attachment nor room hits surface anything →
            #      fall through to general LLM knowledge.
            #
            # Stored attachments arrive ordered LATEST-FIRST. On a follow-
            # up where the user says "this file", the first entry is the
            # referent.
            fresh_name = attachment_original_name or "most recently attached file"
            fresh_text: Optional[str] = None
            if attachment_result and attachment_result.get('flow') == 'direct_injection':
                fresh_text = attachment_result.get('extracted_content') or None
            has_fresh_attachment = bool(fresh_text) or bool(has_vector_attachment)
            stored_list = stored_attachments or []
            is_chitchat = self._is_conversational_query(query)

            search_results: List[Dict[str, Any]] = []

            if has_fresh_attachment:
                # Flow 2 still needs a vector search to pick relevant chunks
                # from the attached file's vectors. Flow 1 already has the
                # full extracted text — nothing to search.
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
                    logger.info("[LLM] Fresh attachment present — skipping room search")
            elif is_chitchat:
                logger.info("[LLM] Chitchat query — skipping room search and sources")
            else:
                # No fresh attachment → search the user's rooms regardless
                # of whether stored context exists.
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

            final_attachment_context = self._build_attachment_block(
                fresh_name=fresh_name,
                fresh_text=fresh_text,
                stored_list=stored_list,
                no_fresh_attachment=not has_fresh_attachment,
            )

            if final_attachment_context:
                logger.info(
                    f"[LLM] Using attachment block ({len(final_attachment_context)} chars) + {len(search_results)} room chunks"
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
        stored_attachments: Optional[List[Any]] = None,
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

            # ─── Routing rules (mirrors non-streaming path) ────────────
            # Fresh attachment (or Flow 2 vectors) → attachment-only mode.
            # Otherwise → always search rooms. See query() for the long
            # comment explaining why stored context no longer disables
            # room search.
            fresh_text_check: Optional[str] = None
            if attachment_result and attachment_result.get('flow') == 'direct_injection':
                fresh_text_check = attachment_result.get('extracted_content') or None
            has_fresh_attachment = (
                bool(fresh_text_check) or bool(has_vector_attachment)
            )

            stored_list = stored_attachments or []
            is_chitchat = self._is_conversational_query(query)

            search_results: List[Dict[str, Any]] = []

            if has_fresh_attachment:
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
                    logger.info("[LLM] Stream: Fresh attachment present — skipping room search")
            elif is_chitchat:
                logger.info("[LLM] Stream: Chitchat query — skipping room search and sources")
            else:
                # No fresh attachment → always search rooms; stored
                # context (if any) is included in the prompt below.
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
            # Note: routing already logged the chosen mode above (fresh
            # attachment vs. room search vs. fallback), so no log here.

            yield {
                "event": "status",
                "data": {"status": "generating", "message": "Generating answer..."}
            }

            # Build the attachment block for the prompt. See
            # _build_attachment_block for the full layering: fresh file
            # first when present, otherwise the latest stored entry takes
            # the "this file" role.
            fresh_name = attachment_original_name or "most recently attached file"
            fresh_text: Optional[str] = None
            if attachment_result and attachment_result.get('flow') == 'direct_injection':
                fresh_text = attachment_result.get('extracted_content') or None

            final_attachment_context = self._build_attachment_block(
                fresh_name=fresh_name,
                fresh_text=fresh_text,
                stored_list=stored_list,
                no_fresh_attachment=not has_fresh_attachment,
            )

            if final_attachment_context:
                logger.info(
                    f"[LLM] Stream: Attachment block ({len(final_attachment_context)} chars), "
                    f"fresh={'yes' if fresh_text else 'no'}, "
                    f"stored={len(stored_list)} file(s)"
                )

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
