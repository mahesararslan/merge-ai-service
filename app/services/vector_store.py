"""
Vector store service using Qdrant Cloud.
Handles storage and retrieval of document embeddings.
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import (
    VectorParams,
    Distance,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    MatchAny,
)

from app.config import get_settings
from app.services.chunking_service import TextChunk

logger = logging.getLogger(__name__)


class VectorStoreService:
    """
    Handles vector storage and retrieval using Qdrant Cloud.
    Optimized for memory efficiency with cloud-based operations.
    """
    
    def __init__(self):
        settings = get_settings()
        
        # Initialize Qdrant client
        if settings.qdrant_api_key:
            self.client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key,
            )
        else:
            self.client = QdrantClient(url=settings.qdrant_url)
        
        self.collection_name = settings.collection_name
        self.dimension = settings.embedding_dimension
        
        # Ensure collection exists
        self._ensure_collection()
    
    def _ensure_collection(self):
        """Create collection if it doesn't exist."""
        try:
            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)
            
            if not exists:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.dimension,
                        distance=Distance.COSINE
                    )
                )
                
                # Create payload indexes for fast filtering
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="room_id",
                    field_schema="keyword"
                )
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="file_id",
                    field_schema="keyword"
                )
                
                logger.info(f"Created collection: {self.collection_name}")
            else:
                logger.info(f"Collection exists: {self.collection_name}")
                
        except Exception as e:
            logger.error(f"Failed to ensure collection: {str(e)}")
            raise
    
    async def store_chunks(
        self,
        chunks: List[TextChunk],
        embeddings: List[List[float]],
        room_id: str,
        file_id: str,
        document_type: str
    ) -> int:
        """
        Store document chunks with their embeddings.
        
        Args:
            chunks: List of text chunks
            embeddings: Corresponding embedding vectors
            room_id: Study room ID
            file_id: Document file ID
            document_type: Type of document
            
        Returns:
            Number of points stored
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Chunks and embeddings must have same length")
        
        if not chunks:
            return 0
        
        try:
            points = []
            timestamp = datetime.utcnow().isoformat()
            
            for chunk, embedding in zip(chunks, embeddings):
                point_id = str(uuid.uuid4())
                
                payload = {
                    "room_id": room_id,
                    "file_id": file_id,
                    "document_type": document_type,
                    "chunk_index": chunk.chunk_index,
                    "total_chunks": chunk.total_chunks,
                    "section_title": chunk.section_title,
                    "char_count": chunk.char_count,
                    "content": chunk.content,
                    "timestamp": timestamp
                }
                
                points.append(PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload
                ))
            
            # Batch upsert for efficiency
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True
            )
            
            logger.info(f"Stored {len(points)} vectors for file {file_id}")
            return len(points)
            
        except Exception as e:
            logger.error(f"Failed to store chunks: {str(e)}")
            raise
    
    async def search(
        self,
        query_embedding: List[float],
        room_ids: List[str],
        file_id: Optional[str] = None,
        top_k: int = 5,
        min_score: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors with metadata filtering.
        
        Args:
            query_embedding: Query vector
            room_ids: List of room IDs to search within
            file_id: Optional specific file to search
            top_k: Number of results to return
            min_score: Minimum relevance score threshold
            
        Returns:
            List of search results with content and metadata
        """
        try:
            # Build filter conditions
            must_conditions = [
                FieldCondition(
                    key="room_id",
                    match=MatchAny(any=room_ids)
                )
            ]
            
            if file_id:
                must_conditions.append(
                    FieldCondition(
                        key="file_id",
                        match=MatchValue(value=file_id)
                    )
                )
            
            search_filter = Filter(must=must_conditions)
            
            # Execute search
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                query_filter=search_filter,
                limit=top_k,
                score_threshold=min_score,
                with_payload=True
            )
            
            # Format results
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "id": result.id,
                    "score": result.score,
                    "content": result.payload.get("content", ""),
                    "file_id": result.payload.get("file_id"),
                    "room_id": result.payload.get("room_id"),
                    "chunk_index": result.payload.get("chunk_index"),
                    "section_title": result.payload.get("section_title"),
                    "document_type": result.payload.get("document_type"),
                })
            
            logger.info(f"Found {len(formatted_results)} results above threshold {min_score}")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            raise
    
    async def delete_file(self, file_id: str) -> int:
        """
        Delete all vectors associated with a file.
        
        Args:
            file_id: File ID to delete
            
        Returns:
            Number of points deleted
        """
        try:
            # Count before deletion
            count_before = self.client.count(
                collection_name=self.collection_name,
                count_filter=Filter(
                    must=[
                        FieldCondition(
                            key="file_id",
                            match=MatchValue(value=file_id)
                        )
                    ]
                )
            ).count
            
            # Delete points
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=Filter(
                        must=[
                            FieldCondition(
                                key="file_id",
                                match=MatchValue(value=file_id)
                            )
                        ]
                    )
                )
            )
            
            logger.info(f"Deleted {count_before} vectors for file {file_id}")
            return count_before
            
        except Exception as e:
            logger.error(f"Failed to delete file vectors: {str(e)}")
            raise
    
    async def delete_room(self, room_id: str) -> int:
        """
        Delete all vectors associated with a room.
        
        Args:
            room_id: Room ID to delete
            
        Returns:
            Number of points deleted
        """
        try:
            count_before = self.client.count(
                collection_name=self.collection_name,
                count_filter=Filter(
                    must=[
                        FieldCondition(
                            key="room_id",
                            match=MatchValue(value=room_id)
                        )
                    ]
                )
            ).count
            
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=Filter(
                        must=[
                            FieldCondition(
                                key="room_id",
                                match=MatchValue(value=room_id)
                            )
                        ]
                    )
                )
            )
            
            logger.info(f"Deleted {count_before} vectors for room {room_id}")
            return count_before
            
        except Exception as e:
            logger.error(f"Failed to delete room vectors: {str(e)}")
            raise
    
    async def health_check(self) -> bool:
        """Check if Qdrant is accessible."""
        try:
            collections = self.client.get_collections()
            return True
        except Exception as e:
            logger.error(f"Qdrant health check failed: {str(e)}")
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": info.status
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {str(e)}")
            return {"error": str(e)}


# Singleton instance
vector_store_service = VectorStoreService()
