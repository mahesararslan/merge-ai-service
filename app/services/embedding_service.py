"""
Embedding service using Cohere API.
Cloud-based embeddings to minimize memory footprint.
"""

import logging
from typing import List
import cohere

from app.config import get_settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Handles text embedding generation using Cohere API.
    Uses embed-english-v3.0 model with 1024 dimensions.
    """
    
    def __init__(self):
        settings = get_settings()
        self.client = cohere.Client(settings.cohere_api_key)
        self.model = settings.embedding_model
        self.dimension = settings.embedding_dimension
        
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of documents.
        Uses 'search_document' input type for document embeddings.
        
        Args:
            texts: List of document texts to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        try:
            # Batch process for efficiency (Cohere handles batching internally)
            # Max 96 texts per request for Cohere
            all_embeddings = []
            batch_size = 96
            
            total_batches = (len(texts) + batch_size - 1) // batch_size
            logger.info(
                f"[EMBEDDING] Generating embeddings for {len(texts)} documents in {total_batches} batch(es)"
            )
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_num = (i // batch_size) + 1
                
                logger.info(
                    f"[EMBEDDING] Batch {batch_num}/{total_batches}: Processing {len(batch)} texts..."
                )
                
                response = self.client.embed(
                    texts=batch,
                    model=self.model,
                    input_type="search_document",  # For documents being stored
                    truncate="END"  # Truncate long texts from the end
                )
                
                all_embeddings.extend(response.embeddings)
                logger.info(
                    f"[EMBEDDING] ✓ Batch {batch_num}/{total_batches}: Generated {len(response.embeddings)} embeddings"
                )
            
            logger.info(f"[EMBEDDING] ✓ Total: Generated {len(all_embeddings)} document embeddings")
            return all_embeddings
            
        except cohere.CohereError as e:
            logger.error(f"Cohere API error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Embedding generation failed: {str(e)}")
            raise
    
    async def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a search query.
        Uses 'search_query' input type for query embeddings.
        
        Args:
            query: Search query text
            
        Returns:
            Embedding vector
        """
        try:
            logger.info(f"[EMBEDDING] Generating query embedding (length: {len(query)} chars)")
            response = self.client.embed(
                texts=[query],
                model=self.model,
                input_type="search_query",  # For search queries
                truncate="END"
            )
            
            logger.info(f"[EMBEDDING] ✓ Query embedding generated (dimension: {len(response.embeddings[0])})")
            logger.debug(f"[EMBEDDING] Query: {query[:100]}...")
            return response.embeddings[0]
            
        except cohere.CohereError as e:
            logger.error(f"Cohere API error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Query embedding failed: {str(e)}")
            raise
    
    async def health_check(self) -> bool:
        """Check if Cohere API is accessible."""
        try:
            # Simple embed call to verify API connectivity
            response = self.client.embed(
                texts=["health check"],
                model=self.model,
                input_type="search_query",
                truncate="END"
            )
            return len(response.embeddings) > 0
        except Exception as e:
            logger.error(f"Cohere health check failed: {str(e)}")
            return False


# Singleton instance
embedding_service = EmbeddingService()
