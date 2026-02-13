"""
Document ingestion endpoint for processing and storing course materials.
"""

import logging
import time
from typing import Optional, Dict

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status, BackgroundTasks

from app.config import get_settings
from app.models import (
    DocumentType, 
    IngestResponse, 
    ErrorResponse, 
    S3IngestRequest,
    ProcessingStatus,
    ProcessingStatusResponse
)
from app.services import (
    document_processor,
    chunking_service,
    embedding_service,
    vector_store_service
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ingest", tags=["Ingestion"])

# In-memory storage for processing status (use Redis in production)
processing_status_store: Dict[str, dict] = {}


@router.post(
    "",
    response_model=IngestResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        413: {"model": ErrorResponse, "description": "File too large"},
        415: {"model": ErrorResponse, "description": "Unsupported file type"},
        500: {"model": ErrorResponse, "description": "Processing error"}
    }
)
async def ingest_document(
    file: UploadFile = File(..., description="Document file to process"),
    room_id: str = Form(..., description="Study room ID"),
    file_id: str = Form(..., description="Unique file identifier"),
    document_type: str = Form(..., description="Document type (pdf, docx, pptx, txt)")
):
    """
    Ingest a document for RAG.
    
    Process:
    1. Validate file size and type
    2. Extract text from document
    3. Chunk text into semantic segments
    4. Generate embeddings via Cohere API
    5. Store vectors in Qdrant with metadata
    
    Returns:
        IngestResponse with processing details
    """
    start_time = time.time()
    settings = get_settings()
    
    # Validate document type
    try:
        doc_type = DocumentType(document_type.lower())
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported document type: {document_type}. Supported: pdf, docx, pptx, txt"
        )
    
    # Read file content
    try:
        content = await file.read()
    except Exception as e:
        logger.error(f"Failed to read file: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to read uploaded file"
        )
    
    # Validate file size
    if len(content) > settings.max_file_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum size of {settings.max_file_size_mb}MB"
        )
    
    if len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty file uploaded"
        )
    
    logger.info(
        f"Processing document: file_id={file_id}, room_id={room_id}, "
        f"type={doc_type.value}, size={len(content)} bytes"
    )
    
    try:
        # Step 1: Extract text
        text, error = await document_processor.extract_text(content, doc_type)
        
        if error:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=error
            )
        
        if not text.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No text content could be extracted from the document"
            )
        
        logger.info(f"Extracted {len(text)} characters from document")
        
        # Step 2: Chunk text
        chunks = chunking_service.chunk_text(text)
        
        if not chunks:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Document produced no valid text chunks"
            )
        
        logger.info(f"Created {len(chunks)} chunks")
        
        # Step 3: Generate embeddings
        chunk_texts = [chunk.content for chunk in chunks]
        embeddings = await embedding_service.embed_documents(chunk_texts)
        
        logger.info(f"Generated {len(embeddings)} embeddings")
        
        # Step 4: Delete existing vectors for this file (idempotent update)
        await vector_store_service.delete_file(file_id)
        
        # Step 5: Store vectors
        chunks_stored = await vector_store_service.store_chunks(
            chunks=chunks,
            embeddings=embeddings,
            room_id=room_id,
            file_id=file_id,
            document_type=doc_type.value
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        logger.info(
            f"Ingestion complete: file_id={file_id}, "
            f"chunks={chunks_stored}, time={processing_time:.0f}ms"
        )
        
        return IngestResponse(
            success=True,
            file_id=file_id,
            room_id=room_id,
            chunks_created=chunks_stored,
            processing_time_ms=processing_time,
            message=f"Successfully processed document with {chunks_stored} chunks"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ingestion failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document processing failed: {str(e)}"
        )


async def process_s3_document(
    s3_url: str,
    file_id: str,
    room_id: str,
    document_type: DocumentType
):
    """
    Background task to process document from S3.
    
    This runs asynchronously to avoid blocking the API response.
    """
    try:
        # Update status to processing
        processing_status_store[file_id] = {
            "status": ProcessingStatus.PROCESSING,
            "chunks_created": None,
            "error": None,
            "processed_at": None
        }
        
        settings = get_settings()
        start_time = time.time()
        
        logger.info(f"Starting S3 document processing: file_id={file_id}, s3_url={s3_url}")
        
        # Step 1: Download from S3
        content, download_error = await document_processor.download_from_s3(s3_url)
        if download_error:
            raise Exception(f"S3 download failed: {download_error}")
        
        # Validate file size
        if len(content) > settings.max_file_size_bytes:
            raise Exception(f"File exceeds maximum size of {settings.max_file_size_mb}MB")
        
        if len(content) == 0:
            raise Exception("Downloaded file is empty")
        
        # Step 2: Extract text
        text, extract_error = await document_processor.extract_text(content, document_type)
        if extract_error:
            raise Exception(f"Text extraction failed: {extract_error}")
        
        if not text.strip():
            raise Exception("No text content could be extracted from the document")
        
        logger.info(f"Extracted {len(text)} characters from S3 document")
        
        # Step 3: Chunk text
        chunks = chunking_service.chunk_text(text)
        if not chunks:
            raise Exception("Document produced no valid text chunks")
        
        logger.info(f"Created {len(chunks)} chunks")
        
        # Step 4: Generate embeddings
        chunk_texts = [chunk.content for chunk in chunks]
        embeddings = await embedding_service.embed_documents(chunk_texts)
        
        logger.info(f"Generated {len(embeddings)} embeddings")
        
        # Step 5: Delete existing vectors for this file (idempotent update)
        await vector_store_service.delete_file(file_id)
        
        # Step 6: Store vectors
        chunks_stored = await vector_store_service.store_chunks(
            chunks=chunks,
            embeddings=embeddings,
            room_id=room_id,
            file_id=file_id,
            document_type=document_type.value
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        # Update status to completed
        from datetime import datetime
        processing_status_store[file_id] = {
            "status": ProcessingStatus.COMPLETED,
            "chunks_created": chunks_stored,
            "error": None,
            "processed_at": datetime.utcnow()
        }
        
        logger.info(
            f"S3 ingestion complete: file_id={file_id}, "
            f"chunks={chunks_stored}, time={processing_time:.0f}ms"
        )
        
    except Exception as e:
        logger.error(f"S3 ingestion failed for file_id={file_id}: {str(e)}", exc_info=True)
        from datetime import datetime
        processing_status_store[file_id] = {
            "status": ProcessingStatus.FAILED,
            "chunks_created": None,
            "error": str(e),
            "processed_at": datetime.utcnow()
        }


@router.post(
    "/ingest-from-s3",
    response_model=IngestResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        202: {"description": "Processing started"},
        424: {"model": ErrorResponse, "description": "S3 download failed"},
        422: {"model": ErrorResponse, "description": "File processing failed"}
    }
)
async def ingest_from_s3(
    request: S3IngestRequest,
    background_tasks: BackgroundTasks
):
    """
    Ingest a document from S3 URL for RAG.
    
    This endpoint accepts an S3 URL and processes the file asynchronously.
    The frontend should:
    1. Upload file directly to S3 using presigned URL
    2. Call this endpoint with the S3 URL
    3. Poll /ingest-status/{file_id} to check processing status
    
    Process:
    1. Download file from S3
    2. Extract text from document
    3. Chunk text into semantic segments
    4. Generate embeddings via Cohere API
    5. Store vectors in Qdrant with metadata
    
    Returns:
        202 Accepted with processing status
    """
    try:
        # Validate document type
        doc_type = request.document_type
        
        # Initialize processing status
        processing_status_store[request.file_id] = {
            "status": ProcessingStatus.PENDING,
            "chunks_created": None,
            "error": None,
            "processed_at": None
        }
        
        # Add background task to process the document
        background_tasks.add_task(
            process_s3_document,
            request.s3_url,
            request.file_id,
            request.room_id,
            doc_type
        )
        
        logger.info(
            f"S3 ingestion queued: file_id={request.file_id}, "
            f"room_id={request.room_id}, s3_url={request.s3_url}"
        )
        
        return IngestResponse(
            success=True,
            file_id=request.file_id,
            room_id=request.room_id,
            chunks_created=0,  # Will be updated when processing completes
            processing_time_ms=0,
            message="Document processing started. Check status at /ingest-status/{file_id}"
        )
        
    except Exception as e:
        logger.error(f"S3 ingestion request failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to queue document processing: {str(e)}"
        )


@router.get(
    "/ingest-status/{file_id}",
    response_model=ProcessingStatusResponse,
    responses={
        404: {"model": ErrorResponse, "description": "File not found"}
    }
)
async def get_processing_status(file_id: str):
    """
    Get the processing status of a document.
    
    Use this endpoint to check if embedding generation is complete.
    
    Args:
        file_id: The file ID to check
        
    Returns:
        ProcessingStatusResponse with current status
    """
    if file_id not in processing_status_store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No processing status found for file_id: {file_id}"
        )
    
    status_data = processing_status_store[file_id]
    
    return ProcessingStatusResponse(
        file_id=file_id,
        status=status_data["status"],
        chunks_created=status_data["chunks_created"],
        error=status_data["error"],
        processed_at=status_data["processed_at"]
    )


@router.delete("/{file_id}")
async def delete_document(file_id: str, room_id: Optional[str] = None):
    """
    Delete a document's vectors from the store.
    
    Args:
        file_id: File ID to delete
        room_id: Optional room ID for verification
        
    Returns:
        Deletion confirmation
    """
    try:
        deleted_count = await vector_store_service.delete_file(file_id)
        
        return {
            "success": True,
            "file_id": file_id,
            "vectors_deleted": deleted_count,
            "message": f"Deleted {deleted_count} vectors for file {file_id}"
        }
        
    except Exception as e:
        logger.error(f"Delete failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete document: {str(e)}"
        )


@router.delete("/room/{room_id}")
async def delete_room_documents(room_id: str):
    """
    Delete all documents for a study room.
    
    Args:
        room_id: Room ID
        
    Returns:
        Deletion confirmation
    """
    try:
        deleted_count = await vector_store_service.delete_room(room_id)
        
        return {
            "success": True,
            "room_id": room_id,
            "vectors_deleted": deleted_count,
            "message": f"Deleted {deleted_count} vectors for room {room_id}"
        }
        
    except Exception as e:
        logger.error(f"Delete room failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete room documents: {str(e)}"
        )
