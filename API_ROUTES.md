# AI Study Assistant - API Routes Documentation

Complete reference for all API endpoints. Use this to build your Postman collection.

**Base URL:** 
- Local: `http://localhost:8001`
- Production: `https://your-render-url.onrender.com`

---

## 📚 Table of Contents

1. [Health Check](#health-check)
2. [Document Ingestion](#document-ingestion)
3. [Query (RAG Q&A)](#query-rag-qa)
4. [Study Plan Generation](#study-plan-generation)

---

## 🏥 Health Check

### Get Service Health

Check the status of all service dependencies (Qdrant, Cohere, Gemini).

**Endpoint:** `GET /health`

**Headers:** None required

**Response (200 OK):**
```json
{
  "status": "healthy",
  "timestamp": "2026-02-17T10:30:00.000Z",
  "services": [
    {
      "name": "Qdrant Vector DB",
      "status": "healthy",
      "latency_ms": 123.45,
      "details": "Connected to cluster"
    },
    {
      "name": "Cohere Embeddings",
      "status": "healthy",
      "latency_ms": 456.78,
      "details": "API accessible"
    },
    {
      "name": "Gemini LLM",
      "status": "healthy",
      "latency_ms": 789.01,
      "details": "Model: gemini-1.5-pro"
    }
  ]
}
```

**Use Case:** Monitor service availability before operations

---

## 📄 Document Ingestion

### 1. Direct File Upload (Testing/Admin)

Upload a file directly to the service for processing.

**Endpoint:** `POST /ingest`

**Headers:**
```
Content-Type: multipart/form-data
```

**Body (form-data):**
```
file: [Binary File] (Required)
room_id: "550e8400-e29b-41d4-a716-446655440000" (Required)
file_id: "650e8400-e29b-41d4-a716-446655440001" (Required)
document_type: "pdf" (Required - Options: pdf, docx, pptx, txt)
```

**Example Request:**
```bash
curl -X POST http://localhost:8001/ingest \
  -F "file=@machine_learning.pdf" \
  -F "room_id=550e8400-e29b-41d4-a716-446655440000" \
  -F "file_id=650e8400-e29b-41d4-a716-446655440001" \
  -F "document_type=pdf"
```

**Response (200 OK):**
```json
{
  "success": true,
  "file_id": "650e8400-e29b-41d4-a716-446655440001",
  "room_id": "550e8400-e29b-41d4-a716-446655440000",
  "chunks_created": 42,
  "processing_time_ms": 3456.78,
  "message": "Successfully processed document with 42 chunks"
}
```

**Error Responses:**

**400 Bad Request:**
```json
{
  "detail": "Empty file uploaded"
}
```

**413 Request Entity Too Large:**
```json
{
  "detail": "File exceeds maximum size of 10MB"
}
```

**415 Unsupported Media Type:**
```json
{
  "detail": "Unsupported document type: doc. Supported: pdf, docx, pptx, txt"
}
```

**Use Case:** Direct testing, admin uploads, small files

---

### 2. S3 URL Ingestion (Production)

Process a file already uploaded to S3. **Recommended for production.**

**Endpoint:** `POST /ingest/ingest-from-s3`

**Headers:**
```
Content-Type: application/json
```

**Body (JSON):**
```json
{
  "s3_url": "https://your-bucket.s3.amazonaws.com/room-files/room-uuid/file-name.pdf",
  "room_id": "550e8400-e29b-41d4-a716-446655440000",
  "file_id": "650e8400-e29b-41d4-a716-446655440001",
  "document_type": "pdf",
  "metadata": {
    "original_name": "Introduction to ML.pdf",
    "uploader_id": "user-uuid-here"
  }
}
```

**Example Request:**
```bash
curl -X POST http://localhost:8001/ingest/ingest-from-s3 \
  -H "Content-Type: application/json" \
  -d '{
    "s3_url": "https://mahesararslan-merge-bucket.s3.eu-north-1.amazonaws.com/room-files/room-uuid/file.pdf",
    "room_id": "550e8400-e29b-41d4-a716-446655440000",
    "file_id": "650e8400-e29b-41d4-a716-446655440001",
    "document_type": "pdf"
  }'
```

**Response (202 Accepted):**
```json
{
  "success": true,
  "file_id": "650e8400-e29b-41d4-a716-446655440001",
  "room_id": "550e8400-e29b-41d4-a716-446655440000",
  "chunks_created": 0,
  "processing_time_ms": 0,
  "message": "Document processing started. Check status at /ingest-status/{file_id}"
}
```

**Processing Flow:**
1. Returns 202 immediately
2. Downloads file from S3 in background
3. Extracts text → Chunks → Embeds → Stores in Qdrant
4. Poll `/ingest/ingest-status/{file_id}` for completion

**Use Case:** Production file uploads after S3 upload

---

### 3. Check Processing Status

Poll this endpoint to check if embedding generation is complete.

**Endpoint:** `GET /ingest/ingest-status/{file_id}`

**URL Parameters:**
- `file_id` (required): File identifier

**Example Request:**
```bash
curl http://localhost:8001/ingest/ingest-status/650e8400-e29b-41d4-a716-446655440001
```

**Response - Processing (200 OK):**
```json
{
  "file_id": "650e8400-e29b-41d4-a716-446655440001",
  "status": "processing",
  "chunks_created": null,
  "error": null,
  "processed_at": null
}
```

**Response - Completed (200 OK):**
```json
{
  "file_id": "650e8400-e29b-41d4-a716-446655440001",
  "status": "completed",
  "chunks_created": 42,
  "error": null,
  "processed_at": "2026-02-17T10:30:45.123Z"
}
```

**Response - Failed (200 OK):**
```json
{
  "file_id": "650e8400-e29b-41d4-a716-446655440001",
  "status": "failed",
  "chunks_created": null,
  "error": "Text extraction failed: No text content could be extracted from the document",
  "processed_at": "2026-02-17T10:30:45.123Z"
}
```

**Response - Not Found (404):**
```json
{
  "detail": "No processing status found for file_id: 650e8400-e29b-41d4-a716-446655440001"
}
```

**Polling Strategy:** Check every 5 seconds, max 5 minutes

**Use Case:** Monitor asynchronous processing completion

---

### 4. Delete File Vectors

Delete all embeddings for a specific file.

**Endpoint:** `DELETE /ingest/{file_id}`

**URL Parameters:**
- `file_id` (required): File identifier

**Query Parameters:**
- `room_id` (optional): Room ID for verification

**Example Request:**
```bash
curl -X DELETE http://localhost:8001/ingest/650e8400-e29b-41d4-a716-446655440001
```

**Response (200 OK):**
```json
{
  "success": true,
  "file_id": "650e8400-e29b-41d4-a716-446655440001",
  "vectors_deleted": 42,
  "message": "Deleted 42 vectors for file 650e8400-e29b-41d4-a716-446655440001"
}
```

**Use Cases:**
- Failed processing cleanup
- File update (delete old → upload new)
- Manual deletion

---

### 5. Delete All Room Vectors

Delete all embeddings for an entire room.

**Endpoint:** `DELETE /ingest/room/{room_id}`

**URL Parameters:**
- `room_id` (required): Room identifier

**Example Request:**
```bash
curl -X DELETE http://localhost:8001/ingest/room/550e8400-e29b-41d4-a716-446655440000
```

**Response (200 OK):**
```json
{
  "success": true,
  "room_id": "550e8400-e29b-41d4-a716-446655440000",
  "vectors_deleted": 156,
  "message": "Deleted 156 vectors for room 550e8400-e29b-41d4-a716-446655440000"
}
```

**Use Cases:**
- Room deletion
- Bulk cleanup
- Testing reset

---

## 💬 Query (RAG Q&A)

### 1. Standard Query

Execute a RAG query and get a complete answer with sources.

**Endpoint:** `POST /query`

**Headers:**
```
Content-Type: application/json
```

**Body (JSON):**
```json
{
  "query": "What are the key concepts in machine learning?",
  "user_id": "778159b1-21c9-43cb-b495-87fcc8108692",
  "room_ids": ["550e8400-e29b-41d4-a716-446655440000"],
  "context_file_id": "650e8400-e29b-41d4-a716-446655440001",
  "top_k": 5
}
```

**Field Descriptions:**
- `query` (required, string, 1-2000 chars): User's question
- `user_id` (required, string): User identifier
- `room_ids` (required, array): List of room IDs to search (min 1)
- `context_file_id` (optional, string): Focus on specific file
- `top_k` (optional, integer, 1-20): Number of chunks to retrieve (default: 5)

**Example Request:**
```bash
curl -X POST http://localhost:8001/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Explain supervised learning",
    "user_id": "user-uuid",
    "room_ids": ["room-uuid-1", "room-uuid-2"],
    "top_k": 5
  }'
```

**Response (200 OK):**
```json
{
  "answer": "Supervised learning is a type of machine learning where the algorithm learns from labeled training data. The model is trained on input-output pairs and learns to map inputs to the correct outputs. Common examples include classification and regression tasks. [Source 1: Introduction to ML, Chapter 2]",
  "sources": [
    {
      "file_id": "650e8400-e29b-41d4-a716-446655440001",
      "chunk_index": 5,
      "content": "Supervised learning algorithms learn from labeled examples...",
      "relevance_score": 0.89,
      "section_title": "Chapter 2: Supervised Learning"
    },
    {
      "file_id": "650e8400-e29b-41d4-a716-446655440001",
      "chunk_index": 12,
      "content": "Classification and regression are the two main types...",
      "relevance_score": 0.82,
      "section_title": "Types of ML Tasks"
    }
  ],
  "query": "Explain supervised learning",
  "processing_time_ms": 1250.5,
  "chunks_retrieved": 5
}
```

**Error Responses:**

**400 Bad Request:**
```json
{
  "detail": "At least one room_id is required"
}
```

**500 Internal Server Error:**
```json
{
  "detail": "Query processing failed: Connection timeout"
}
```

**Use Case:** Standard Q&A interaction

---

### 2. Streaming Query (SSE)

Execute a RAG query with real-time streaming response.

**Endpoint:** `POST /query/stream`

**Headers:**
```
Content-Type: application/json
Accept: text/event-stream
```

**Body (JSON):** Same as standard query

**Example Request:**
```bash
curl -X POST http://localhost:8001/query/stream \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{
    "query": "What is deep learning?",
    "user_id": "user-uuid",
    "room_ids": ["room-uuid"]
  }'
```

**Response (SSE Stream):**

**Event 1 - Status:**
```
event: status
data: {"message": "Searching course materials..."}
```

**Event 2 - Sources:**
```
event: sources
data: {"sources": [{"file_id": "...", "chunk_index": 5, "relevance_score": 0.89}]}
```

**Event 3-N - Answer Chunks:**
```
event: chunk
data: {"text": "Deep learning is"}

event: chunk
data: {"text": " a subset of machine learning"}
```

**Event N+1 - Complete:**
```
event: complete
data: {"processing_time_ms": 2345.67, "chunks_retrieved": 5}
```

**Event - Error (if failed):**
```
event: error
data: {"error": "Connection timeout"}
```

**Event Types:**
- `status`: Processing updates
- `sources`: Retrieved source chunks
- `chunk`: Answer text fragments
- `complete`: Final metadata
- `error`: Error information

**Use Case:** Real-time chat interface, progressive loading

---

## 📅 Study Plan Generation

### 1. Generate Study Plan

Create a personalized study schedule with calendar integration.

**Endpoint:** `POST /study-plan/generate`

**Headers:**
```
Content-Type: application/json
```

**Body (JSON):**
```json
{
  "user_id": "778159b1-21c9-43cb-b495-87fcc8108692",
  "goal": "Master machine learning fundamentals for final exam",
  "topics": [
    "Supervised Learning",
    "Unsupervised Learning",
    "Neural Networks",
    "Model Evaluation"
  ],
  "start_date": "2026-03-01",
  "end_date": "2026-04-15",
  "difficulty_level": "intermediate",
  "preferences": {
    "preferred_study_hours": [9, 10, 14, 15, 16, 20, 21],
    "session_duration_minutes": 90,
    "break_duration_minutes": 15,
    "days_per_week": 5
  }
}
```

**Field Descriptions:**
- `user_id` (required, string): User identifier
- `goal` (required, string): Study objective
- `topics` (required, array): List of topics to cover (min 1)
- `start_date` (required, string, YYYY-MM-DD): Plan start date
- `end_date` (required, string, YYYY-MM-DD): Plan end date (max 6 months)
- `difficulty_level` (required, enum): "beginner", "intermediate", "advanced"
- `preferences` (optional, object): Study preferences

**Preferences Object:**
- `preferred_study_hours` (array): Hours of day (0-23)
- `session_duration_minutes` (number): Default 60
- `break_duration_minutes` (number): Default 15
- `days_per_week` (number): Default 5

**Example Request:**
```bash
curl -X POST http://localhost:8001/study-plan/generate \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user-uuid",
    "goal": "Learn Python programming",
    "topics": ["Variables", "Functions", "OOP", "Data Structures"],
    "start_date": "2026-03-01",
    "end_date": "2026-03-31",
    "difficulty_level": "beginner"
  }'
```

**Response (200 OK):**
```json
{
  "weekly_schedule": [
    {
      "week_number": 1,
      "start_date": "2026-03-01",
      "end_date": "2026-03-07",
      "sessions": [
        {
          "date": "2026-03-01",
          "start_time": "09:00",
          "end_time": "10:30",
          "topic": "Variables and Data Types",
          "learning_objectives": [
            "Understand primitive data types",
            "Learn variable declaration and assignment"
          ],
          "resources": [
            "Chapter 1: Introduction to Python",
            "Practice exercises 1-10"
          ],
          "notes": "Start with interactive coding examples"
        }
      ],
      "weekly_goals": [
        "Complete basic syntax",
        "Understand control flow"
      ],
      "estimated_hours": 7.5
    }
  ],
  "milestones": [
    {
      "week": 2,
      "milestone": "Complete fundamentals",
      "verification": "Quiz on basic concepts"
    }
  ],
  "calendar_conflicts": [
    {
      "date": "2026-03-05",
      "event_title": "Team Meeting",
      "conflict_type": "overlap",
      "suggestion": "Move session to evening or next day"
    }
  ],
  "adjustment_tips": [
    "Review previous concepts before starting new topics",
    "Take breaks every 90 minutes",
    "Practice coding daily for at least 30 minutes"
  ],
  "total_weeks": 4,
  "total_sessions": 20,
  "total_hours": 30.0
}
```

**Process:**
1. LLM receives request
2. LLM calls `get_user_calendar` function (via function calling)
3. Backend fetches calendar from NestJS API (`{API_SERVER_URL}/calendar`)
4. LLM analyzes availability
5. LLM generates structured study plan
6. Returns JSON with schedule

**Error Responses:**

**400 Bad Request - Invalid Dates:**
```json
{
  "detail": "End date must be after start date"
}
```

**400 Bad Request - Too Long:**
```json
{
  "detail": "Study plan duration cannot exceed 6 months"
}
```

**400 Bad Request - No Topics:**
```json
{
  "detail": "At least one topic is required"
}
```

**Use Case:** Personalized study schedule creation

---

### 2. Preview Calendar (Debug)

Preview user's calendar data without generating a plan.

**Endpoint:** `POST /study-plan/preview`

**Headers:**
```
Content-Type: application/json
```

**Body (JSON):**
```json
{
  "user_id": "778159b1-21c9-43cb-b495-87fcc8108692",
  "start_date": "2026-03-01",
  "end_date": "2026-03-31"
}
```

**Example Request:**
```bash
curl -X POST http://localhost:8001/study-plan/preview \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user-uuid",
    "start_date": "2026-03-01",
    "end_date": "2026-03-31"
  }'
```

**Response (200 OK):**
```json
{
  "user_id": "778159b1-21c9-43cb-b495-87fcc8108692",
  "date_range": {
    "start": "2026-03-01",
    "end": "2026-03-31"
  },
  "calendar_data": {
    "events": [
      {
        "id": "event-uuid",
        "title": "Team Meeting",
        "start": "2026-03-05T14:00:00Z",
        "end": "2026-03-05T15:00:00Z",
        "type": "meeting"
      }
    ],
    "total_events": 15,
    "busy_hours": 45.5
  },
  "available_slots": [
    {
      "date": "2026-03-01",
      "available_hours": [9, 10, 11, 14, 15, 16, 20, 21]
    }
  ],
  "api_status": "success"
}
```

**Response - Calendar API Failed (200 OK):**
```json
{
  "user_id": "user-uuid",
  "date_range": {
    "start": "2026-03-01",
    "end": "2026-03-31"
  },
  "calendar_data": {
    "error": "Calendar API unavailable",
    "events": [],
    "total_events": 0
  },
  "available_slots": [],
  "api_status": "failed"
}
```

**Use Case:** Debug calendar integration, test API availability

---

## 🧪 Testing Workflows

### Complete Ingestion & Query Flow

```bash
# 1. Check service health
GET /health

# 2. Upload file from S3
POST /ingest/ingest-from-s3
{
  "s3_url": "https://...",
  "room_id": "room-uuid",
  "file_id": "file-uuid",
  "document_type": "pdf"
}

# 3. Poll status (every 5s)
GET /ingest/ingest-status/file-uuid

# 4. Once completed, query
POST /query
{
  "query": "What is this document about?",
  "user_id": "user-uuid",
  "room_ids": ["room-uuid"],
  "context_file_id": "file-uuid"
}
```

### Study Plan Generation Flow

```bash
# 1. Preview calendar first (optional)
POST /study-plan/preview
{
  "user_id": "user-uuid",
  "start_date": "2026-03-01",
  "end_date": "2026-03-31"
}

# 2. Generate plan
POST /study-plan/generate
{
  "user_id": "user-uuid",
  "goal": "Learn ML",
  "topics": ["Supervised", "Unsupervised"],
  "start_date": "2026-03-01",
  "end_date": "2026-03-31",
  "difficulty_level": "intermediate"
}
```

### Cleanup Flow

```bash
# Delete single file
DELETE /ingest/file-uuid

# Or delete entire room
DELETE /ingest/room/room-uuid
```

---

## 📋 Postman Environment Variables

Create a Postman environment with these variables:

```json
{
  "base_url": "http://localhost:8001",
  "user_id": "778159b1-21c9-43cb-b495-87fcc8108692",
  "room_id": "550e8400-e29b-41d4-a716-446655440000",
  "file_id": "650e8400-e29b-41d4-a716-446655440001",
  "s3_url": "https://mahesararslan-merge-bucket.s3.eu-north-1.amazonaws.com/room-files/...",
  "start_date": "2026-03-01",
  "end_date": "2026-03-31"
}
```

---

## 🚨 Common Error Codes

| Code | Meaning | Common Causes |
|------|---------|---------------|
| 400 | Bad Request | Invalid input, missing required fields |
| 404 | Not Found | File/status not found |
| 413 | Payload Too Large | File exceeds 10MB limit |
| 415 | Unsupported Media Type | Invalid document type |
| 500 | Internal Server Error | Service failure, timeout |

---

## 💡 Tips for Postman

1. **Import as Collection**: Save all requests in a collection
2. **Use Variables**: Replace UUIDs with `{{file_id}}`, `{{room_id}}`
3. **Tests Tab**: Add auto-parsing of response IDs
4. **Pre-request Scripts**: Generate UUIDs automatically
5. **Environments**: Create Dev, Staging, Production environments

---

## 📖 Additional Resources

- **Interactive Docs**: `http://localhost:8001/docs` (Swagger UI)
- **Alternative Docs**: `http://localhost:8001/redoc` (ReDoc)
- **Health Check**: `http://localhost:8001/health`

---

**Last Updated:** February 17, 2026  
**API Version:** 1.0.0
