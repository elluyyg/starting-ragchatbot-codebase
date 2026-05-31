# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A RAG (Retrieval-Augmented Generation) system for answering questions about course materials. Users query through a web interface, which triggers semantic search via ChromaDB, then Claude AI generates context-aware responses using Anthropic's tool-calling capability.

## Running the Application

```bash
# Install dependencies
uv sync

# Start the server (from project root or backend dir)
./run.sh
# Or: cd backend && uv run uvicorn app:app --reload --port 8000

# Access
# Web interface: http://localhost:8000
# API docs: http://localhost:8000/docs
```

## Architecture

```
User Query → FastAPI (/api/query) → RAGSystem.query()
                                    ↓
                           AIGenerator.generate_response()
                                    ↓
                           ToolManager executes CourseSearchTool
                                    ↓
                           VectorStore.search() → ChromaDB
```

**Core flow**: The AI uses Anthropic's tool-calling to invoke `search_course_content`, which queries ChromaDB via `VectorStore.search()`. Results are returned to Claude for response synthesis.

**Key files**:
- `backend/app.py` — FastAPI app, CORS setup, API endpoints
- `backend/rag_system.py` — Main orchestrator; coordinates document processing, search, and AI generation
- `backend/vector_store.py` — ChromaDB wrapper; two collections: `course_catalog` (metadata) and `course_content` (chunks)
- `backend/document_processor.py` — Parses course files into `Course`/`CourseChunk` models; chunks text with sentence-splitting and configurable overlap
- `backend/ai_generator.py` — Claude API client with tool execution support
- `backend/search_tools.py` — ToolManager and CourseSearchTool implementing Anthropic tool definition schema
- `backend/session_manager.py` — In-memory conversation history per session
- `backend/models.py` — Pydantic models: Course, Lesson, CourseChunk
- `backend/config.py` — Config dataclass; env vars via python-dotenv

## Data Model

Course documents follow this format:
```
Course Title: [title]
Course Link: [url]
Course Instructor: [name]

Lesson N: [title]
Lesson Link: [url]
[lesson content...]

Lesson N+1: [title]
...
```

Documents are chunked by sentences with configurable `CHUNK_SIZE` (800 chars) and `CHUNK_OVERLAP` (100 chars). Each chunk is embedded and stored with course/lesson metadata.

## Configuration

Set `ANTHROPIC_API_KEY` in `.env`. Other settings in `backend/config.py`:
- `ANTHROPIC_MODEL` — Claude model (default: claude-sonnet-4-20250514)
- `EMBEDDING_MODEL` — Sentence transformer (default: all-MiniLM-L6-v2)
- `CHROMA_PATH` — ChromaDB storage location (default: ./chroma_db)
- `MAX_RESULTS` — Search results limit (default: 5)
- `MAX_HISTORY` — Conversation message pairs to retain (default: 2)

## API Endpoints

- `POST /api/query` — Process a query; returns `{answer, sources, session_id}`
- `GET /api/courses` — Returns `{total_courses, course_titles}`

On startup, the app auto-loads documents from the `docs/` directory.