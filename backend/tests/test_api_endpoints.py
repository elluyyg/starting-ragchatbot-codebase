"""API endpoint tests for the FastAPI application.

These tests define the API endpoints inline to avoid issues with
static file mounting in the test environment.
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel
from typing import List, Optional, Union, Dict


class QueryRequest(BaseModel):
    """Request model for course queries"""
    query: str
    session_id: Optional[str] = None


class SourceItem(BaseModel):
    """A source with optional link"""
    title: str
    link: Optional[str] = None


class QueryResponse(BaseModel):
    """Response model for course queries"""
    answer: str
    sources: List[Union[str, Dict]]
    session_id: str


class CourseStats(BaseModel):
    """Response model for course statistics"""
    total_courses: int
    course_titles: List[str]


@pytest.fixture
def mock_rag_system():
    """Create a mock RAG system for API testing"""
    mock = MagicMock()
    mock.query = MagicMock(return_value=(
        "Mock response about course content",
        [{"title": "Test Source", "link": "https://example.com/source"}]
    ))
    mock.get_course_analytics = MagicMock(return_value={
        "total_courses": 2,
        "course_titles": ["MCP Course", "Advanced Python"]
    })
    mock.session_manager = MagicMock()
    # Return unique session IDs on each call
    session_counter = [0]
    def create_session():
        session_counter[0] += 1
        return f"test-session-{session_counter[0]}"
    mock.session_manager.create_session = MagicMock(side_effect=create_session)
    return mock


@pytest.fixture
def test_app(mock_rag_system):
    """Create a test FastAPI app with mocked RAG system"""
    app = FastAPI()

    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        """Process a query and return response with sources"""
        try:
            session_id = request.session_id
            if not session_id:
                session_id = mock_rag_system.session_manager.create_session()

            answer, sources = mock_rag_system.query(request.query, session_id)

            return QueryResponse(
                answer=answer,
                sources=sources,
                session_id=session_id
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        """Get course analytics and statistics"""
        try:
            analytics = mock_rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return app


@pytest.fixture
def client(test_app):
    """Create a test client for the test app"""
    return TestClient(test_app)


class TestQueryEndpoint:
    """Tests for POST /api/query endpoint"""

    def test_query_returns_response_with_sources(self, client, mock_rag_system):
        """Test that query endpoint returns proper response structure"""
        response = client.post(
            "/api/query",
            json={"query": "What is MCP?"}
        )

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        assert data["answer"] == "Mock response about course content"

    def test_query_with_custom_session_id(self, client, mock_rag_system):
        """Test that custom session_id is used when provided"""
        response = client.post(
            "/api/query",
            json={"query": "What is MCP?", "session_id": "my-custom-session"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "my-custom-session"

    def test_query_without_session_id_generates_one(self, client, mock_rag_system):
        """Test that a session_id is generated when not provided"""
        response = client.post(
            "/api/query",
            json={"query": "What is MCP?"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] is not None
        assert data["session_id"] != ""

    def test_query_empty_request_returns_422(self, client):
        """Test that empty request body returns validation error"""
        response = client.post("/api/query", json={})
        assert response.status_code == 422

    def test_query_missing_query_field_returns_422(self, client):
        """Test that missing query field returns validation error"""
        response = client.post("/api/query", json={"session_id": "test"})
        assert response.status_code == 422

    def test_query_rag_system_error_returns_500(self, client, mock_rag_system):
        """Test that RAG system errors return 500 status"""
        mock_rag_system.query.side_effect = Exception("RAG system failure")

        response = client.post(
            "/api/query",
            json={"query": "Test query"}
        )

        assert response.status_code == 500


class TestCoursesEndpoint:
    """Tests for GET /api/courses endpoint"""

    def test_courses_returns_analytics(self, client, mock_rag_system):
        """Test that courses endpoint returns course statistics"""
        response = client.get("/api/courses")

        assert response.status_code == 200
        data = response.json()
        assert "total_courses" in data
        assert "course_titles" in data
        assert data["total_courses"] == 2
        assert "MCP Course" in data["course_titles"]

    def test_courses_empty_when_no_courses(self, client, mock_rag_system):
        """Test courses endpoint with no courses"""
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": []
        }

        response = client.get("/api/courses")

        assert response.status_code == 200
        data = response.json()
        assert data["total_courses"] == 0
        assert data["course_titles"] == []

    def test_courses_error_returns_500(self, client, mock_rag_system):
        """Test that analytics errors return 500 status"""
        mock_rag_system.get_course_analytics.side_effect = Exception("Analytics failure")

        response = client.get("/api/courses")

        assert response.status_code == 500


class TestQueryValidation:
    """Tests for query request validation"""

    def test_query_with_none_values(self, client):
        """Test query with null values in request"""
        response = client.post(
            "/api/query",
            json={"query": None}
        )
        # Pydantic validation should reject None for query field
        assert response.status_code == 422

    def test_query_with_empty_string(self, client):
        """Test query with empty string is accepted (query could be valid)"""
        response = client.post(
            "/api/query",
            json={"query": ""}
        )
        # Empty string is technically valid, RAG system may handle it
        assert response.status_code in [200, 500]

    def test_query_with_very_long_query(self, client, mock_rag_system):
        """Test query with very long input is handled"""
        long_query = "A" * 10000
        response = client.post(
            "/api/query",
            json={"query": long_query}
        )
        assert response.status_code in [200, 500]  # Should handle gracefully


class TestResponseFormat:
    """Tests for response format consistency"""

    def test_query_response_has_all_required_fields(self, client, mock_rag_system):
        """Verify all required fields are present in query response"""
        response = client.post(
            "/api/query",
            json={"query": "Test query"}
        )

        data = response.json()
        assert isinstance(data["answer"], str)
        assert isinstance(data["sources"], list)
        assert isinstance(data["session_id"], str)

    def test_courses_response_has_all_required_fields(self, client, mock_rag_system):
        """Verify all required fields are present in courses response"""
        response = client.get("/api/courses")

        data = response.json()
        assert isinstance(data["total_courses"], int)
        assert isinstance(data["course_titles"], list)
        assert all(isinstance(title, str) for title in data["course_titles"])

    def test_sources_can_contain_dicts_or_strings(self, client, mock_rag_system):
        """Test that sources can be either string or dict format"""
        mock_rag_system.query.return_value = (
            "Answer",
            [
                "Source as string",
                {"title": "Source as dict", "link": "https://example.com"}
            ]
        )

        response = client.post(
            "/api/query",
            json={"query": "Test"}
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["sources"]) == 2


class TestSessionManagement:
    """Tests for session management in query endpoint"""

    def test_different_requests_get_different_sessions(self, client, mock_rag_system):
        """Test that separate requests without session_id get unique sessions"""
        response1 = client.post("/api/query", json={"query": "First query"})
        response2 = client.post("/api/query", json={"query": "Second query"})

        assert response1.json()["session_id"] != response2.json()["session_id"]

    def test_same_session_id_reuses_session(self, client, mock_rag_system):
        """Test that same session_id reuses the same session"""
        session_id = "reuse-this-session"

        response1 = client.post(
            "/api/query",
            json={"query": "First", "session_id": session_id}
        )
        response2 = client.post(
            "/api/query",
            json={"query": "Second", "session_id": session_id}
        )

        assert response1.json()["session_id"] == response2.json()["session_id"]