"""Shared pytest fixtures for RAG system testing"""

import pytest
from unittest.mock import MagicMock, patch, Mock
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from search_tools import SearchResults, CourseSearchTool, CourseOutlineTool
from config import Config
from models import Course, Lesson, CourseChunk


@pytest.fixture
def mock_config():
    """Create a mock configuration for testing"""
    config = MagicMock(spec=Config)
    config.CHUNK_SIZE = 800
    config.CHUNK_OVERLAP = 100
    config.CHROMA_PATH = "./test_chroma"
    config.OLLAMA_EMBEDDING_MODEL = "test-model"
    config.OLLAMA_EMBEDDING_URL = "http://localhost:11434"
    config.MAX_RESULTS = 5
    config.MAX_HISTORY = 2
    config.ANTHROPIC_API_KEY = "test-key"
    config.ANTHROPIC_MODEL = "test-model"
    config.ANTHROPIC_BASE_URL = "https://api.test.com"
    return config


@pytest.fixture
def mock_vector_store():
    """Create a mock VectorStore with controlled responses"""
    mock = MagicMock()
    return mock


@pytest.fixture
def sample_course():
    """Create a sample Course model for testing"""
    return Course(
        title="MCP: Build Rich-Context AI Apps with Anthropic",
        course_link="https://example.com/mcp-course",
        instructor="Test Instructor",
        lessons=[
            Lesson(lesson_number=1, title="Introduction to MCP", lesson_link="https://example.com/lesson1"),
            Lesson(lesson_number=2, title="MCP Architecture", lesson_link="https://example.com/lesson2"),
            Lesson(lesson_number=3, title="Building MCP Servers", lesson_link="https://example.com/lesson3"),
            Lesson(lesson_number=4, title="Creating An MCP Client", lesson_link="https://example.com/lesson4"),
            Lesson(lesson_number=5, title="Advanced MCP Patterns", lesson_link="https://example.com/lesson5"),
        ]
    )


@pytest.fixture
def sample_search_results():
    """Create sample SearchResults for testing"""
    return SearchResults(
        documents=[
            "Lesson 1 content about MCP introduction and basics.",
            "Lesson 2 content about MCP architecture and design patterns.",
            "Lesson 5 content about advanced MCP patterns and best practices."
        ],
        metadata=[
            {"course_title": "MCP: Build Rich-Context AI Apps with Anthropic", "lesson_number": 1, "lesson_link": "https://example.com/lesson1"},
            {"course_title": "MCP: Build Rich-Context AI Apps with Anthropic", "lesson_number": 2, "lesson_link": "https://example.com/lesson2"},
            {"course_title": "MCP: Build Rich-Context AI Apps with Anthropic", "lesson_number": 5, "lesson_link": "https://example.com/lesson5"},
        ],
        distances=[0.1, 0.15, 0.2]
    )


@pytest.fixture
def lesson_5_only_results():
    """Create SearchResults containing only lesson 5 content"""
    return SearchResults(
        documents=[
            "Lesson 5 content: Advanced MCP patterns including error handling, retries, and scaling."
        ],
        metadata=[{
            "course_title": "MCP: Build Rich-Context AI Apps with Anthropic",
            "lesson_number": 5,
            "lesson_link": "https://example.com/lesson5"
        }],
        distances=[0.1]
    )


@pytest.fixture
def mock_search_tool(mock_vector_store, sample_search_results):
    """Create a mock CourseSearchTool with predefined results"""
    mock_tool = MagicMock(spec=CourseSearchTool)
    mock_tool.last_sources = []
    mock_tool.execute = MagicMock(return_value=sample_search_results.formatted_text if hasattr(sample_search_results, 'formatted_text') else str(sample_search_results))
    return mock_tool


@pytest.fixture
def mock_tool_manager(mock_vector_store, sample_search_results):
    """Create a mock ToolManager with registered tools"""
    mock_tm = MagicMock()
    mock_tm.get_tool_definitions = MagicMock(return_value=[
        {"name": "search_course_content", "input_schema": {"type": "object", "properties": {}}},
        {"name": "get_course_outline", "input_schema": {"type": "object", "properties": {}}}
    ])
    mock_tm.get_last_sources = MagicMock(return_value=[])
    mock_tm.reset_sources = MagicMock()
    mock_tm.execute_tool = MagicMock(return_value="Mock tool result")
    return mock_tm


@pytest.fixture
def mock_ai_generator():
    """Create a mock AIGenerator"""
    mock = MagicMock()
    mock.generate_response = MagicMock(return_value="Mock AI response about course content")
    return mock


class MockResponse:
    """Mock HTTP response for API testing"""
    def __init__(self, status_code, json_data):
        self.status_code = status_code
        self._json_data = json_data

    def json(self):
        return self._json_data