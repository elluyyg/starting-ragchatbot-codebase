"""Tests for CourseSearchTool.execute() to verify lesson_number filtering works correctly"""

from unittest.mock import MagicMock
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from search_tools import CourseSearchTool, SearchResults


class TestCourseSearchToolExecute:
    """Test suite for CourseSearchTool.execute() method"""

    def create_mock_vector_store(self, search_results):
        """Create a mock VectorStore that returns predefined search results"""
        mock_store = MagicMock()
        mock_store.search.return_value = search_results
        return mock_store

    def test_execute_with_lesson_number_filters_correctly(self):
        """Test that execute() correctly filters by lesson_number parameter"""
        # Arrange: Create mock search results that return ONLY lesson 5 content
        mock_results = SearchResults(
            documents=["Lesson 5 content: This is about MCP servers and how to build them."],
            metadata=[
                {
                    "course_title": "MCP: Build Rich-Context AI Apps with Anthropic",
                    "lesson_number": 5,
                    "lesson_link": "https://example.com/lesson5",
                }
            ],
            distances=[0.1],
        )

        mock_store = self.create_mock_vector_store(mock_results)
        tool = CourseSearchTool(mock_store)

        # Act: Execute search for lesson 5
        result = tool.execute(query="what was covered", course_name="MCP", lesson_number=5)

        # Assert: Verify VectorStore.search was called with correct lesson_number
        mock_store.search.assert_called_once()
        call_kwargs = mock_store.search.call_args.kwargs

        assert (
            call_kwargs.get("lesson_number") == 5
        ), f"Expected lesson_number=5, but search was called with lesson_number={call_kwargs.get('lesson_number')}"

    def test_execute_with_lesson_number_returns_lesson_5_content(self):
        """Test that execute returns content specifically from lesson 5, not other lessons"""
        # Arrange: Create mock results with ONLY lesson 5 content
        lesson_5_content = "Lesson 5 covers creating an MCP client that connects to servers."
        mock_results = SearchResults(
            documents=[lesson_5_content],
            metadata=[
                {
                    "course_title": "MCP: Build Rich-Context AI Apps with Anthropic",
                    "lesson_number": 5,
                    "lesson_link": "https://learn.deeplearning.ai/lesson5",
                }
            ],
            distances=[0.1],
        )

        mock_store = self.create_mock_vector_store(mock_results)
        tool = CourseSearchTool(mock_store)

        # Act
        result = tool.execute(query="creating an MCP client", course_name="MCP", lesson_number=5)

        # Assert: Result should contain lesson 5 specific content
        assert (
            "Lesson 5" in result or "lesson 5" in result.lower()
        ), f"Expected result to mention Lesson 5, but got: {result[:200]}"

    def test_execute_without_lesson_number_returns_all_lessons(self):
        """Test that execute without lesson_number returns content from all lessons"""
        # Arrange: Create mock results from multiple lessons
        mock_results = SearchResults(
            documents=[
                "Lesson 1 content about why MCP.",
                "Lesson 2 content about MCP architecture.",
                "Lesson 5 content about MCP client.",
            ],
            metadata=[
                {
                    "course_title": "MCP Course",
                    "lesson_number": 1,
                    "lesson_link": "https://example.com/1",
                },
                {
                    "course_title": "MCP Course",
                    "lesson_number": 2,
                    "lesson_link": "https://example.com/2",
                },
                {
                    "course_title": "MCP Course",
                    "lesson_number": 5,
                    "lesson_link": "https://example.com/5",
                },
            ],
            distances=[0.1, 0.2, 0.15],
        )

        mock_store = self.create_mock_vector_store(mock_results)
        tool = CourseSearchTool(mock_store)

        # Act
        result = tool.execute(query="MCP", course_name="MCP")

        # Assert: Verify search was called with lesson_number=None
        call_kwargs = mock_store.search.call_args.kwargs
        assert (
            call_kwargs.get("lesson_number") is None
        ), f"Expected lesson_number=None when not specified, but got {call_kwargs.get('lesson_number')}"

    def test_execute_empty_results_returns_empty_message(self):
        """Test that execute handles empty results gracefully"""
        # Arrange
        mock_results = SearchResults.empty("No results found")
        mock_store = self.create_mock_vector_store(mock_results)
        tool = CourseSearchTool(mock_store)

        # Act
        result = tool.execute(query="nonexistent topic", lesson_number=99)

        # Assert
        assert "No relevant content found" in result or "No results found" in result

    def test_execute_tracks_sources_for_ui(self):
        """Test that execute properly tracks sources for UI display"""
        # Arrange
        mock_results = SearchResults(
            documents=["Content about lesson 5"],
            metadata=[
                {
                    "course_title": "MCP Course",
                    "lesson_number": 5,
                    "lesson_link": "https://example.com/lesson5",
                }
            ],
            distances=[0.1],
        )

        mock_store = self.create_mock_vector_store(mock_results)
        tool = CourseSearchTool(mock_store)

        # Act
        result = tool.execute(query="test", course_name="MCP", lesson_number=5)

        # Assert: Verify last_sources is populated
        assert len(tool.last_sources) > 0, "last_sources should not be empty after search"
        # Verify the source has lesson 5 in it (either via lesson_number field or in title)
        first_source = tool.last_sources[0]
        assert first_source.get("lesson_number") == 5 or "5" in str(
            first_source
        ), f"Expected source to contain lesson 5, got: {tool.last_sources}"


class TestCourseSearchToolSorting:
    """Test suite for multi-course result sorting (course first, then lesson number)"""

    def create_mock_vector_store(self, search_results):
        """Create a mock VectorStore that returns predefined search results"""
        mock_store = MagicMock()
        mock_store.search.return_value = search_results
        return mock_store

    def test_results_sorted_by_course_then_lesson_number(self):
        """Test that results are sorted by course first, then lesson number within course"""
        # Arrange: Results from multiple courses in random order
        mock_results = SearchResults(
            documents=["Doc PC L3", "Doc B L6", "Doc B L1", "Doc PC L1", "Doc B L2"],
            metadata=[
                {"course_title": "Prompt Compression", "lesson_number": 3, "lesson_link": "pc-l3"},
                {"course_title": "Building Towards", "lesson_number": 6, "lesson_link": "b-l6"},
                {"course_title": "Building Towards", "lesson_number": 1, "lesson_link": "b-l1"},
                {"course_title": "Prompt Compression", "lesson_number": 1, "lesson_link": "pc-l1"},
                {"course_title": "Building Towards", "lesson_number": 2, "lesson_link": "b-l2"},
            ],
            distances=[0.1, 0.2, 0.15, 0.1, 0.1],
        )

        mock_store = self.create_mock_vector_store(mock_results)
        tool = CourseSearchTool(mock_store)

        # Act
        tool.execute(query="test", course_name=None)

        # Get the formatted result text
        result = tool._format_results(mock_results)

        # Assert: Should be sorted by course first (alphabetically), then by lesson number
        # Building Towards comes before Prompt Compression alphabetically
        # Within Building Towards: L1, L2, L6
        # Within Prompt Compression: L1, L3
        lines = result.split("\n\n")
        headers = [line.split("\n")[0] for line in lines if line]

        # Check order: Building Towards lessons 1, 2, 6 should come before Prompt Compression lessons 1, 3
        assert (
            "Building Towards" in headers[0]
        ), f"Expected Building Towards first, got: {headers[0]}"
        assert "Lesson 1" in headers[0], f"Expected Lesson 1 first, got: {headers[0]}"
        assert "Lesson 2" in headers[1], f"Expected Lesson 2 second, got: {headers[1]}"
        assert "Lesson 6" in headers[2], f"Expected Lesson 6 third, got: {headers[2]}"
        assert (
            "Prompt Compression" in headers[3]
        ), f"Expected Prompt Compression fourth, got: {headers[3]}"
        assert "Lesson 1" in headers[3], f"Expected Lesson 1 fourth, got: {headers[3]}"
        assert "Lesson 3" in headers[4], f"Expected Lesson 3 fifth, got: {headers[4]}"

    def test_sources_sorted_by_course_then_lesson_number(self):
        """Test that sources list is also sorted by course first, then lesson number"""
        # Arrange
        mock_results = SearchResults(
            documents=["Doc PC L3", "Doc B L6", "Doc B L1"],
            metadata=[
                {"course_title": "Prompt Compression", "lesson_number": 3, "lesson_link": "pc-l3"},
                {"course_title": "Building Towards", "lesson_number": 6, "lesson_link": "b-l6"},
                {"course_title": "Building Towards", "lesson_number": 1, "lesson_link": "b-l1"},
            ],
            distances=[0.1, 0.2, 0.15],
        )

        mock_store = self.create_mock_vector_store(mock_results)
        tool = CourseSearchTool(mock_store)

        # Act
        tool.execute(query="test", course_name=None)
        tool._format_results(mock_results)

        # Assert: Sources should be sorted by course then lesson
        sources = tool.last_sources
        assert len(sources) == 3
        # Building Towards comes before Prompt Compression
        assert sources[0]["course_title"] == "Building Towards"
        assert sources[0].get("lesson_number") == 1 or "1" in sources[0].get("title", "")
        assert sources[1]["course_title"] == "Building Towards"
        assert sources[1].get("lesson_number") == 6 or "6" in sources[1].get("title", "")
        assert sources[2]["course_title"] == "Prompt Compression"

    def test_single_course_results_sorted_by_lesson_number(self):
        """Test that results from a single course are sorted by lesson number"""
        # Arrange
        mock_results = SearchResults(
            documents=["Doc L5", "Doc L2", "Doc L1", "Doc L3"],
            metadata=[
                {"course_title": "MCP Course", "lesson_number": 5, "lesson_link": "l5"},
                {"course_title": "MCP Course", "lesson_number": 2, "lesson_link": "l2"},
                {"course_title": "MCP Course", "lesson_number": 1, "lesson_link": "l1"},
                {"course_title": "MCP Course", "lesson_number": 3, "lesson_link": "l3"},
            ],
            distances=[0.1, 0.2, 0.15, 0.1],
        )

        mock_store = self.create_mock_vector_store(mock_results)
        tool = CourseSearchTool(mock_store)

        # Act
        tool.execute(query="test", course_name=None)
        tool._format_results(mock_results)

        # Assert
        sources = tool.last_sources
        assert sources[0].get("lesson_number") == 1 or "Lesson 1" in sources[0]["title"]
        assert sources[1].get("lesson_number") == 2 or "Lesson 2" in sources[1]["title"]
        assert sources[2].get("lesson_number") == 3 or "Lesson 3" in sources[2]["title"]
        assert sources[3].get("lesson_number") == 5 or "Lesson 5" in sources[3]["title"]

    def test_sorting_uses_lesson_number_field_not_title(self):
        """Verify sorting uses the lesson_number field directly, not extracted from title"""
        # This is a key test - the fix was to use item.get('lesson_number') directly
        # because the title doesn't contain "Lesson N" pattern when sources are built

        # Arrange
        mock_results = SearchResults(
            documents=["Doc 1", "Doc 2"],
            metadata=[
                {
                    "course_title": "Course A",
                    "lesson_number": 10,
                    "lesson_link": "link10",
                },  # lesson 10
                {
                    "course_title": "Course A",
                    "lesson_number": 2,
                    "lesson_link": "link2",
                },  # lesson 2
            ],
            distances=[0.1, 0.2],
        )

        mock_store = self.create_mock_vector_store(mock_results)
        tool = CourseSearchTool(mock_store)

        # Act
        tool.execute(query="test", course_name=None)
        tool._format_results(mock_results)

        # Assert: Should sort by lesson_number field (2 before 10)
        sources = tool.last_sources
        assert (
            "Lesson 2" in sources[0]["title"]
        ), f"Expected Lesson 2 first, got: {sources[0]['title']}"
        assert (
            "Lesson 10" in sources[1]["title"]
        ), f"Expected Lesson 10 second, got: {sources[1]['title']}"


class TestCourseSearchToolDefinition:
    """Test that tool definition is correctly formed for AI consumption"""

    def test_tool_definition_has_lesson_number_parameter(self):
        """Verify the tool definition includes lesson_number as a parameter"""
        mock_store = MagicMock()
        mock_store.search.return_value = SearchResults.empty("error")
        tool = CourseSearchTool(mock_store)

        definition = tool.get_tool_definition()

        # Assert lesson_number is in the input schema
        properties = definition["input_schema"]["properties"]
        assert (
            "lesson_number" in properties
        ), f"lesson_number not in tool definition properties: {properties.keys()}"

    def test_tool_definition_lesson_number_is_integer(self):
        """Verify lesson_number is defined as integer type"""
        mock_store = MagicMock()
        mock_store.search.return_value = SearchResults.empty("error")
        tool = CourseSearchTool(mock_store)

        definition = tool.get_tool_definition()
        lesson_number_schema = definition["input_schema"]["properties"]["lesson_number"]

        assert (
            lesson_number_schema.get("type") == "integer"
        ), f"lesson_number should be integer, got: {lesson_number_schema.get('type')}"
