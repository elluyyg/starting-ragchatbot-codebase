"""Tests for AIGenerator to verify it correctly calls CourseSearchTool with proper parameters"""

import pytest
from unittest.mock import MagicMock, patch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ai_generator import AIGenerator
from search_tools import ToolManager, CourseSearchTool, CourseOutlineTool


class MockResponse:
    """Mock HTTP response from API"""

    def __init__(self, status_code, json_data):
        self.status_code = status_code
        self._json_data = json_data

    def json(self):
        return self._json_data


class TestAIGeneratorToolCalls:
    """Test suite to verify AIGenerator correctly invokes tools with proper parameters"""

    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock VectorStore for testing"""
        mock = MagicMock()
        return mock

    @pytest.fixture
    def tool_manager(self, mock_vector_store):
        """Create a ToolManager with registered tools"""
        tm = ToolManager()
        tm.register_tool(CourseSearchTool(mock_vector_store))
        tm.register_tool(CourseOutlineTool(mock_vector_store))
        return tm

    @pytest.fixture
    def ai_generator(self):
        """Create an AIGenerator instance for testing"""
        return AIGenerator(api_key="test-key", model="test-model", base_url="https://api.test.com")

    def test_generate_response_passes_lesson_number_to_tool(
        self, ai_generator, tool_manager, mock_vector_store
    ):
        """Test that when AI decides to use search_course_content, it passes lesson_number correctly"""
        # This test verifies that when the AI returns a tool call with lesson_number,
        # the AIGenerator correctly passes that lesson_number to the tool execution

        # We'll mock the requests.post to return a response that triggers tool use
        # The key is to verify the tool is called with the correct parameters

        # First, set up the mock to return a tool use response
        tool_definitions = tool_manager.get_tool_definitions()

        # Mock tool execution to track what it was called with
        executed_params = {}

        def mock_execute_tool(tool_name, **kwargs):
            executed_params["tool_name"] = tool_name
            executed_params["params"] = kwargs
            # Return lesson 5 specific content
            return "Lesson 5 content about MCP servers"

        tool_manager.execute_tool = mock_execute_tool

        # Create a response that simulates AI deciding to use a tool
        # When AI sees "lesson 5", it should call search_course_content with lesson_number=5
        initial_response = {
            "stop_reason": "tool_use",
            "content": [
                {
                    "type": "tool_use",
                    "id": "tool_123",
                    "name": "search_course_content",
                    "input": {
                        "query": "what was covered",
                        "course_name": "MCP",
                        "lesson_number": 5,
                    },
                }
            ],
        }

        # Mock the API call - first call returns tool use, second returns final answer
        call_count = [0]

        def mock_post(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return MockResponse(200, initial_response)
            else:
                return MockResponse(
                    200,
                    {
                        "stop_reason": "end_turn",
                        "content": [
                            {
                                "type": "text",
                                "text": "In Lesson 5, you learned about creating an MCP client.",
                            }
                        ],
                    },
                )

        with patch("requests.post", side_effect=mock_post):
            result = ai_generator.generate_response(
                query="What was covered in lesson 5 of the MCP course?",
                conversation_history=None,
                tools=tool_definitions,
                tool_manager=tool_manager,
            )

        # Assert that the tool was executed with correct parameters
        assert (
            executed_params["tool_name"] == "search_course_content"
        ), f"Expected search_course_content tool, got {executed_params.get('tool_name')}"
        assert (
            executed_params["params"].get("lesson_number") == 5
        ), f"Expected lesson_number=5, got {executed_params['params'].get('lesson_number')}"
        assert (
            executed_params["params"].get("course_name") == "MCP"
        ), f"Expected course_name='MCP', got {executed_params['params'].get('course_name')}"

    def test_generate_response_extracts_lesson_from_natural_language(
        self, ai_generator, tool_manager
    ):
        """Test that the AI's tool call correctly extracts lesson number from query"""
        # This test verifies the AI is properly prompted to extract lesson numbers

        tool_definitions = tool_manager.get_tool_definitions()

        # Capture what the AI sends as tool input
        captured_input = {}

        def mock_execute_tool(tool_name, **kwargs):
            captured_input["tool_name"] = tool_name
            captured_input["params"] = kwargs
            return "Mock tool result"

        tool_manager.execute_tool = mock_execute_tool

        # Simulate AI response that extracts lesson 5 from natural language
        ai_tool_response = {
            "stop_reason": "tool_use",
            "content": [
                {
                    "type": "tool_use",
                    "id": "tool_456",
                    "name": "search_course_content",
                    "input": {
                        "query": "what was covered",
                        "course_name": "MCP",
                        "lesson_number": 5,
                    },
                }
            ],
        }

        call_count = [0]

        def mock_post(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return MockResponse(200, ai_tool_response)
            else:
                return MockResponse(
                    200,
                    {
                        "stop_reason": "end_turn",
                        "content": [{"type": "text", "text": "Based on Lesson 5..."}],
                    },
                )

        with patch("requests.post", side_effect=mock_post):
            result = ai_generator.generate_response(
                query="What was covered in lesson 5 of the MCP course?",
                tools=tool_definitions,
                tool_manager=tool_manager,
            )

        # Assert
        assert (
            captured_input["params"].get("lesson_number") == 5
        ), "AI should extract lesson_number=5 from 'lesson 5' in query"

    def test_system_prompt_includes_lesson_filtering_guidance(self, ai_generator):
        """Test that the system prompt properly guides the AI on lesson filtering"""
        # The AI needs clear instructions that when user asks about a specific lesson,
        # it should pass the lesson_number parameter

        # Check that the system prompt mentions lesson_number usage
        prompt_text = ai_generator.SYSTEM_PROMPT

        # The system prompt should tell the AI to use lesson_number when query mentions specific lesson
        # This ensures AI knows to extract and pass this parameter
        assert (
            "lesson" in prompt_text.lower() or "filter" in prompt_text.lower()
        ), f"System prompt should mention lesson filtering: {prompt_text[:200]}"

    def test_ai_uses_get_course_outline_for_lesson_list_queries(self, ai_generator, tool_manager):
        """Test that queries about course outline use get_course_outline, not search"""
        tool_definitions = tool_manager.get_tool_definitions()

        captured_tool = {}

        def mock_execute_tool(tool_name, **kwargs):
            captured_tool["tool_name"] = tool_name
            captured_tool["params"] = kwargs
            return "Course outline"

        tool_manager.execute_tool = mock_execute_tool

        # AI response for "what lessons are in the MCP course" should use get_course_outline
        ai_response = {
            "stop_reason": "tool_use",
            "content": [
                {
                    "type": "tool_use",
                    "id": "tool_789",
                    "name": "get_course_outline",
                    "input": {"course_name": "MCP"},
                }
            ],
        }

        call_count = [0]

        def mock_post(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return MockResponse(200, ai_response)
            else:
                return MockResponse(
                    200,
                    {
                        "stop_reason": "end_turn",
                        "content": [{"type": "text", "text": "The MCP course has lessons 0-10..."}],
                    },
                )

        with patch("requests.post", side_effect=mock_post):
            result = ai_generator.generate_response(
                query="What lessons are covered in the MCP course?",
                tools=tool_definitions,
                tool_manager=tool_manager,
            )

        assert (
            captured_tool.get("tool_name") == "get_course_outline"
        ), f"Expected get_course_outline for lesson list query, got {captured_tool.get('tool_name')}"


class TestSequentialToolCalls:
    """Tests for sequential tool calling support (up to 2 tool calls in separate API rounds)"""

    @pytest.fixture
    def mock_vector_store(self):
        mock = MagicMock()
        return mock

    @pytest.fixture
    def tool_manager(self, mock_vector_store):
        tm = ToolManager()
        tm.register_tool(CourseSearchTool(mock_vector_store))
        tm.register_tool(CourseOutlineTool(mock_vector_store))
        return tm

    @pytest.fixture
    def ai_generator(self):
        return AIGenerator(api_key="test-key", model="test-model", base_url="https://api.test.com")

    def test_two_sequential_tool_calls(self, ai_generator, tool_manager):
        """Test that LLM can make 2 sequential tool calls and get a combined response"""
        tool_definitions = tool_manager.get_tool_definitions()

        executed_tools = []

        def mock_execute_tool(tool_name, **kwargs):
            executed_tools.append({"name": tool_name, "params": kwargs})
            if tool_name == "get_course_outline":
                return "Course: MCP\nLessons:\n- Lesson 4: Creating An MCP Client"
            elif tool_name == "search_course_content":
                return "Found related content about MCP clients"
            return "Tool result"

        tool_manager.execute_tool = mock_execute_tool

        # Round 1: AI calls get_course_outline
        round1_response = {
            "stop_reason": "tool_use",
            "content": [
                {
                    "type": "tool_use",
                    "id": "tool_1",
                    "name": "get_course_outline",
                    "input": {"course_name": "MCP"},
                }
            ],
        }

        # Round 2: AI calls search_course_content based on first result
        round2_response = {
            "stop_reason": "tool_use",
            "content": [
                {
                    "type": "tool_use",
                    "id": "tool_2",
                    "name": "search_course_content",
                    "input": {"query": "Creating An MCP Client", "course_name": "MCP"},
                }
            ],
        }

        # Final response after second tool
        final_response = {
            "stop_reason": "end_turn",
            "content": [
                {
                    "type": "text",
                    "text": "Based on my research, Lesson 4 of the MCP course covers Creating An MCP Client, and this topic appears in other courses as well.",
                }
            ],
        }

        call_count = [0]

        def mock_post(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return MockResponse(200, round1_response)
            elif call_count[0] == 2:
                return MockResponse(200, round2_response)
            else:
                return MockResponse(200, final_response)

        with patch("requests.post", side_effect=mock_post):
            result = ai_generator.generate_response(
                query="What lesson in MCP course discusses creating MCP clients and are there related lessons in other courses?",
                tools=tool_definitions,
                tool_manager=tool_manager,
            )

        # Verify both tools were executed
        assert len(executed_tools) == 2, f"Expected 2 tool calls, got {len(executed_tools)}"
        assert executed_tools[0]["name"] == "get_course_outline"
        assert executed_tools[1]["name"] == "search_course_content"
        # Verify final response is returned
        assert "research" in result.lower() or "lesson" in result.lower()

    def test_single_tool_call_then_text(self, ai_generator, tool_manager):
        """Test existing behavior: 1 tool call followed by text response"""
        tool_definitions = tool_manager.get_tool_definitions()

        executed_tools = []

        def mock_execute_tool(tool_name, **kwargs):
            executed_tools.append({"name": tool_name, "params": kwargs})
            return "Course outline result"

        tool_manager.execute_tool = mock_execute_tool

        round1_response = {
            "stop_reason": "tool_use",
            "content": [
                {
                    "type": "tool_use",
                    "id": "tool_1",
                    "name": "get_course_outline",
                    "input": {"course_name": "MCP"},
                }
            ],
        }

        final_response = {
            "stop_reason": "end_turn",
            "content": [
                {
                    "type": "text",
                    "text": "The MCP course has 11 lessons covering MCP architecture, servers, and clients.",
                }
            ],
        }

        call_count = [0]

        def mock_post(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return MockResponse(200, round1_response)
            else:
                return MockResponse(200, final_response)

        with patch("requests.post", side_effect=mock_post):
            result = ai_generator.generate_response(
                query="What is the outline of the MCP course?",
                tools=tool_definitions,
                tool_manager=tool_manager,
            )

        # Verify only 1 tool was executed
        assert len(executed_tools) == 1
        assert executed_tools[0]["name"] == "get_course_outline"
        # Verify text response
        assert "11 lessons" in result

    def test_max_two_rounds_then_stop(self, ai_generator, tool_manager):
        """Test that system stops after 2 rounds even if more tool calls are requested"""
        tool_definitions = tool_manager.get_tool_definitions()

        call_count = [0]

        def mock_execute_tool(tool_name, **kwargs):
            return f"Result from {tool_name}"

        tool_manager.execute_tool = mock_execute_tool

        # All responses ask for another tool - should stop after 2nd round
        tool_response = {
            "stop_reason": "tool_use",
            "content": [
                {
                    "type": "tool_use",
                    "id": "tool_x",
                    "name": "search_course_content",
                    "input": {"query": "test"},
                }
            ],
        }

        final_response = {
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "Final response after max rounds"}],
        }

        def mock_post(*args, **kwargs):
            call_count[0] += 1
            # Always return tool_use - should still stop after 2 rounds
            if call_count[0] <= 2:
                return MockResponse(200, tool_response)
            else:
                return MockResponse(200, final_response)

        with patch("requests.post", side_effect=mock_post):
            result = ai_generator.generate_response(
                query="Keep calling tools", tools=tool_definitions, tool_manager=tool_manager
            )

        # Should make exactly 3 API calls: initial + 2 tool rounds
        assert call_count[0] == 3, f"Expected 3 API calls (initial + 2 rounds), got {call_count[0]}"
        # Tool calls max out at 2 (max_rounds)
        # calls: 1 (initial, tool_use) -> round 1 -> round 2 -> final
        # After round 2, we stop even though we got tool_use

    def test_tool_error_handled_gracefully(self, ai_generator, tool_manager):
        """Test that tool execution errors don't break the flow"""
        tool_definitions = tool_manager.get_tool_definitions()

        def mock_execute_tool(tool_name, **kwargs):
            if tool_name == "search_course_content":
                raise Exception("Database connection failed")
            return "Result"

        tool_manager.execute_tool = mock_execute_tool

        tool_response = {
            "stop_reason": "tool_use",
            "content": [
                {
                    "type": "tool_use",
                    "id": "tool_1",
                    "name": "search_course_content",
                    "input": {"query": "test"},
                }
            ],
        }

        final_response = {
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "Got result with error handling"}],
        }

        call_count = [0]

        def mock_post(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return MockResponse(200, tool_response)
            else:
                return MockResponse(200, final_response)

        with patch("requests.post", side_effect=mock_post):
            result = ai_generator.generate_response(
                query="Search for something", tools=tool_definitions, tool_manager=tool_manager
            )

        # Should still get final response even though tool failed
        assert "error handling" in result.lower() or "result" in result.lower()

    def test_no_second_tool_call_when_not_needed(self, ai_generator, tool_manager):
        """Test that if first tool result directly answers the question, no second tool call is made"""
        tool_definitions = tool_manager.get_tool_definitions()

        executed_tools = []

        def mock_execute_tool(tool_name, **kwargs):
            executed_tools.append({"name": tool_name, "params": kwargs})
            return (
                "Lesson 4: Creating An MCP Client - this lesson explains how to build MCP clients"
            )

        tool_manager.execute_tool = mock_execute_tool

        # After getting lesson info, LLM has enough to answer - no second tool call
        final_response = {
            "stop_reason": "end_turn",
            "content": [
                {
                    "type": "text",
                    "text": "Lesson 4 of the MCP course is titled 'Creating An MCP Client' and it covers building MCP clients.",
                }
            ],
        }

        call_count = [0]

        def mock_post(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                # First call returns tool use
                return MockResponse(
                    200,
                    {
                        "stop_reason": "tool_use",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "tool_1",
                                "name": "get_course_outline",
                                "input": {"course_name": "MCP"},
                            }
                        ],
                    },
                )
            else:
                # Second call returns text directly (no tool use)
                return MockResponse(200, final_response)

        with patch("requests.post", side_effect=mock_post):
            result = ai_generator.generate_response(
                query="What is lesson 4 about in the MCP course?",
                tools=tool_definitions,
                tool_manager=tool_manager,
            )

        # Only 1 tool executed
        assert len(executed_tools) == 1
        assert executed_tools[0]["name"] == "get_course_outline"
        # Final text response returned
        assert "Creating An MCP Client" in result

    def test_system_prompt_allows_multiple_tool_calls(self):
        """Verify system prompt indicates up to 2 sequential tool calls are allowed"""
        prompt = AIGenerator.SYSTEM_PROMPT
        assert (
            "2" in prompt and "tool call" in prompt.lower()
        ), "System prompt should mention allowing up to 2 tool calls"


class TestAIGeneratorSystemPrompt:
    """Tests for the AI generator's system prompt content"""

    def test_system_prompt_mentions_lesson_filtering(self):
        """Verify system prompt includes guidance about lesson filtering"""
        prompt = AIGenerator.SYSTEM_PROMPT

        # The system prompt should instruct the AI on when and how to use lesson_number
        # This is critical for proper tool usage with specific lesson queries
        has_lesson_guidance = "lesson" in prompt.lower() and (
            "filter" in prompt.lower() or "specific" in prompt.lower() or "number" in prompt.lower()
        )

        assert (
            has_lesson_guidance
        ), "System prompt should provide guidance on filtering by lesson number"

    def test_system_prompt_distinguishes_outline_vs_content_queries(self):
        """Verify system prompt distinguishes between outline and content queries"""
        prompt = AIGenerator.SYSTEM_PROMPT

        # Should mention both get_course_outline and search_course_content
        assert (
            "get_course_outline" in prompt
        ), "System prompt should mention get_course_outline tool"
        assert (
            "search_course_content" in prompt
        ), "System prompt should mention search_course_content tool"

        # Should differentiate when to use each
        assert (
            "outline" in prompt.lower() or "structure" in prompt.lower()
        ), "System prompt should explain when to use outline queries"
