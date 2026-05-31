import requests
from typing import List, Optional, Dict, Any

class AIGenerator:
    """Handles interactions with DeepSeek API for generating responses"""

    SYSTEM_PROMPT = """You are an AI assistant specialized in course materials and educational content with access to comprehensive search tools for course information.

Available Tools:
1. **search_course_content** - For searching specific content within courses
2. **get_course_outline** - For getting course structure with all lessons (use when asked about course outline, lesson list, or course structure)

Tool Usage:
- Use **get_course_outline** for questions about course outline, all lessons, or course structure
- Use **search_course_content** for specific content or detailed material questions
- **Up to 2 sequential tool calls per query** — when a tool result reveals new information, you may call another tool to follow up
- Synthesize search results into accurate, fact-based responses
- If search yields no results, state this clearly without offering alternatives

**CRITICAL - Parameter Extraction:**
When the user mentions a specific lesson number (e.g., "lesson 5", "lesson 3"), you MUST:
1. Extract the lesson number as an integer (e.g., lesson 5 -> lesson_number=5)
2. Also extract the course name if mentioned (e.g., "MCP course" -> course_name="MCP")
3. Pass BOTH parameters to the search_course_content tool

Examples:
- "What was covered in lesson 5 of the MCP course?" -> search_course_content(query="what was covered", course_name="MCP", lesson_number=5)
- "Tell me about lesson 3" -> search_course_content(query="lesson 3 content", lesson_number=3)
- "What's in the Anthropic course?" -> get_course_outline(course_name="Anthropic")

Response Protocol:
- **Course outline queries**: Use get_course_outline, then provide structured response with course title, link, and all lesson numbers/titles
- **Content-specific questions**: Use search_course_content, then answer based on results
- **No meta-commentary**: Provide direct answers only — no reasoning process or search explanations
"""

    def __init__(self, api_key: str, model: str, base_url: str = ""):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip('/') if base_url else "https://api.deepseek.com/anthropic"
        self.api_url = f"{self.base_url}/v1/messages"
        self.max_tokens = 800

    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with optional tool usage and conversation context.
        """

        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        api_payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": query}],
            "system": system_content
        }

        if tools:
            api_payload["tools"] = tools
            api_payload["tool_choice"] = {"type": "auto"}

        response = requests.post(self.api_url, headers=headers, json=api_payload, timeout=60)

        if response.status_code != 200:
            raise Exception(f"API error {response.status_code}: {response.text}")

        resp_data = response.json()

        # Check for tool use
        if resp_data.get("stop_reason") == "tool_use" and tool_manager:
            return self._handle_tool_execution(resp_data, api_payload, tool_manager)

        # Return text response
        for block in resp_data.get("content", []):
            if block.get("type") == "text":
                return block["text"]
        return ""

    def _handle_tool_execution(self, initial_response: Dict, base_params: Dict, tool_manager, max_rounds: int = 2):
        """Handle execution of tool calls and get follow-up response. Supports up to max_rounds sequential tool calls."""
        messages = base_params["messages"].copy()
        rounds = 0
        current_response = initial_response

        while rounds < max_rounds:
            content = current_response.get("content", [])
            thinking_blocks = [b for b in content if b.get("type") == "thinking"]
            tool_blocks = [b for b in content if b.get("type") == "tool_use"]

            if self._is_dsml_echo(content):
                dsml_result = self._extract_and_execute_dsml(content, tool_manager)
                assistant_content = thinking_blocks + tool_blocks if thinking_blocks else current_response["content"]
                messages.append({"role": "assistant", "content": assistant_content})
                messages.append({
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": tool_blocks[0]["id"] if tool_blocks else "dsml_echo", "content": dsml_result}]
                })
            elif tool_blocks:
                messages.append({"role": "assistant", "content": current_response["content"]})

                tool_results = []
                for block in tool_blocks:
                    try:
                        tool_result = tool_manager.execute_tool(block["name"], **block["input"])
                    except Exception as e:
                        tool_result = f"Error executing tool: {str(e)}"
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block["id"],
                        "content": tool_result
                    })

                if tool_results:
                    messages.append({"role": "user", "content": tool_results})
            else:
                # No tool blocks - this is the final response
                break

            # Make next API call WITH tools still enabled (for potential second round)
            next_params = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "messages": messages,
                "system": base_params["system"],
                "tools": base_params.get("tools"),
                "tool_choice": {"type": "auto"}
            }

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            next_response = requests.post(self.api_url, headers=headers, json=next_params, timeout=60)

            if next_response.status_code != 200:
                raise Exception(f"API error {next_response.status_code}: {next_response.text}")

            current_response = next_response.json()

            # Check if LLM wants to use more tools or return text
            if current_response.get("stop_reason") != "tool_use":
                break

            rounds += 1

        # Final response processing
        content = current_response.get("content", [])

        if self._is_dsml_echo(content):
            dsml_result = self._extract_and_execute_dsml(content, tool_manager)
            return dsml_result

        for block in content:
            if block.get("type") == "text":
                text = block["text"]
                if not text.startswith("<｜｜DSML｜｜"):
                    return text
        return ""

    def _is_dsml_echo(self, content: List) -> bool:
        """Check if content is a DeepSeek DSML echo instead of proper tool use"""
        if not content:
            return False
        for block in content:
            if block.get("type") == "text":
                text = block.get("text", "")
                if text and ("<｜｜DSML｜｜tool_calls>" in text or "tool_calls>" in text):
                    return True
        return False

    def _extract_and_execute_dsml(self, content: List, tool_manager) -> str:
        """Extract tool call from DSML echo and execute it"""
        for block in content:
            if block.get("type") == "text":
                text = block.get("text", "")

                # Extract tool name - look for invoke name="..."
                import re
                name_match = re.search(r'invoke name="([^"]+)"', text)
                if not name_match:
                    return "Error: Could not extract tool name from DSML echo"
                tool_name = name_match.group(1)

                # Extract parameters - find all name="X" value pairs
                params = {}
                # Pattern: name="paramname" followed by string="value" or just >value<
                param_pattern = r'name="([^"]+)"[^>]*string="([^"]+)"'
                for match in re.finditer(param_pattern, text):
                    params[match.group(1)] = match.group(2)

                # If no string= params found, try extracting from >...< pattern
                if not params:
                    param_pattern2 = r'name="([^"]+)"[^>]*>([^<]+)<'
                    for match in re.finditer(param_pattern2, text):
                        params[match.group(1)] = match.group(2)

                # Execute the tool
                try:
                    result = tool_manager.execute_tool(tool_name, **params)
                    return result
                except Exception as e:
                    return f"Error executing tool: {str(e)}"
        return "Error: No text block found in DSML content"