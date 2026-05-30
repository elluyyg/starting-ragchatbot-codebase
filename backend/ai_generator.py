import requests
from typing import List, Optional, Dict, Any

class AIGenerator:
    """Handles interactions with DeepSeek API for generating responses"""

    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to a comprehensive search tool for course information.

Search Tool Usage:
- Use the search tool **only** for questions about specific course content or detailed educational materials
- **One search per query maximum**
- Synthesize search results into accurate, fact-based responses
- If search yields no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course-specific questions**: Search first, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results"


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
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

    def _handle_tool_execution(self, initial_response: Dict, base_params: Dict, tool_manager):
        """Handle execution of tool calls and get follow-up response."""
        messages = base_params["messages"].copy()

        content = initial_response.get("content", [])
        thinking_blocks = [b for b in content if b.get("type") == "thinking"]
        tool_blocks = [b for b in content if b.get("type") == "tool_use"]

        if self._is_dsml_echo(content):
            dsml_result = self._extract_and_execute_dsml(content, tool_manager)
            # Pass back thinking blocks if present, then tool result
            assistant_content = thinking_blocks + tool_blocks if thinking_blocks else initial_response["content"]
            messages.append({"role": "assistant", "content": assistant_content})
            messages.append({
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": tool_blocks[0]["id"] if tool_blocks else "dsml_echo", "content": dsml_result}]
            })
        else:
            messages.append({"role": "assistant", "content": initial_response["content"]})

            tool_results = []
            for block in tool_blocks:
                tool_result = tool_manager.execute_tool(
                    block["name"],
                    **block["input"]
                )
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block["id"],
                    "content": tool_result
                })

            if tool_results:
                messages.append({"role": "user", "content": tool_results})

        final_params = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": messages,
            "system": base_params["system"]
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        final_response = requests.post(self.api_url, headers=headers, json=final_params, timeout=60)

        if final_response.status_code != 200:
            raise Exception(f"API error {final_response.status_code}: {final_response.text}")

        resp_data = final_response.json()
        content = resp_data.get("content", [])

        # Check if we got DSML text blocks (DeepSeek embeds tool calls in text instead of proper format)
        if self._is_dsml_echo(content):
            # Extract and execute the DSML tool call, return the search result directly
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