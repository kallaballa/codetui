import asyncio
import os
from openai import OpenAI
from typing import Any, AsyncIterator

from .tools import MCPToolManager, ToolDefinition, to_openai_tools


class Agent:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str = "gpt-3.5-turbo",
        tool_manager: MCPToolManager | None = None,
    ):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.history: list[dict] = []
        self.tool_manager = tool_manager

    def _build_tools(self) -> list[dict[str, Any]]:
        if not self.tool_manager:
            return []
        return to_openai_tools(self.tool_manager.list_tools())

    def _system_message(self) -> dict[str, Any]:
        tools = self._build_tools()
        if not tools:
            return {
                "role": "system",
                "content": (
                    "You are a helpful assistant. No tools are currently available.\n"
                    f"Current working directory: {os.getcwd()}"
                ),
            }
        tool_descriptions = "\n".join(
            f"- {t['function']['name']}: {t['function'].get('description', '')}"
            for t in tools
        )
        return {
            "role": "system",
            "content": (
                "You are a helpful assistant with access to the following tools:\n"
                f"{tool_descriptions}\n"
                "When you need to use a tool, call it using the provided function calling interface. "
                "Only call a tool if it is relevant to the user's request.\n"
                f"Current working directory: {os.getcwd()}"
            ),
        }

    async def send_message(self, content: str) -> str:
        self.history.append({"role": "user", "content": content})
        tools = self._build_tools()
        messages = [self._system_message()] + self.history
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
        )
        message = response.choices[0].message
        tool_calls = getattr(message, "tool_calls", None)
        while isinstance(tool_calls, list) and tool_calls:
            self.history.append(message.dict(exclude_none=True))
            for tool_call in tool_calls:
                name = tool_call.function.name
                arguments = tool_call.function.arguments
                try:
                    import json as _json
                    args = _json.loads(arguments)
                except Exception:
                    args = {"raw": arguments}
                result = await self.tool_manager.call_tool(name, args)
                self.history.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result),
                })
            follow = self.client.chat.completions.create(
                model=self.model,
                messages=[self._system_message()] + self.history,
                tools=tools,
            )
            message = follow.choices[0].message
            tool_calls = getattr(message, "tool_calls", None)
        final = message.content or ""
        self.history.append({"role": "assistant", "content": final})
        return final

    async def stream_message(self, content: str) -> AsyncIterator[str]:
        self.history.append({"role": "user", "content": content})
        tools = self._build_tools()
        while True:
            messages = [self._system_message()] + self.history
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                stream=True,
            )
            full_content = ""
            tool_calls_accum: list[dict[str, Any]] = []
            current_tool_call = None
            async for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    token = delta.content
                    full_content += token
                    yield token
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        if current_tool_call is None or tc.index != current_tool_call.get("index"):
                            current_tool_call = {
                                "index": tc.index,
                                "id": tc.id,
                                "function": {"name": "", "arguments": ""},
                            }
                            tool_calls_accum.append(current_tool_call)
                        if tc.function:
                            if tc.function.name:
                                current_tool_call["function"]["name"] = tc.function.name
                            if tc.function.arguments:
                                current_tool_call["function"]["arguments"] += tc.function.arguments
            if not tool_calls_accum:
                self.history.append({"role": "assistant", "content": full_content})
                return
            self.history.append({
                "role": "assistant",
                "tool_calls": tool_calls_accum,
            })
            for tc in tool_calls_accum:
                name = tc["function"]["name"]
                arguments = tc["function"]["arguments"]
                try:
                    import json as _json
                    args = _json.loads(arguments)
                except Exception:
                    args = {"raw": arguments}
                result = await self.tool_manager.call_tool(name, args)
                self.history.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id", name),
                    "content": str(result),
                })


