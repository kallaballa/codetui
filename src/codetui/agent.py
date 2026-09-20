import asyncio
import json
import os
from pathlib import Path
from typing import Any, AsyncIterator, Callable

from openai import APIConnectionError, AsyncOpenAI

from .history import load_history, save_history
from .tools import MCPToolManager, to_openai_tools

MAX_TOOL_ROUNDS = 100
TOOL_LOOP_MESSAGE = "(stopped: exceeded the maximum number of tool rounds without a final response)"
MAX_CONNECTION_RETRIES = 5
CONNECTION_RETRY_BASE_DELAY = 1.0


class Agent:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str = "gpt-3.5-turbo",
        tool_manager: MCPToolManager | None = None,
        history_path: Path | str | None = None,
        listener: Callable[[dict[str, Any]], None] | None = None,
        max_connection_retries: int = MAX_CONNECTION_RETRIES,
        system_prompt: str | None = None,
    ):
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.max_connection_retries = max_connection_retries
        self.tool_manager = tool_manager
        self.listener = listener
        self.history_path = Path(history_path) if history_path else None
        self.history: list[dict] = (
            load_history(self.history_path) if self.history_path else []
        )
        self.system_prompt = system_prompt

    def set_listener(self, listener: Callable[[dict[str, Any]], None]) -> None:
        self.listener = listener

    def _emit(self, event: dict[str, Any]) -> None:
        if self.listener:
            self.listener(event)

    def _build_tools(self) -> list[dict[str, Any]]:
        if not self.tool_manager:
            return []
        return to_openai_tools(self.tool_manager.list_tools())

    def _system_message(self) -> dict[str, Any]:
        if self.system_prompt is not None:
            return {
                "role": "system",
                "content": self.system_prompt,
            }
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
                "When you need to use a tool, call it using the provided function calling interface. \n\n"
                f"Current working directory: {os.getcwd()}"
            ),
        }

    def _messages(self) -> list[dict[str, Any]]:
        return [self._system_message()] + self.history

    def save(self) -> None:
        if self.history_path:
            save_history(self.history_path, self.history)

    def clear_history(self) -> None:
        self.history = []
        if self.history_path and self.history_path.is_file():
            try:
                self.history_path.unlink()
            except OSError:
                pass

    async def _run_tool(self, name: str, arguments: str, tool_call_id: str) -> None:
        self._emit({"type": "tool_start", "name": name, "arguments": arguments})
        try:
            args = json.loads(arguments)
        except (TypeError, ValueError):
            args = None
        if not isinstance(args, dict):
            args = {"raw": arguments}
        if self.tool_manager is None:
            result: Any = f"Tool '{name}' not found or unavailable."
        else:
            result = await self.tool_manager.call_tool(name, args)
        self._emit({"type": "tool_end", "name": name, "result": str(result)})
        self.history.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": str(result),
            }
        )

    async def _create_chat_completion(
        self,
        *,
        stream: bool,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> Any:
        attempts = 0
        while True:
            try:
                return await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=tools,
                    stream=stream,
                )
            except APIConnectionError:
                attempts += 1
                if attempts >= self.max_connection_retries:
                    raise
                self._emit({"type": "connection_retry", "attempt": attempts})
                await asyncio.sleep(CONNECTION_RETRY_BASE_DELAY * 2 ** (attempts - 1))

    async def send_message(self, content: str) -> str:
        self.history.append({"role": "user", "content": content})
        tools = self._build_tools()
        for _ in range(MAX_TOOL_ROUNDS):
            response = await self._create_chat_completion(
                stream=False,
                messages=self._messages(),
                tools=tools,
            )
            message = response.choices[0].message
            tool_calls = getattr(message, "tool_calls", None)
            if not (isinstance(tool_calls, list) and tool_calls):
                final = message.content or ""
                self.history.append({"role": "assistant", "content": final})
                self.save()
                return final
            self.history.append(message.dict(exclude_none=True))
            for tool_call in tool_calls:
                await self._run_tool(
                    tool_call.function.name,
                    tool_call.function.arguments,
                    tool_call.id,
                )
        self.history.append({"role": "assistant", "content": TOOL_LOOP_MESSAGE})
        self.save()
        return TOOL_LOOP_MESSAGE

    async def stream_message(self, content: str) -> AsyncIterator[str]:
        self.history.append({"role": "user", "content": content})
        tools = self._build_tools()
        for _ in range(MAX_TOOL_ROUNDS):
            response_stream = await self._create_chat_completion(
                stream=True,
                messages=self._messages(),
                tools=tools,
            )
            full_content = ""
            tool_calls: list[dict[str, Any]] = []
            current: dict[str, Any] | None = None
            async for chunk in response_stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    full_content += delta.content
                    yield delta.content
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        if current is None or tc.index != current["index"]:
                            current = {
                                "index": tc.index,
                                "id": tc.id,
                                "function": {"name": "", "arguments": ""},
                            }
                            tool_calls.append(current)
                        if tc.function:
                            if tc.function.name:
                                current["function"]["name"] = tc.function.name
                            if tc.function.arguments:
                                current["function"]["arguments"] += tc.function.arguments
            if not tool_calls:
                self.history.append({"role": "assistant", "content": full_content})
                self.save()
                return
            assistant_tool_calls = [
                {
                    "id": tc.get("id") or tc["function"].get("name", ""),
                    "type": "function",
                    "function": tc["function"],
                }
                for tc in tool_calls
            ]
            self.history.append(
                {"role": "assistant", "tool_calls": assistant_tool_calls}
            )
            for tc in tool_calls:
                await self._run_tool(
                    tc["function"]["name"],
                    tc["function"]["arguments"],
                    tc.get("id") or tc["function"].get("name", ""),
                )
        self.history.append({"role": "assistant", "content": TOOL_LOOP_MESSAGE})
        self.save()
        yield TOOL_LOOP_MESSAGE
