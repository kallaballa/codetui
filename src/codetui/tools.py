import asyncio
import json
import os
from pathlib import Path
from typing import Any

from mcp import Tool
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters


DEFAULT_CONFIG_PATHS = [
    Path("mcp.json"),
    Path.home() / ".config" / "codetui" / "mcp.json",
]


class ToolDefinition:
    def __init__(self, name: str, description: str, parameters: dict[str, Any]):
        self.name = name
        self.description = description
        self.parameters = parameters


class MCPToolManager:
    def __init__(self, config_path: str | None = None):
        self.config_path = config_path
        self.tools: list[ToolDefinition] = []
        self._sessions: list[ClientSession] = []
        self._contexts: list[Any] = []
        self._config: dict[str, Any] = {}
        self._load_config()

    def _load_config(self) -> None:
        if self.config_path:
            paths = [Path(self.config_path)]
        else:
            paths = DEFAULT_CONFIG_PATHS
        for path in paths:
            if path.exists():
                with open(path, "r", encoding="utf-8") as fh:
                    self._config = json.load(fh)
                return
        self._config = {"mcpServers": {}}

    def _resolve_env(self, value: Any) -> Any:
        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            return os.environ.get(value[2:-1], "")
        return value

    def list_tools(self) -> list[ToolDefinition]:
        return list(self.tools)

    async def initialize(self) -> None:
        self.tools = []
        self._sessions.clear()
        self._contexts = []
        servers = self._config.get("mcpServers", {})
        enabled_servers: list[tuple[str, dict[str, Any], StdioServerParameters]] = []
        for name, cfg in servers.items():
            command = cfg.get("command")
            if not command:
                continue
            args = cfg.get("args", [])
            env = {k: self._resolve_env(v) for k, v in cfg.get("env", {}).items()}
            params = StdioServerParameters(command=command, args=args, env=env)
            enabled_servers.append((name, cfg, params))
            self._contexts.append(stdio_client(params))

        stream_pairs = await asyncio.gather(
            *[ctx.__aenter__() for ctx in self._contexts], return_exceptions=True
        )

        for stream_pair, (name, cfg, _) in zip(stream_pairs, enabled_servers):
            if isinstance(stream_pair, BaseException):
                continue
            read, write = stream_pair
            session = ClientSession(read, write)
            await session.__aenter__()
            await session.initialize()
            list_resp = await session.list_tools()
            for tool in list_resp.tools:
                self.tools.append(
                    ToolDefinition(
                        name=tool.name,
                        description=tool.description or "",
                        parameters=tool.input_schema or {"type": "object", "properties": {}},
                    )
                )
            self._sessions.append(session)

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        for session in self._sessions:
            try:
                resp = await session.call_tool(name, arguments)
                if getattr(resp, "content", None):
                    parts = []
                    for part in resp.content:
                        if hasattr(part, "text"):
                            parts.append(part.text)
                    return "\n".join(parts) if parts else str(resp.content)
                return str(resp)
            except Exception:
                continue
        return f"Tool '{name}' not found or unavailable."

    async def shutdown(self) -> None:
        for ctx in reversed(self._contexts):
            try:
                await ctx.__aexit__(None, None, None)
            except Exception:
                pass
        for session in reversed(self._sessions):
            try:
                await session.__aexit__(None, None, None)
            except Exception:
                pass
        self._sessions.clear()
        self.tools = []
        self._contexts = []


def to_openai_tools(tools: list[ToolDefinition]) -> list[dict[str, Any]]:
    result = []
    for tool in tools:
        parameters = tool.parameters or {"type": "object", "properties": {}}
        if isinstance(parameters, dict):
            parameters.setdefault("type", "object")
            parameters.setdefault("properties", {})
        result.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": parameters,
                },
            }
        )
    return result
