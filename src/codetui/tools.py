import json
import logging
import os
import re
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from mcp import Client
from mcp.client.stdio import StdioServerParameters
from mcp.types import TextContent

logger = logging.getLogger(__name__)

ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

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
        self._clients: list[Client] = []
        self._exit_stack: AsyncExitStack | None = None
        self._config: dict[str, Any] = {}
        self._tool_clients: dict[str, Client] = {}
        self._load_config()

    def _load_config(self) -> None:
        paths = [Path(self.config_path)] if self.config_path else DEFAULT_CONFIG_PATHS
        for path in paths:
            if not path.exists():
                continue
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    self._config = json.load(fh)
                return
            except (OSError, ValueError) as exc:
                logger.warning("Failed to load MCP config '%s': %s", path, exc)
        self._config = {"mcpServers": {}}

    def _resolve_env(self, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        return ENV_PATTERN.sub(
            lambda m: os.environ.get(m.group(1), m.group(0)), value
        )

    def list_tools(self) -> list[ToolDefinition]:
        return list(self.tools)

    async def initialize(self) -> None:
        self.tools = []
        self._clients = []
        self._tool_clients = {}
        self._exit_stack = None
        seen_names: set[str] = set()
        servers = self._config.get("mcpServers", {})
        async with AsyncExitStack() as exit_stack:
            for name, cfg in servers.items():
                command = cfg.get("command")
                if not command:
                    continue
                command = self._resolve_env(command)
                args = [self._resolve_env(arg) for arg in cfg.get("args", [])]
                env = {k: self._resolve_env(v) for k, v in cfg.get("env", {}).items()}
                params = StdioServerParameters(command=command, args=args, env=env)
                try:
                    client = Client(params)
                    await exit_stack.enter_async_context(client)
                    result = await client.list_tools()
                except Exception as exc:
                    logger.warning("Failed to initialize MCP server '%s': %s", name, exc)
                    continue
                for tool in result.tools:
                    if tool.name in seen_names:
                        logger.warning(
                            "Ignoring duplicate tool '%s' from server '%s'", tool.name, name
                        )
                        continue
                    seen_names.add(tool.name)
                    self.tools.append(
                        ToolDefinition(
                            name=tool.name,
                            description=tool.description or "",
                            parameters=tool.input_schema or {"type": "object", "properties": {}},
                        )
                    )
                    self._tool_clients[tool.name] = client
                self._clients.append(client)
            self._exit_stack = exit_stack.pop_all()

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        client = self._tool_clients.get(name)
        if client is None:
            return f"Tool '{name}' not found or unavailable."
        try:
            result = await client.call_tool(name, arguments)
        except Exception as e:
            return f"Tool '{name}' failed: {e}"
        if result.is_error:
            return f"Tool '{name}' error: {self._extract_text(result.content)}"
        text = self._extract_text(result.content)
        if text:
            return text
        return str(result.content)

    @staticmethod
    def _extract_text(content: list[Any]) -> str:
        parts = []
        for block in content:
            if isinstance(block, TextContent):
                parts.append(block.text)
        return "\n".join(parts)

    async def shutdown(self) -> None:
        if self._exit_stack is not None:
            try:
                await self._exit_stack.__aexit__(None, None, None)
            except Exception:
                pass
            finally:
                self._exit_stack = None
        self._clients = []
        self.tools = []


def to_openai_tools(tools: list[ToolDefinition]) -> list[dict[str, Any]]:
    result = []
    for tool in tools:
        parameters = tool.parameters or {"type": "object", "properties": {}}
        if isinstance(parameters, dict):
            parameters = dict(parameters)
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
