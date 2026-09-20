import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from codetui.tools import MCPToolManager, ToolDefinition, to_openai_tools


def test_to_openai_tools():
    tools = [
        ToolDefinition(name="foo", description="desc", parameters={"type": "object", "properties": {"x": {"type": "string"}}}),
        ToolDefinition(name="bar", description="", parameters={"type": "object"}),
    ]
    result = to_openai_tools(tools)
    assert result == [
        {"type": "function", "function": {"name": "foo", "description": "desc", "parameters": {"type": "object", "properties": {"x": {"type": "string"}}}}},
        {"type": "function", "function": {"name": "bar", "description": "", "parameters": {"type": "object", "properties": {}}}},
    ]


def test_tool_manager_loads_default_config(tmp_path):
    cfg = {
        "mcpServers": {
            "test": {"command": "python", "args": ["-c", "print('hi')"]}
        }
    }
    config = tmp_path / "mcp.json"
    config.write_text(json.dumps(cfg))
    mgr = MCPToolManager(config_path=str(config))
    assert mgr._config == cfg
    assert mgr.list_tools() == []


@pytest.mark.asyncio
async def test_tool_manager_initialize_connect_server():
    mock_tool = MagicMock()
    mock_tool.name = "search"
    mock_tool.description = "Search"
    mock_tool.input_schema = {"type": "object", "properties": {"q": {"type": "string"}}}
    mock_list_result = MagicMock()
    mock_list_result.tools = [mock_tool]

    mock_client = MagicMock()
    mock_client.list_tools = AsyncMock(return_value=mock_list_result)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("codetui.tools.Client", return_value=mock_client):
        mgr = MCPToolManager()
        mgr._config = {
            "mcpServers": {
                "search": {"command": "python", "args": ["server.py"]}
            }
        }
        await mgr.initialize()
        assert len(mgr.list_tools()) == 1
        assert mgr.list_tools()[0].name == "search"


@pytest.mark.asyncio
async def test_tool_manager_call_tool_fallback():
    mgr = MCPToolManager()
    mgr._clients = []
    result = await mgr.call_tool("missing", {})
    assert result == "Tool 'missing' not found or unavailable."
