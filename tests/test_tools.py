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


def test_resolve_env_expands_set_variable(monkeypatch):
    monkeypatch.setenv("CODETUI_TEST_TOKEN", "secret")
    mgr = MCPToolManager()
    assert mgr._resolve_env("${CODETUI_TEST_TOKEN}") == "secret"


def test_resolve_env_preserves_unset_variable(monkeypatch):
    monkeypatch.delenv("CODETUI_TEST_UNSET", raising=False)
    mgr = MCPToolManager()
    assert mgr._resolve_env("${CODETUI_TEST_UNSET}") == "${CODETUI_TEST_UNSET}"


def test_resolve_env_leaves_plain_values_alone(monkeypatch):
    mgr = MCPToolManager()
    assert mgr._resolve_env("plain") == "plain"
    assert mgr._resolve_env("a${b}c") == "a${b}c"
    assert mgr._resolve_env(42) == 42


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


@pytest.mark.asyncio
async def test_tool_manager_deduplicates_tool_names():
    def make_tool(name):
        tool = MagicMock()
        tool.name = name
        tool.description = "desc"
        tool.input_schema = {"type": "object", "properties": {}}
        return tool

    def make_result(names):
        result = MagicMock()
        result.tools = [make_tool(name) for name in names]
        return result

    def make_client(names):
        client = MagicMock()
        client.list_tools = AsyncMock(return_value=make_result(names))
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)
        return client

    client_a = make_client(["search"])
    client_b = make_client(["search", "other"])

    with patch("codetui.tools.Client", side_effect=iter([client_a, client_b])):
        mgr = MCPToolManager()
        mgr._config = {
            "mcpServers": {"a": {"command": "python", "args": ["a.py"]}, "b": {"command": "python", "args": ["b.py"]}}
        }
        await mgr.initialize()
        assert [t.name for t in mgr.list_tools()] == ["search", "other"]
        assert mgr._tool_clients["search"] is client_a


def test_resolve_env_interpolates_embedded(monkeypatch):
    monkeypatch.setenv("CODETUI_PART", "xyz")
    mgr = MCPToolManager()
    assert mgr._resolve_env("run-${CODETUI_PART}-x") == "run-xyz-x"


@pytest.mark.asyncio
async def test_tool_manager_interpolates_command_and_args(monkeypatch):
    monkeypatch.setenv("CODETUI_HOME", "/home/ci")
    result = MagicMock()
    result.tools = []
    mock_client = MagicMock()
    mock_client.list_tools = AsyncMock(return_value=result)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("codetui.tools.Client", return_value=mock_client) as client_cls:
        mgr = MCPToolManager()
        mgr._config = {
            "mcpServers": {
                "s": {
                    "command": "${CODETUI_HOME}/bin/mcp",
                    "args": ["--cfg", "${CODETUI_HOME}/cfg.json"],
                }
            }
        }
        await mgr.initialize()
        params = client_cls.call_args.args[0]
        assert params.command == "/home/ci/bin/mcp"
        assert params.args == ["--cfg", "/home/ci/cfg.json"]
