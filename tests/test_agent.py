import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from codetui.agent import Agent
from codetui.tools import MCPToolManager, ToolDefinition, to_openai_tools


async def test_send_message():
    with patch("codetui.agent.OpenAI") as mock_openai:
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Hello"
        mock_client.chat.completions.create.return_value = mock_response

        agent = Agent(base_url="http://test", api_key="key", model="test-model")
        response = await agent.send_message("Hi")
        assert response == "Hello"
        assert len(agent.history) == 2
        assert agent.history[0] == {"role": "user", "content": "Hi"}
        assert agent.history[1] == {"role": "assistant", "content": "Hello"}


async def test_stream_message():
    with patch("codetui.agent.OpenAI") as mock_openai:
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_chunk1 = MagicMock()
        mock_chunk1.choices[0].delta.content = "Hello"
        mock_chunk2 = MagicMock()
        mock_chunk2.choices[0].delta.content = " world"

        async def mock_stream():
            yield mock_chunk1
            yield mock_chunk2

        mock_client.chat.completions.create.return_value = mock_stream()

        agent = Agent(base_url="http://test", api_key="key", model="test-model")
        tokens = []
        async for token in agent.stream_message("Hi"):
            tokens.append(token)
        assert tokens == ["Hello", " world"]
        assert len(agent.history) == 2
        assert agent.history[1]["content"] == "Hello world"


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


async def test_agent_with_tools_send_message():
    with patch("codetui.agent.OpenAI") as mock_openai:
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        mock_msg = MagicMock()
        mock_msg.content = "final"
        mock_msg.tool_calls = None
        mock_msg.dict.return_value = {"role": "assistant", "content": "final"}

        mock_response = MagicMock()
        mock_response.choices[0].message = mock_msg
        mock_client.chat.completions.create.return_value = mock_response

        tool_manager = MagicMock()
        tool_manager.list_tools.return_value = [ToolDefinition(name="t1", description="", parameters={})]

        agent = Agent(base_url="http://test", api_key="key", model="test-model", tool_manager=tool_manager)
        response = await agent.send_message("Hi")
        assert response == "final"
        assert len(agent.history) == 2
        assert agent.history[0] == {"role": "user", "content": "Hi"}
        assert agent.history[1] == {"role": "assistant", "content": "final"}


async def test_agent_with_tool_calls_loop():
    with patch("codetui.agent.OpenAI") as mock_openai:
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        tool_manager = MagicMock()
        tool_manager.list_tools.return_value = [ToolDefinition(name="t1", description="", parameters={})]
        tool_manager.call_tool = AsyncMock(return_value="tool result")

        mock_msg_with_tools = MagicMock()
        mock_msg_with_tools.content = None
        mock_msg_with_tools.tool_calls = [MagicMock(function=MagicMock(name="t1", arguments="{}"), id="call_1")]
        mock_msg_with_tools.dict.return_value = {"role": "assistant", "tool_calls": [{"id": "call_1"}]}

        mock_final_msg = MagicMock()
        mock_final_msg.content = "done"
        mock_final_msg.tool_calls = None
        mock_final_msg.dict.return_value = {"role": "assistant", "content": "done"}

        mock_client.chat.completions.create.side_effect = [
            MagicMock(choices=[MagicMock(message=mock_msg_with_tools)]),
            MagicMock(choices=[MagicMock(message=mock_final_msg)]),
        ]

        agent = Agent(base_url="http://test", api_key="key", model="test-model", tool_manager=tool_manager)
        response = await agent.send_message("Hi")
        assert response == "done"
        assert tool_manager.call_tool.await_count == 1
