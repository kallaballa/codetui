import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openai import APIConnectionError

from codetui.agent import Agent, MAX_TOOL_ROUNDS, TOOL_LOOP_MESSAGE
from codetui.tools import ToolDefinition


def _connection_error():
    import httpx

    return APIConnectionError(
        message="boom", request=httpx.Request("POST", "http://localhost")
    )


class _ToolCall:
    def __init__(self, tid, name, arguments):
        self.id = tid
        self.function = _ToolFunction(name, arguments)


class _ToolFunction:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


def _make_chunk(content=None, tc_index=None, tc_id=None, tc_name=None, tc_args=None):
    chunk = MagicMock()
    delta = chunk.choices[0].delta
    delta.content = content
    if tc_index is None:
        delta.tool_calls = None
    else:
        tc = MagicMock()
        tc.index = tc_index
        tc.id = tc_id
        fn = MagicMock()
        fn.name = tc_name
        fn.arguments = tc_args
        tc.function = fn
        delta.tool_calls = [tc]
    return chunk


def _mock_client():
    client = MagicMock()
    client.chat.completions.create = AsyncMock()
    return client


async def test_send_message():
    with patch("codetui.agent.AsyncOpenAI") as mock_openai:
        mock_client = _mock_client()
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
    with patch("codetui.agent.AsyncOpenAI") as mock_openai:
        mock_client = _mock_client()
        mock_openai.return_value = mock_client
        mock_chunk1 = MagicMock()
        mock_chunk1.choices[0].delta.content = "Hello"
        mock_chunk1.choices[0].delta.tool_calls = None
        mock_chunk2 = MagicMock()
        mock_chunk2.choices[0].delta.content = " world"
        mock_chunk2.choices[0].delta.tool_calls = None

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


async def test_agent_with_tools_send_message():
    with patch("codetui.agent.AsyncOpenAI") as mock_openai:
        mock_client = _mock_client()
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


async def test_stream_message_with_tool_calls_sends_valid_message():
    with patch("codetui.agent.AsyncOpenAI") as mock_openai:
        mock_client = _mock_client()
        mock_openai.return_value = mock_client

        tool_manager = MagicMock()
        tool_manager.list_tools.return_value = [ToolDefinition(name="t1", description="", parameters={})]
        tool_manager.call_tool = AsyncMock(return_value="tool result")

        def make_chunk(content=None, tc_index=None, tc_id=None, tc_name=None, tc_args=None):
            chunk = MagicMock()
            delta = chunk.choices[0].delta
            delta.content = content
            if tc_index is None:
                delta.tool_calls = None
            else:
                tc = MagicMock()
                tc.index = tc_index
                tc.id = tc_id
                fn = MagicMock()
                fn.name = tc_name
                fn.arguments = tc_args
                tc.function = fn
                delta.tool_calls = [tc]
            return chunk

        def final_chunk():
            return make_chunk(content="done")

        async def stream_with_tools():
            yield make_chunk(tc_index=0, tc_id="call_1", tc_name="t1", tc_args="")
            yield make_chunk(tc_index=0, tc_id=None, tc_name=None, tc_args='{"q":')
            yield make_chunk(tc_index=0, tc_id=None, tc_name=None, tc_args="1}")

        async def stream_final():
            yield final_chunk()

        mock_client.chat.completions.create.side_effect = [
            stream_with_tools(),
            stream_final(),
        ]

        agent = Agent(base_url="http://test", api_key="key", model="test-model", tool_manager=tool_manager)
        tokens = []
        async for token in agent.stream_message("Hi"):
            tokens.append(token)

        assert tokens == ["done"]
        assert tool_manager.call_tool.await_count == 1
        sent = mock_client.chat.completions.create.call_args_list[1].kwargs
        assert sent["model"] == "test-model"
        tool_calls_msg = [
            m
            for m in sent["messages"]
            if m.get("role") == "assistant" and "tool_calls" in m
        ][0]
        assert tool_calls_msg["tool_calls"] == [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "t1", "arguments": '{"q":1}'},
            }
        ]
        tool_result = [
            m
            for m in agent.history
            if m.get("role") == "tool"
        ][0]
        assert tool_result["tool_call_id"] == "call_1"


async def test_agent_with_tool_calls_loop():
    with patch("codetui.agent.AsyncOpenAI") as mock_openai:
        mock_client = _mock_client()
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


async def test_agent_emits_tool_events():
    with patch("codetui.agent.AsyncOpenAI") as mock_openai:
        mock_client = _mock_client()
        mock_openai.return_value = mock_client

        tool_manager = MagicMock()
        tool_manager.list_tools.return_value = [ToolDefinition(name="t1", description="", parameters={})]
        tool_manager.call_tool = AsyncMock(return_value="tool result")

        mock_msg_with_tools = MagicMock()
        mock_msg_with_tools.content = None
        mock_msg_with_tools.tool_calls = [
            _ToolCall("call_1", "t1", '{"q":1}')
        ]
        mock_msg_with_tools.dict.return_value = {"role": "assistant", "tool_calls": [{"id": "call_1"}]}

        mock_final_msg = MagicMock()
        mock_final_msg.content = "done"
        mock_final_msg.tool_calls = None
        mock_final_msg.dict.return_value = {"role": "assistant", "content": "done"}

        mock_client.chat.completions.create.side_effect = [
            MagicMock(choices=[MagicMock(message=mock_msg_with_tools)]),
            MagicMock(choices=[MagicMock(message=mock_final_msg)]),
        ]

        events = []
        agent = Agent(
            base_url="http://test",
            api_key="key",
            model="test-model",
            tool_manager=tool_manager,
            listener=events.append,
        )
        await agent.send_message("Hi")

        assert events == [
            {"type": "tool_start", "name": "t1", "arguments": '{"q":1}'},
            {"type": "tool_end", "name": "t1", "result": "tool result"},
        ]


async def test_agent_set_listener_replaces_listener():
    with patch("codetui.agent.AsyncOpenAI") as mock_openai:
        mock_client = _mock_client()
        mock_openai.return_value = mock_client

        tool_manager = MagicMock()
        tool_manager.list_tools.return_value = [ToolDefinition(name="t1", description="", parameters={})]
        tool_manager.call_tool = AsyncMock(return_value="tool result")

        mock_msg_with_tools = MagicMock()
        mock_msg_with_tools.content = None
        mock_msg_with_tools.tool_calls = [_ToolCall("call_1", "t1", "{}")]
        mock_msg_with_tools.dict.return_value = {"role": "assistant", "tool_calls": [{"id": "call_1"}]}

        mock_final_msg = MagicMock()
        mock_final_msg.content = "done"
        mock_final_msg.tool_calls = None
        mock_final_msg.dict.return_value = {"role": "assistant", "content": "done"}

        mock_client.chat.completions.create.side_effect = [
            MagicMock(choices=[MagicMock(message=mock_msg_with_tools)]),
            MagicMock(choices=[MagicMock(message=mock_final_msg)]),
        ]

        events = []
        agent = Agent(base_url="http://test", api_key="key", model="test-model", tool_manager=tool_manager)
        agent.set_listener(events.append)
        await agent.send_message("Hi")

        assert len(events) == 2
        assert events[0]["type"] == "tool_start"
        assert events[1]["type"] == "tool_end"
        assert events[1]["result"] == "tool result"


async def test_send_message_stops_after_max_tool_rounds():
    with patch("codetui.agent.AsyncOpenAI") as mock_openai:
        mock_client = _mock_client()
        mock_openai.return_value = mock_client

        tool_manager = MagicMock()
        tool_manager.list_tools.return_value = [ToolDefinition(name="t1", description="", parameters={})]
        tool_manager.call_tool = AsyncMock(return_value="tool result")

        mock_tool_msg = MagicMock()
        mock_tool_msg.content = None
        mock_tool_msg.tool_calls = [_ToolCall("call_1", "t1", "{}")]
        mock_tool_msg.dict.return_value = {"role": "assistant", "tool_calls": [{"id": "call_1"}]}

        mock_client.chat.completions.create = AsyncMock(
            return_value=MagicMock(choices=[MagicMock(message=mock_tool_msg)])
        )

        agent = Agent(base_url="http://test", api_key="key", model="test-model", tool_manager=tool_manager)
        response = await agent.send_message("Hi")

        assert response == TOOL_LOOP_MESSAGE
        assert mock_client.chat.completions.create.await_count == MAX_TOOL_ROUNDS
        assert tool_manager.call_tool.await_count == MAX_TOOL_ROUNDS
        assert agent.history[-1]["role"] == "tool"
        assert TOOL_LOOP_MESSAGE not in [m.get("content") for m in agent.history]


async def test_stream_message_stops_after_max_tool_rounds():
    with patch("codetui.agent.AsyncOpenAI") as mock_openai:
        mock_client = _mock_client()
        mock_openai.return_value = mock_client

        tool_manager = MagicMock()
        tool_manager.list_tools.return_value = [ToolDefinition(name="t1", description="", parameters={})]
        tool_manager.call_tool = AsyncMock(return_value="tool result")

        async def tool_stream():
            chunk = MagicMock()
            delta = chunk.choices[0].delta
            delta.content = None
            tc = MagicMock()
            tc.index = 0
            tc.id = "call_1"
            fn = MagicMock()
            fn.name = "t1"
            fn.arguments = "{}"
            tc.function = fn
            delta.tool_calls = [tc]
            yield chunk

        mock_client.chat.completions.create.side_effect = [
            tool_stream() for _ in range(MAX_TOOL_ROUNDS)
        ]

        agent = Agent(base_url="http://test", api_key="key", model="test-model", tool_manager=tool_manager)
        tokens = []
        async for token in agent.stream_message("Hi"):
            tokens.append(token)

        assert tokens == [TOOL_LOOP_MESSAGE]
        assert mock_client.chat.completions.create.await_count == MAX_TOOL_ROUNDS
        assert tool_manager.call_tool.await_count == MAX_TOOL_ROUNDS
        assert agent.history[-1]["role"] == "tool"
        assert TOOL_LOOP_MESSAGE not in [m.get("content") for m in agent.history]


async def test_send_message_retries_on_connection_error():
    with patch("codetui.agent.AsyncOpenAI") as mock_openai, patch(
        "codetui.agent.asyncio.sleep", AsyncMock()
    ) as mock_sleep:
        mock_client = _mock_client()
        mock_openai.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Hello"
        mock_client.chat.completions.create = AsyncMock(
            side_effect=[_connection_error(), mock_response]
        )

        events = []
        agent = Agent(
            base_url="http://test",
            api_key="key",
            model="test-model",
            listener=events.append,
        )
        response = await agent.send_message("Hi")

        assert response == "Hello"
        assert mock_client.chat.completions.create.await_count == 2
        assert mock_sleep.await_count == 1
        assert events == [{"type": "connection_retry", "attempt": 1}]


async def test_send_message_gives_up_after_max_retries():
    with patch("codetui.agent.AsyncOpenAI") as mock_openai, patch(
        "codetui.agent.asyncio.sleep", AsyncMock()
    ) as mock_sleep:
        mock_client = _mock_client()
        mock_openai.return_value = mock_client
        mock_client.chat.completions.create = AsyncMock(
            side_effect=_connection_error()
        )

        agent = Agent(
            base_url="http://test",
            api_key="key",
            model="test-model",
            max_connection_retries=3,
        )
        try:
            await agent.send_message("Hi")
        except APIConnectionError:
            pass
        else:
            raise AssertionError("expected APIConnectionError")

        assert mock_client.chat.completions.create.await_count == 3
        assert mock_sleep.await_count == 2


async def test_stream_message_retries_on_connection_error():
    with patch("codetui.agent.AsyncOpenAI") as mock_openai, patch(
        "codetui.agent.asyncio.sleep", AsyncMock()
    ) as mock_sleep:
        mock_client = _mock_client()
        mock_openai.return_value = mock_client

        mock_chunk = MagicMock()
        mock_chunk.choices[0].delta.content = "Hello"
        mock_chunk.choices[0].delta.tool_calls = None

        async def mock_stream():
            yield mock_chunk

        mock_client.chat.completions.create = AsyncMock(
            side_effect=[_connection_error(), mock_stream()]
        )

        agent = Agent(base_url="http://test", api_key="key", model="test-model")
        tokens = []
        async for token in agent.stream_message("Hi"):
            tokens.append(token)

        assert tokens == ["Hello"]
        assert mock_client.chat.completions.create.await_count == 2
        assert mock_sleep.await_count == 1


async def test_stream_message_ignores_empty_choices_chunks():
    with patch("codetui.agent.AsyncOpenAI") as mock_openai:
        mock_client = _mock_client()
        mock_openai.return_value = mock_client
        blank = MagicMock()
        blank.choices = []
        chunk = MagicMock()
        chunk.choices[0].delta.content = "hello"
        chunk.choices[0].delta.tool_calls = None

        async def mock_stream():
            yield blank
            yield chunk

        mock_client.chat.completions.create.return_value = mock_stream()

        agent = Agent(base_url="http://test", api_key="key", model="test-model")
        tokens = []
        async for token in agent.stream_message("Hi"):
            tokens.append(token)

        assert tokens == ["hello"]
        assert len(agent.history) == 2
        assert agent.history[1]["content"] == "hello"


async def test_send_message_restores_history_on_failure():
    with patch("codetui.agent.AsyncOpenAI") as mock_openai:
        mock_client = _mock_client()
        mock_openai.return_value = mock_client

        tool_manager = MagicMock()
        tool_manager.list_tools.return_value = [ToolDefinition(name="t1", description="", parameters={})]
        tool_manager.call_tool = AsyncMock(return_value="tool result")

        mock_tool_msg = MagicMock()
        mock_tool_msg.content = None
        mock_tool_msg.tool_calls = [_ToolCall("call_1", "t1", "{}")]
        mock_tool_msg.dict.return_value = {"role": "assistant", "tool_calls": [{"id": "call_1"}]}

        mock_client.chat.completions.create = AsyncMock(
            side_effect=[
                MagicMock(choices=[MagicMock(message=mock_tool_msg)]),
                RuntimeError("boom"),
            ]
        )

        agent = Agent(
            base_url="http://test",
            api_key="key",
            model="test-model",
            tool_manager=tool_manager,
        )
        try:
            await agent.send_message("Hi")
        except RuntimeError:
            pass
        else:
            raise AssertionError("expected RuntimeError")

        assert agent.history == []


async def test_stream_message_restores_history_on_failure():
    with patch("codetui.agent.AsyncOpenAI") as mock_openai:
        mock_client = _mock_client()
        mock_openai.return_value = mock_client

        tool_manager = MagicMock()
        tool_manager.list_tools.return_value = [ToolDefinition(name="t1", description="", parameters={})]
        tool_manager.call_tool = AsyncMock(return_value="tool result")

        chunk = MagicMock()
        delta = chunk.choices[0].delta
        delta.content = None
        tc = MagicMock()
        tc.index = 0
        tc.id = "call_1"
        fn = MagicMock()
        fn.name = "t1"
        fn.arguments = "{}"
        tc.function = fn
        delta.tool_calls = [tc]

        async def tool_stream():
            yield chunk

        mock_client.chat.completions.create = AsyncMock(
            side_effect=[tool_stream(), RuntimeError("boom")]
        )

        agent = Agent(
            base_url="http://test",
            api_key="key",
            model="test-model",
            tool_manager=tool_manager,
        )
        try:
            async for _ in agent.stream_message("Hi"):
                pass
        except RuntimeError:
            pass
        else:
            raise AssertionError("expected RuntimeError")

        assert agent.history == []
        assert tool_manager.call_tool.await_count == 1


async def test_stream_message_merges_unnumbered_tool_call_deltas():
    with patch("codetui.agent.AsyncOpenAI") as mock_openai:
        mock_client = _mock_client()
        mock_openai.return_value = mock_client

        tool_manager = MagicMock()
        tool_manager.list_tools.return_value = [ToolDefinition(name="t1", description="", parameters={})]
        tool_manager.call_tool = AsyncMock(return_value="tool result")

        def make_chunk(tc_index, tc_id=None, tc_name=None, tc_args=None):
            chunk = MagicMock()
            delta = chunk.choices[0].delta
            delta.content = None
            tc = MagicMock()
            tc.index = tc_index
            tc.id = tc_id
            fn = MagicMock()
            fn.name = tc_name
            fn.arguments = tc_args
            tc.function = fn
            delta.tool_calls = [tc]
            return chunk

        def final_chunk():
            chunk = MagicMock()
            chunk.choices[0].delta.content = "done"
            chunk.choices[0].delta.tool_calls = None
            return chunk

        async def stream_with_tools():
            yield make_chunk(tc_index=0, tc_id="call_1", tc_name="t1", tc_args='{"a":')
            yield make_chunk(tc_index=None, tc_id=None, tc_name=None, tc_args='1}')
            yield make_chunk(tc_index=1, tc_id="call_2", tc_name="t1", tc_args='{"b":2}')
            yield make_chunk(tc_index=None, tc_id=None, tc_name=None, tc_args="")

        async def stream_final():
            yield final_chunk()

        mock_client.chat.completions.create.side_effect = [
            stream_with_tools(),
            stream_final(),
        ]

        agent = Agent(base_url="http://test", api_key="key", model="test-model", tool_manager=tool_manager)
        tokens = []
        async for token in agent.stream_message("Hi"):
            tokens.append(token)

        assert tokens == ["done"]
        assert tool_manager.call_tool.await_count == 2
        tool_calls_msg = [
            m
            for m in mock_client.chat.completions.create.call_args_list[1].kwargs["messages"]
            if m.get("role") == "assistant" and "tool_calls" in m
        ][0]
        assert tool_calls_msg["tool_calls"] == [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "t1", "arguments": '{"a":1}'},
            },
            {
                "id": "call_2",
                "type": "function",
                "function": {"name": "t1", "arguments": '{"b":2}'},
            },
        ]


async def test_send_message_restores_history_on_cancel():
    with patch("codetui.agent.AsyncOpenAI") as mock_openai:
        mock_client = _mock_client()
        mock_openai.return_value = mock_client
        mock_client.chat.completions.create = AsyncMock(side_effect=asyncio.CancelledError())

        agent = Agent(base_url="http://test", api_key="key", model="test-model")
        try:
            await agent.send_message("Hi")
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError("expected CancelledError")

        assert agent.history == []


def test_agent_accepts_missing_api_key():
    with patch("codetui.agent.AsyncOpenAI") as mock_openai:
        agent = Agent(base_url="http://test", api_key=None, model="test-model")
        assert agent.client is mock_openai.return_value
        mock_openai.assert_called_once_with(
            base_url="http://test", api_key="ollama"
        )


async def test_stream_message_merges_interleaved_tool_call_deltas():
    with patch("codetui.agent.AsyncOpenAI") as mock_openai:
        mock_client = _mock_client()
        mock_openai.return_value = mock_client

        tool_manager = MagicMock()
        tool_manager.list_tools.return_value = [ToolDefinition(name="t1", description="", parameters={})]
        tool_manager.call_tool = AsyncMock(return_value="tool result")

        def make_chunk(tc_index, tc_id=None, tc_name=None, tc_args=None):
            chunk = MagicMock()
            delta = chunk.choices[0].delta
            delta.content = None
            tc = MagicMock()
            tc.index = tc_index
            tc.id = tc_id
            fn = MagicMock()
            fn.name = tc_name
            fn.arguments = tc_args
            tc.function = fn
            delta.tool_calls = [tc]
            return chunk

        def final_chunk():
            chunk = MagicMock()
            chunk.choices[0].delta.content = "done"
            chunk.choices[0].delta.tool_calls = None
            return chunk

        async def stream_with_tools():
            yield make_chunk(0, "call_1", "t1", '{"a":')
            yield make_chunk(1, "call_2", "t1", '{"b":')
            yield make_chunk(0, None, None, "1}")
            yield make_chunk(1, None, None, "2}")

        async def stream_final():
            yield final_chunk()

        mock_client.chat.completions.create.side_effect = [
            stream_with_tools(),
            stream_final(),
        ]

        agent = Agent(base_url="http://test", api_key="key", model="test-model", tool_manager=tool_manager)
        tokens = []
        async for token in agent.stream_message("Hi"):
            tokens.append(token)

        assert tokens == ["done"]
        assert tool_manager.call_tool.await_count == 2
        tool_calls_msg = [
            m
            for m in mock_client.chat.completions.create.call_args_list[1].kwargs["messages"]
            if m.get("role") == "assistant" and "tool_calls" in m
        ][0]
        assert tool_calls_msg["tool_calls"] == [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "t1", "arguments": '{"a":1}'},
            },
            {
                "id": "call_2",
                "type": "function",
                "function": {"name": "t1", "arguments": '{"b":2}'},
            },
        ]


async def test_stream_message_keeps_preamble_when_tool_called():
    with patch("codetui.agent.AsyncOpenAI") as mock_openai:
        mock_client = _mock_client()
        mock_openai.return_value = mock_client

        tool_manager = MagicMock()
        tool_manager.list_tools.return_value = [ToolDefinition(name="t1", description="", parameters={})]
        tool_manager.call_tool = AsyncMock(return_value="tool result")

        async def stream_with_content_and_tool():
            yield _make_chunk(content="Looking up...", tc_index=None)
            yield _make_chunk(content=None, tc_index=0, tc_id="call_1", tc_name="t1", tc_args="{}")

        async def stream_final():
            yield _make_chunk(content="done", tc_index=None)

        mock_client.chat.completions.create.side_effect = [
            stream_with_content_and_tool(),
            stream_final(),
        ]

        agent = Agent(base_url="http://test", api_key="key", model="test-model", tool_manager=tool_manager)
        tokens = []
        async for token in agent.stream_message("Hi"):
            tokens.append(token)

        assert tokens == ["Looking up...", "done"]
        assert {"role": "assistant", "content": "Looking up..."} in agent.history
        for m in agent.history:
            if m.get("role") == "assistant":
                assert not (m.get("content") and "tool_calls" in m)


async def test_send_message_preserves_content_when_tool_called():
    with patch("codetui.agent.AsyncOpenAI") as mock_openai:
        mock_client = _mock_client()
        mock_openai.return_value = mock_client

        tool_manager = MagicMock()
        tool_manager.list_tools.return_value = [ToolDefinition(name="t1", description="", parameters={})]
        tool_manager.call_tool = AsyncMock(return_value="tool result")

        mock_msg_with_tools = MagicMock()
        mock_msg_with_tools.content = "ok, searching"
        mock_msg_with_tools.tool_calls = [_ToolCall("call_1", "t1", "{}")]

        mock_final_msg = MagicMock()
        mock_final_msg.content = "done"
        mock_final_msg.tool_calls = None

        mock_client.chat.completions.create.side_effect = [
            MagicMock(choices=[MagicMock(message=mock_msg_with_tools)]),
            MagicMock(choices=[MagicMock(message=mock_final_msg)]),
        ]

        agent = Agent(base_url="http://test", api_key="key", model="test-model", tool_manager=tool_manager)
        response = await agent.send_message("Hi")

        assert response == "done"
        assert {"role": "assistant", "content": "ok, searching"} in agent.history
        tool_calls_msg = [m for m in agent.history if m.get("role") == "assistant" and "tool_calls" in m][0]
        assert tool_calls_msg["tool_calls"] == [
            {"id": "call_1", "type": "function", "function": {"name": "t1", "arguments": "{}"}}
        ]
        for m in agent.history:
            if m.get("role") == "assistant":
                assert not (m.get("content") and "tool_calls" in m)


async def test_stream_message_restores_history_on_close():
    with patch("codetui.agent.AsyncOpenAI") as mock_openai:
        mock_client = _mock_client()
        mock_openai.return_value = mock_client

        chunk = MagicMock()
        chunk.choices[0].delta.content = "partial"
        chunk.choices[0].delta.tool_calls = None

        async def mock_stream():
            yield chunk
            await asyncio.sleep(3600)

        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream())

        agent = Agent(base_url="http://test", api_key="key")
        agen = agent.stream_message("Hi")
        tokens = []
        async for token in agen:
            tokens.append(token)
            break
        await agen.aclose()

        assert tokens == ["partial"]
        assert agent.history == []


def test_agent_trims_history_but_keeps_tool_pairs():
    with patch("codetui.agent.AsyncOpenAI"):
        agent = Agent(base_url="http://test", api_key="key", max_history_messages=3)
        agent.history = [
            {"role": "user", "content": "old"},
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "t1", "arguments": "{}"}}
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "result"},
            {"role": "assistant", "content": "done"},
            {"role": "user", "content": "new"},
        ]
        messages = agent._messages()
        assert messages[0]["role"] == "system"
        assert messages[1:] == [
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "t1", "arguments": "{}"}}
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "result"},
            {"role": "assistant", "content": "done"},
            {"role": "user", "content": "new"},
        ]
