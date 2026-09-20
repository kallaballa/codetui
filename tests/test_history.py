from unittest.mock import AsyncMock, MagicMock, patch

from codetui.agent import Agent
from codetui.history import (
    load_history,
    load_prompt_history,
    save_history,
    save_prompt_history,
)


def test_save_and_load_history_roundtrip(tmp_path):
    path = tmp_path / "nested" / "history.json"
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    save_history(path, messages)
    assert load_history(path) == messages


def test_load_history_missing_or_invalid(tmp_path):
    assert load_history(tmp_path / "missing.json") == []
    broken = tmp_path / "broken.json"
    broken.write_text("{not json")
    assert load_history(broken) == []
    not_list = tmp_path / "not_list.json"
    not_list.write_text('{"role": "user"}')
    assert load_history(not_list) == []


def test_load_history_drops_invalid_messages_but_keeps_tools(tmp_path):
    path = tmp_path / "history.json"
    path.write_text(
        '[{"role": "user", "content": "ok"},'
        ' {"role": "tool", "content": "secret"},'
        ' {"role": "assistant", "content": 5},'
        ' "nope"]'
    )
    assert load_history(path) == [
        {"role": "user", "content": "ok"},
        {"role": "tool", "content": "secret"},
    ]


async def test_agent_persists_and_clears_history(tmp_path):
    path = tmp_path / "history.json"
    with patch("codetui.agent.AsyncOpenAI") as mock_openai:
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Hello"
        mock_response.choices[0].message.tool_calls = None
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        agent = Agent(base_url="http://test", api_key="key", history_path=path)
        await agent.send_message("Hi")
        assert load_history(path) == [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]

        agent.clear_history()
        assert agent.history == []
        assert load_history(path) == []


def test_save_and_load_prompt_history_roundtrip(tmp_path):
    path = tmp_path / "nested" / "prompt_history.json"
    prompts = ["first question", "second question"]
    save_prompt_history(path, prompts)
    assert load_prompt_history(path) == prompts


def test_load_prompt_history_missing_or_invalid(tmp_path):
    assert load_prompt_history(tmp_path / "missing.json") == []
    broken = tmp_path / "broken.json"
    broken.write_text("{not json")
    assert load_prompt_history(broken) == []


def test_load_prompt_history_sanitizes(tmp_path):
    path = tmp_path / "prompt_history.json"
    path.write_text(
        '["  hi  ", "repeat", "repeat", "   ", 42, "ok", "ok"]'
    )
    assert load_prompt_history(path) == ["hi", "repeat", "ok"]
