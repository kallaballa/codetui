import asyncio
import io

import pytest
from rich.console import Console
from textual.geometry import Region
from textual.widgets import Select

from codetui.history import load_prompt_history
from codetui.ui import ChatScroll, ModelSelectScreen, PromptInput, ToolCallView, TUI


class DummyAgent:
    def __init__(self):
        self.model = "test-model"
        self.tool_manager = None
        self.history = []

    async def stream_message(self, content: str):
        self.history.append({"role": "user", "content": content})
        response = f"echo: {content}"
        for chunk in response.split(" "):
            yield chunk + " "
        self.history.append({"role": "assistant", "content": response})

    async def send_message(self, content: str) -> str:
        response = f"echo: {content}"
        self.history.append({"role": "user", "content": content})
        self.history.append({"role": "assistant", "content": response})
        return response

    def clear_history(self) -> None:
        self.history = []


class ToolAgent:
    def __init__(self):
        self.model = "test-model"
        self.tool_manager = None
        self.history = []
        self.listener = None

    def set_listener(self, listener):
        self.listener = listener

    def _emit(self, event):
        if self.listener:
            self.listener(event)

    async def stream_message(self, content):
        self.history.append({"role": "user", "content": content})
        self._emit(
            {"type": "tool_start", "name": "search", "arguments": {"query": "foo"}}
        )
        yield "Looking things up... "
        self._emit(
            {"type": "tool_end", "name": "search", "result": "found 3 results"}
        )
        yield "Done"
        self.history.append(
            {"role": "assistant", "content": "Looking things up... Done"}
        )


def chat_text(tui: TUI) -> str:
    console = Console(width=100, record=True, file=io.StringIO())
    for widget in tui.query("#chat Static"):
        console.print(widget.content)
    return console.export_text()


@pytest.mark.asyncio
async def test_input_not_discarded_while_tools_loading():
    agent = DummyAgent()
    tui = TUI(agent=agent)
    async with tui.run_test() as pilot:
        tui._tools_loading = True
        input_widget = tui.query_one("#input")
        input_widget.value = "hello"
        await pilot.press("enter")
        assert input_widget.value == "hello"
        assert agent.history == []

        tui._tools_loading = False
        await pilot.press("enter")
        await pilot.pause()
        assert input_widget.value == ""
        assert agent.history == [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "echo: hello"},
        ]
        rendered = chat_text(tui)
        assert "You" in rendered
        assert "Assistant" in rendered
        assert "echo: hello" in rendered
        assert not tui._sending


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 24), (40, 24)])
async def test_model_dialog_from_focused_input(size):
    agent = DummyAgent()
    tui = TUI(agent=agent, models=["other-model"])
    async with tui.run_test(size=size) as pilot:
        input_widget = tui.query_one("#input")
        input_widget.value = "draft"
        await pilot.press("ctrl+p")
        await pilot.pause()
        assert isinstance(tui.screen, ModelSelectScreen)
        dialog = tui.screen.query_one("#model_dialog")
        assert tui.screen.region.contains_region(dialog.region)
        for selector in ("#model_title", "#model_hint", "#model_select", "#cancel", "#switch"):
            widget = tui.screen.query_one(selector)
            assert widget.region.width > 0 and widget.region.height > 0
            assert dialog.region.contains_region(widget.region)
        select = tui.screen.query_one("#model_select", Select)
        assert select.value == agent.model
        select.value = "other-model"
        await pilot.click("#switch")
        await pilot.pause()
        assert agent.model == "other-model"
        assert input_widget.value == "draft"
        assert input_widget.has_focus
        await pilot.press("ctrl+p")
        await pilot.press("escape")
        await pilot.pause()
        assert not isinstance(tui.screen, ModelSelectScreen)


@pytest.mark.asyncio
async def test_clear_view_preserves_history_and_draft():
    agent = DummyAgent()
    tui = TUI(agent=agent)
    async with tui.run_test() as pilot:
        input_widget = tui.query_one("#input")
        input_widget.value = "hello"
        await pilot.press("enter")
        await pilot.pause()
        history = list(agent.history)
        input_widget.value = "draft"
        await pilot.press("ctrl+l")
        await pilot.pause()
        rendered = chat_text(tui)
        assert "View cleared" in rendered
        assert "echo: hello" not in rendered
        assert agent.history == history
        assert input_widget.value == "draft"


@pytest.mark.asyncio
async def test_new_conversation_clears_history():
    agent = DummyAgent()
    tui = TUI(agent=agent)
    async with tui.run_test() as pilot:
        input_widget = tui.query_one("#input")
        input_widget.value = "hello"
        await pilot.press("enter")
        await pilot.pause()
        assert agent.history
        await pilot.press("ctrl+n")
        await pilot.pause()
        assert agent.history == []
        assert "Started a new conversation" in chat_text(tui)


@pytest.mark.asyncio
async def test_history_is_restored_on_mount():
    agent = DummyAgent()
    agent.history = [
        {"role": "user", "content": "earlier question"},
        {"role": "assistant", "content": "earlier answer"},
    ]
    tui = TUI(agent=agent)
    async with tui.run_test():
        rendered = chat_text(tui)
        assert "Resumed conversation" in rendered
        assert "earlier question" in rendered
        assert "earlier answer" in rendered


@pytest.mark.asyncio
async def test_tool_call_activity_displayed():
    agent = ToolAgent()
    tui = TUI(agent=agent)
    async with tui.run_test() as pilot:
        input_widget = tui.query_one("#input")
        input_widget.value = "search for foo"
        await pilot.press("enter")
        await pilot.pause()
        rendered = chat_text(tui)
        assert "Tool: search" in rendered
        assert "found 3 results" in rendered
        assert "Running" not in rendered


@pytest.mark.asyncio
async def test_tool_call_view_renders_on_screen():
    agent = DummyAgent()
    tui = TUI(agent=agent)
    async with tui.run_test() as pilot:
        tui._on_tool_event(
            {"type": "tool_start", "name": "search", "arguments": {"q": "x"}}
        )
        await pilot.pause()
        view = tui.query_one(ToolCallView)
        assert view.size.height > 0
        strips = view.render_lines(Region(0, 0, view.size.width, view.size.height))
        text = "".join(strip.text for strip in strips)
        assert "Tool: search" in text


@pytest.mark.asyncio
async def test_tool_call_expand_toggle():
    agent = ToolAgent()
    tui = TUI(agent=agent)
    async with tui.run_test() as pilot:
        input_widget = tui.query_one("#input")
        input_widget.value = "search"
        await pilot.press("enter")
        await pilot.pause()
        view = tui.query_one(ToolCallView)
        assert not view.expanded
        assert view.result_text == "found 3 results"

        view.toggle_expand()
        await pilot.pause()
        assert view.expanded
        assert "click to collapse" in chat_text(tui)

        view.toggle_expand()
        await pilot.pause()
        assert not view.expanded
        assert "click to expand" in chat_text(tui)


@pytest.mark.asyncio
async def test_status_shows_running_tool():
    agent = DummyAgent()
    tui = TUI(agent=agent)
    async with tui.run_test() as pilot:
        tui._on_tool_event(
            {"type": "tool_start", "name": "search", "arguments": {"q": "x"}}
        )
        await pilot.pause()
        status = tui.query_one("#status")
        assert "Running: search" in str(status.content)
        assert "running..." in chat_text(tui)

        tui._on_tool_event(
            {"type": "tool_end", "name": "search", "result": "ok"}
        )
        await pilot.pause()
        status = tui.query_one("#status")
        assert "Running: search" not in str(status.content)


@pytest.mark.asyncio
async def test_tool_result_attached_to_matching_view():
    agent = DummyAgent()
    tui = TUI(agent=agent)
    async with tui.run_test() as pilot:
        tui._on_tool_event({"type": "tool_start", "name": "read", "arguments": {}})
        tui._on_tool_event({"type": "tool_start", "name": "grep", "arguments": {}})
        await pilot.pause()
        assert [v.tool_name for v in tui._tool_views] == ["read", "grep"]

        tui._on_tool_event({"type": "tool_end", "name": "grep", "result": "grep result"})
        tui._on_tool_event({"type": "tool_end", "name": "read", "result": "read result"})
        await pilot.pause()

        read_view, grep_view = tui._tool_views
        assert read_view.result_text == "read result"
        assert grep_view.result_text == "grep result"
        assert not tui._active_tool


@pytest.mark.asyncio
async def test_tool_call_expanded_preserves_whitespace():
    agent = DummyAgent()
    tui = TUI(agent=agent)
    async with tui.run_test() as pilot:
        tui._on_tool_event(
            {"type": "tool_start", "name": "read", "arguments": {"path": "a.txt"}}
        )
        tui._on_tool_event(
            {"type": "tool_end", "name": "read", "result": "line one\n\n  indented line"}
        )
        await pilot.pause()
        view = tui.query_one(ToolCallView)
        assert view.result_text == "line one\n\n  indented line"
        view.toggle_expand()
        await pilot.pause()
        assert view.expanded
        rendered = chat_text(tui)
        assert "indented line" in rendered
        assert "line one indented line" not in rendered


class BlockingAgent(DummyAgent):
    def __init__(self):
        super().__init__()
        self.release = asyncio.Event()

    async def stream_message(self, content: str):
        self.history.append({"role": "user", "content": content})
        yield "partial "
        await self.release.wait()
        yield "done"
        self.history.append({"role": "assistant", "content": "partial done"})


async def _submit(tui: TUI, pilot, content: str) -> None:
    input_widget = tui.query_one("#input", PromptInput)
    input_widget.value = content
    await pilot.press("enter")
    await pilot.pause()
    worker = tui._response_worker
    if worker is not None:
        await worker.wait()
    await pilot.pause()


@pytest.mark.asyncio
async def test_shift_enter_inserts_newline_and_enter_sends():
    agent = DummyAgent()
    tui = TUI(agent=agent)
    async with tui.run_test() as pilot:
        input_widget = tui.query_one("#input", PromptInput)
        input_widget.focus()
        input_widget.value = "line one"
        await pilot.press("shift+enter")
        input_widget.insert("line two")
        await pilot.pause()
        assert input_widget.text == "line one\nline two"
        assert agent.history == []

        await pilot.press("enter")
        await pilot.pause()
        await tui._response_worker.wait()
        assert agent.history[0] == {"role": "user", "content": "line one\nline two"}
        assert input_widget.text == ""


@pytest.mark.asyncio
async def test_input_history_navigation():
    agent = DummyAgent()
    tui = TUI(agent=agent)
    async with tui.run_test() as pilot:
        input_widget = tui.query_one("#input", PromptInput)
        await _submit(tui, pilot, "first")
        await _submit(tui, pilot, "second")

        await pilot.press("up")
        await pilot.pause()
        assert input_widget.text == "second"
        await pilot.press("up")
        await pilot.pause()
        assert input_widget.text == "first"
        await pilot.press("down")
        await pilot.pause()
        assert input_widget.text == "second"
        await pilot.press("down")
        await pilot.pause()
        assert input_widget.text == ""


@pytest.mark.asyncio
async def test_input_history_persisted_across_restarts(tmp_path):
    path = tmp_path / "prompt_history.json"
    agent = DummyAgent()
    tui = TUI(agent=agent, prompt_history_path=path)
    async with tui.run_test() as pilot:
        await _submit(tui, pilot, "first question")
        await _submit(tui, pilot, "second question")
        assert load_prompt_history(path) == ["first question", "second question"]

    restart = TUI(agent=DummyAgent(), prompt_history_path=path)
    async with restart.run_test() as pilot:
        input_widget = restart.query_one("#input", PromptInput)
        await pilot.press("up")
        await pilot.pause()
        assert input_widget.text == "second question"
        await pilot.press("up")
        await pilot.pause()
        assert input_widget.text == "first question"


@pytest.mark.asyncio
async def test_new_conversation_clears_prompt_history(tmp_path):
    path = tmp_path / "prompt_history.json"
    agent = DummyAgent()
    tui = TUI(agent=agent, prompt_history_path=path)
    async with tui.run_test() as pilot:
        await _submit(tui, pilot, "keep me")
        assert load_prompt_history(path) == ["keep me"]

        await pilot.press("ctrl+n")
        await pilot.pause()

        input_widget = tui.query_one("#input", PromptInput)
        assert input_widget._history == []
        assert not path.exists()


@pytest.mark.asyncio
async def test_escape_interrupts_streaming():
    agent = BlockingAgent()
    tui = TUI(agent=agent)
    async with tui.run_test() as pilot:
        input_widget = tui.query_one("#input", PromptInput)
        input_widget.value = "long running"
        await pilot.press("enter")
        await pilot.pause()
        assert tui._sending

        await pilot.press("escape")
        assert tui._response_worker is not None
        await tui._response_worker.wait()
        await pilot.pause()
        assert not tui._sending
        assert "interrupted" in chat_text(tui)


@pytest.mark.asyncio
async def test_jump_to_end_reenables_follow():
    agent = DummyAgent()
    tui = TUI(agent=agent)
    async with tui.run_test() as pilot:
        chat = tui.query_one("#chat", ChatScroll)
        chat.follow_output = False
        tui.action_jump_to_end()
        await pilot.pause()
        assert chat.follow_output
