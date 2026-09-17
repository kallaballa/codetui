import pytest

from textual.widgets import RichLog, Select

from codetui.ui import ModelSelectScreen, TUI


class DummyAgent:
    def __init__(self):
        self.model = "test-model"
        self.tool_manager = None
        self.history = []

    async def send_message(self, content: str) -> str:
        self.history.append({"role": "user", "content": content})
        self.history.append({"role": "assistant", "content": f"echo: {content}"})
        return f"echo: {content}"


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
        rendered = "\n".join(line.text for line in tui.query_one("#chat", RichLog).lines)
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
        rendered = "\n".join(line.text for line in tui.query_one("#chat", RichLog).lines)
        assert "View cleared" in rendered
        assert "echo: hello" not in rendered
        assert agent.history == history
        assert input_widget.value == "draft"
