import asyncio
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Header, Footer, Input, RichLog, Static, Button, Select
from rich.text import Text
from rich.markdown import Markdown
from openai import OpenAI
from .agent import Agent


class ModelSelectScreen(ModalScreen):
    CSS = """
    ModelSelectScreen {
        align: center middle;
    }
    #model_dialog {
        width: 60;
        height: 20;
        border: thick $primary;
        padding: 1;
    }
    """

    def __init__(self, models: list[str], current_model: str, on_select):
        super().__init__()
        self.models = models
        self.current_model = current_model
        self.on_select = on_select

    def compose(self) -> ComposeResult:
        with Vertical(id="model_dialog"):
            yield Static("Select Model")
            yield Select(
                [(m, m) for m in self.models],
                value=self.current_model,
                allow_blank=False,
                id="model_select",
            )
            yield Horizontal(
                Button("Cancel", variant="error", id="cancel"),
                Button("Switch", variant="primary", id="switch"),
            )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel":
            self.app.pop_screen()
            return
        select = self.query_one("#model_select", Select)
        if select.value:
            self.on_select(str(select.value))
            self.app.pop_screen()


class TUI(App):
    CSS = """
    Screen {
        layout: vertical;
    }
    #chat {
        height: 1fr;
    }
    #input {
        dock: bottom;
        height: 3;
    }
    #status {
        dock: top;
        height: 1;
        content-align: center middle;
        background: $panel;
    }
    """

    BINDINGS = [
        ("m", "open_models", "Models"),
        ("r", "reload_tools", "Reload Tools"),
        ("ctrl+q", "quit", "Quit"),
    ]

    def __init__(self, agent: Agent, models: list[str] | None = None):
        super().__init__()
        self.agent = agent
        self.models = models or [agent.model]
        self.tool_count = len(agent.tool_manager.list_tools()) if agent.tool_manager else 0
        self._tools_loading = False

    def compose(self) -> ComposeResult:
        yield Static(id="status")
        yield Header()
        yield RichLog(id="chat", wrap=True, markup=True, auto_scroll=True)
        yield Input(placeholder="Send a message...", id="input")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#input").focus()
        self._update_status()
        self._initialize_tools()

    def _initialize_tools(self) -> None:
        if not self.agent.tool_manager:
            return
        if self.agent.tool_manager.list_tools():
            self.tool_count = len(self.agent.tool_manager.list_tools())
            self._update_status()
            return
        self._tools_loading = True
        self._update_status()
        try:
            coro = self.agent.tool_manager.initialize()
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(coro)
                self._tools_loading = False
                self.tool_count = len(self.agent.tool_manager.list_tools())
                self._update_status()
            else:
                loop.create_task(self._run_tool_init(coro))
        except Exception as e:
            self._tools_loading = False
            self._update_status()
            self.query_one("#chat", RichLog).write(
                Text(f"Failed to initialize tools: {e}", style="bold red")
            )

    async def _run_tool_init(self, coro):
        try:
            await coro
            self.tool_count = len(self.agent.tool_manager.list_tools())
            self._update_status()
        except Exception as e:
            self.query_one("#chat", RichLog).write(
                Text(f"Failed to initialize tools: {e}", style="bold red")
            )
        finally:
            self._tools_loading = False
            self._update_status()

    def _update_status(self) -> None:
        status = self.query_one("#status", Static)
        tools = self.tool_count
        loading = " (loading...)" if self._tools_loading else ""
        status.update(
            f"Model: {self.agent.model} | Tools loaded: {tools}{loading} | "
            "Press 'm' for models, 'r' to reload tools, 'ctrl+q' to quit"
        )

    def action_open_models(self) -> None:
        if not self.models:
            self.query_one("#chat", RichLog).write(
                Text("No models available", style="bold red")
            )
            return
        self.push_screen(
            ModelSelectScreen(
                models=self.models,
                current_model=self.agent.model,
                on_select=self._switch_model,
            )
        )

    def action_reload_tools(self) -> None:
        chat = self.query_one("#chat", RichLog)
        chat.write(Text("System", style="bold yellow"))
        if not self.agent.tool_manager:
            chat.write(Markdown("No tool manager configured."))
            return
        chat.write(Markdown("Reloading tools..."))
        self._initialize_tools()

    def _switch_model(self, model: str) -> None:
        self.agent.model = model
        chat = self.query_one("#chat", RichLog)
        chat.write(Text("System", style="bold yellow"))
        chat.write(Markdown(f"Switched model to: **{model}**"))
        self._update_status()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        content = event.value.strip()
        if not content:
            return
        self.query_one("#input").value = ""
        chat = self.query_one("#chat", RichLog)
        if self._tools_loading:
            chat.write(Text("System", style="bold yellow"))
            chat.write(Markdown("Tools are still loading, please wait..."))
            return
        chat.write(Text("You", style="bold cyan"))
        chat.write(Markdown(content))
        try:
            response = await self.agent.send_message(content)
        except Exception as e:
            response = f"Error: {e}"
        chat.write(Text("Assistant", style="bold green"))
        chat.write(Markdown(response))
