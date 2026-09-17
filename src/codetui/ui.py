import asyncio
from datetime import datetime
from rich.panel import Panel
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Header, Footer, Input, RichLog, Static, Button, Select
from rich.text import Text
from rich.markdown import Markdown
from .agent import Agent


class ModelSelectScreen(ModalScreen):
    CSS = """
    ModelSelectScreen {
        align: center middle;
        background: $background 70%;
    }
    #model_dialog {
        width: 64;
        max-width: 100%;
        height: auto;
        max-height: 90%;
        border: round $primary;
        background: $surface;
        padding: 1 2;
    }
    #model_title {
        text-style: bold;
        color: $text;
        margin-bottom: 1;
    }
    #model_hint {
        color: $text-muted;
        margin-bottom: 1;
    }
    #model_select {
        width: 1fr;
        margin-bottom: 1;
    }
    #model_actions {
        height: auto;
        align-horizontal: right;
    }
    #model_actions Button {
        margin-left: 1;
    }
    """

    BINDINGS = [("escape", "dismiss", "Cancel")]

    def __init__(self, models: list[str], current_model: str, on_select):
        super().__init__()
        self.models = models
        self.current_model = current_model
        self.on_select = on_select

    def compose(self) -> ComposeResult:
        with Vertical(id="model_dialog"):
            yield Static("Select a model", id="model_title")
            yield Static("Choose the model for your conversation.", id="model_hint")
            yield Select(
                [(m, m) for m in self.models],
                value=self.current_model,
                allow_blank=False,
                id="model_select",
            )
            yield Horizontal(
                Button("Cancel", id="cancel"),
                Button("Switch", variant="primary", id="switch"),
                id="model_actions",
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
    TITLE = "CodeTUI"
    SUB_TITLE = "Your coding companion"

    CSS = """
    Screen {
        layout: vertical;
        background: $background;
    }
    Header {
        background: $surface;
    }
    #status {
        height: 1;
        padding: 0 2;
        background: $surface;
        color: $text-muted;
    }
    #chat {
        height: 1fr;
        margin: 1 2 0 2;
        padding: 0 1;
        border: round $panel;
        background: $background;
        scrollbar-size: 1 1;
    }
    #input_area {
        height: auto;
        margin: 1 2 0 2;
    }
    #input {
        height: 3;
        border: round $primary;
        background: $surface;
        padding: 0 1;
    }
    #input:focus {
        border: round $accent;
    }
    #input_hint {
        height: 1;
        color: $text-muted;
        padding: 0 1;
        margin-bottom: 1;
    }
    Footer {
        background: $surface;
    }
    """

    BINDINGS = [
        Binding("ctrl+p", "open_models", "Models", priority=True),
        Binding("ctrl+r", "reload_tools", "Reload tools", priority=True),
        Binding("ctrl+l", "clear_chat", "Clear view", priority=True),
        ("ctrl+q", "quit", "Quit"),
    ]

    def __init__(self, agent: Agent, models: list[str] | None = None):
        super().__init__()
        self.agent = agent
        self.models = list(dict.fromkeys([agent.model, *(models or [])]))
        self.tool_count = len(agent.tool_manager.list_tools()) if agent.tool_manager else 0
        self._tools_loading = False
        self._sending = False

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static(id="status")
        yield RichLog(id="chat", wrap=True, min_width=1, auto_scroll=True)
        with Vertical(id="input_area"):
            yield Input(placeholder="Ask a question or describe a task...", id="input")
            yield Static("Enter to send  ·  Ctrl+P to choose a model", id="input_hint")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#input").focus()
        self.query_one("#chat", RichLog).border_title = "Conversation"
        self._show_welcome()
        self._update_status()
        self._initialize_tools()

    def _write_message(self, role: str, content: str) -> None:
        colors = {"You": "cyan", "Assistant": "green", "System": "yellow", "Error": "red"}
        color = colors.get(role, "white")
        chat = self.query_one("#chat", RichLog)
        chat.write(Panel(
            Markdown(content) if role in ("You", "Assistant") else Text(content),
            title=Text(role, style=f"bold {color}"),
            title_align="left",
            subtitle=Text(datetime.now().strftime("%H:%M"), style="dim"),
            subtitle_align="right",
            border_style=color,
            padding=(1, 2),
        ))
        chat.write("")

    def _show_welcome(self) -> None:
        self._write_message(
            "System",
            "Welcome to CodeTUI\n\n"
            "Ask a question, explore your code, or describe what you want to build.\n"
            "Choose a model with Ctrl+P, then send your first message below.",
        )

    def action_clear_chat(self) -> None:
        self.query_one("#chat", RichLog).clear()
        self._write_message("System", "View cleared. Conversation context is preserved.")

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
            self._write_message("Error", f"Failed to initialize tools: {e}")

    async def _run_tool_init(self, coro):
        try:
            await coro
            self.tool_count = len(self.agent.tool_manager.list_tools())
            self._update_status()
        except Exception as e:
            self._write_message("Error", f"Failed to initialize tools: {e}")
        finally:
            self._tools_loading = False
            self._update_status()

    def _update_status(self) -> None:
        status = self.query_one("#status", Static)
        state = "Thinking..." if self._sending else "Ready"
        tools = "Loading tools..." if self._tools_loading else f"{self.tool_count} tools"
        status.update(Text(f"{self.agent.model}  ·  {tools}  ·  {state}"))
        self.query_one("#input_hint", Static).update(
            "Waiting for a response..." if self._sending
            else "Enter to send  ·  Ctrl+P to choose a model"
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
        if not self.agent.tool_manager:
            self._write_message("System", "No tool manager configured.")
            return
        self._write_message("System", "Reloading tools...")
        self._initialize_tools()

    def _switch_model(self, model: str) -> None:
        self.agent.model = model
        self._write_message("System", f"Switched model to: {model}")
        self._update_status()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        content = event.value.strip()
        if not content:
            return
        if self._tools_loading:
            self._write_message("System", "Tools are still loading, please wait...")
            return
        self.query_one("#input", Input).value = ""
        self._write_message("You", content)
        self._sending = True
        self._update_status()
        try:
            response = await self.agent.send_message(content)
        except Exception as e:
            self._write_message("Error", str(e))
        else:
            self._write_message("Assistant", response)
        finally:
            self._sending = False
            self._update_status()
