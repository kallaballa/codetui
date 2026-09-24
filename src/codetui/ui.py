import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    Footer,
    Header,
    LoadingIndicator,
    Select,
    Static,
    TextArea,
)
from textual.worker import Worker

from .agent import Agent
from .history import load_prompt_history, save_prompt_history

ROLE_STYLES = {"You": "cyan", "Assistant": "green", "System": "yellow", "Error": "red"}

STREAM_UPDATE_INTERVAL = 0.05
SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
STREAMING_CURSOR = "▌"


class ChatScroll(VerticalScroll):
    """Scroll region that follows new output only while pinned to the bottom.

    Scrolling up detaches the view from the stream so the user can read earlier
    output; scrolling back to the bottom re-enables auto-follow.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.follow_output = True

    def watch_scroll_y(self, old_value: float, new_value: float) -> None:
        super().watch_scroll_y(old_value, new_value)
        self.follow_output = self.is_vertical_scroll_end

    def follow_tail(self) -> None:
        if self.follow_output:
            self.call_after_refresh(self.scroll_end, animate=False)

    def jump_to_end(self) -> None:
        self.follow_output = True
        self.call_after_refresh(self.scroll_end, animate=False)


class PromptInput(TextArea):
    """Multi-line prompt with history navigation and Enter-to-send."""

    class Submitted(Message):
        """Posted when the prompt should be sent."""

        def __init__(self, value: str) -> None:
            self.value = value
            super().__init__()

    NEWLINE_KEYS = {"shift+enter", "ctrl+j", "alt+enter"}

    def __init__(self, history_path: Path | str | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.history_path = Path(history_path) if history_path else None
        self._history = (
            load_prompt_history(self.history_path) if self.history_path else []
        )
        self._history_index: int | None = None
        self._draft = ""

    @property
    def value(self) -> str:
        return self.text

    @value.setter
    def value(self, value: str) -> None:
        self.text = value or ""
        self._history_index = None
        self._move_cursor_to_end()

    def _move_cursor_to_end(self) -> None:
        lines = self.text.split("\n")
        self.cursor_location = (len(lines) - 1, len(lines[-1]))

    def push_history(self, value: str) -> None:
        value = value.strip()
        if value and (not self._history or self._history[-1] != value):
            self._history.append(value)
            self._save_history()
        self._history_index = None

    def clear_history(self) -> None:
        self._history = []
        self._history_index = None
        self._draft = ""
        if self.history_path:
            try:
                self.history_path.unlink(missing_ok=True)
            except OSError:
                pass

    def _save_history(self) -> None:
        if self.history_path:
            save_prompt_history(self.history_path, self._history)

    async def _on_key(self, event: events.Key) -> None:
        if event.key == "enter":
            event.stop()
            event.prevent_default()
            self.post_message(self.Submitted(self.text))
            return
        if event.key in self.NEWLINE_KEYS:
            event.stop()
            event.prevent_default()
            self.insert("\n")
            return
        if self._history and self.selection.is_empty:
            row, _ = self.cursor_location
            if event.key == "up" and row == 0:
                event.stop()
                event.prevent_default()
                self._history_prev()
                return
            if event.key == "down" and row == self.document.line_count - 1:
                event.stop()
                event.prevent_default()
                self._history_next()
                return
        await super()._on_key(event)

    def _history_prev(self) -> None:
        if self._history_index is None:
            self._draft = self.text
            self._history_index = len(self._history) - 1
        elif self._history_index > 0:
            self._history_index -= 1
        else:
            return
        self._load_history_value(self._history[self._history_index])

    def _history_next(self) -> None:
        if self._history_index is None:
            return
        if self._history_index < len(self._history) - 1:
            self._history_index += 1
            self._load_history_value(self._history[self._history_index])
        else:
            self._history_index = None
            self._load_history_value(self._draft)

    def _load_history_value(self, value: str) -> None:
        self.text = value
        self._move_cursor_to_end()


class ToolCallView(Static):
    """A compact, clickable panel showing a single tool invocation."""

    COLLAPSED_LIMIT = 160
    EXPANDED_LIMIT = 4000
    TOOL_BORDER = "magenta"
    ERROR_BORDER = "#e5484d"
    OK_BORDER = "#30a46c"

    def __init__(self, name: str, arguments: object):
        super().__init__()
        self.tool_name = name
        self.argument_text = self._fmt(arguments)
        self.result_text: str | None = None
        self.expanded = False
        self.started_at = time.monotonic()
        self._refresh_panel()

    @staticmethod
    def _fmt(value: object) -> str:
        if isinstance(value, dict):
            try:
                return json.dumps(value, ensure_ascii=False)
            except (TypeError, ValueError):
                pass
        return str(value)

    @classmethod
    def _clip(cls, text: str, limit: int) -> str:
        text = " ".join(text.split())
        return text if len(text) <= limit else f"{text[:limit]}..."

    @staticmethod
    def _truncate(text: str, limit: int) -> str:
        return text if len(text) <= limit else f"{text[:limit]}..."

    def toggle_expand(self) -> None:
        self.expanded = not self.expanded
        self._refresh_panel()

    def set_result(self, result: str) -> None:
        self.result_text = str(result)
        self._refresh_panel()

    def _duration(self) -> str:
        elapsed = time.monotonic() - self.started_at
        return f"{elapsed:.1f}s" if elapsed >= 1.0 else f"{int(elapsed * 1000)}ms"

    def _is_error(self) -> bool:
        text = (self.result_text or "").strip().lower()
        return bool(text) and text.startswith(("error", "failed", "timeout"))

    def _border_style(self) -> str:
        if self._is_error():
            return self.ERROR_BORDER
        if self.result_text is not None:
            return self.OK_BORDER
        return self.TOOL_BORDER

    def _collapsed_line(self) -> Text:
        style = (
            "bold red"
            if self._is_error()
            else "bold green"
            if self.result_text is not None
            else "bold magenta"
        )
        line = Text()
        line.append("▸ ", style="dim")
        line.append(f"Tool: {self.tool_name}", style=style)
        if self.result_text is None:
            line.append(f"  ·  running {self._duration()}", style="dim italic")
            line.append("  (running...)", style="dim")
        else:
            line.append(f"  ·  {self._clip(self.result_text, 120)}", style="dim")
            line.append("  (click to expand)", style="dim")
        return line

    def _expanded_panel(self) -> Panel:
        border_style = self._border_style()
        body = Text(f"args: {self._truncate(self.argument_text, self.EXPANDED_LIMIT)}")
        if self.result_text is None:
            body.append(f"\nRunning for {self._duration()}...", style="dim italic")
        else:
            body.append(f"\nresult: {self._truncate(self.result_text, self.EXPANDED_LIMIT)}")
        return Panel(
            body,
            title=Text(f"Tool: {self.tool_name}", style="bold " + border_style),
            title_align="left",
            subtitle=Text("click to collapse", style="dim"),
            subtitle_align="right",
            border_style=border_style,
            padding=(0, 1),
        )

    def _refresh_panel(self) -> None:
        if self.expanded:
            self.update(self._expanded_panel())
        else:
            self.update(self._collapsed_line())

    def on_click(self, event: events.Click) -> None:
        self.toggle_expand()


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

    def __init__(
        self,
        models: list[str],
        current_model: str,
        on_select: Callable[[str], None],
    ):
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

    def on_mount(self) -> None:
        self.query_one("#switch", Button).focus()

    def _submit(self) -> None:
        select = self.query_one("#model_select", Select)
        if select.value:
            self.on_select(str(select.value))
        self.app.pop_screen()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel":
            self.app.pop_screen()
        else:
            self._submit()


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
    }
    #chat {
        height: 1fr;
        margin: 1 2 0 2;
        padding: 0 1;
        border: round #6b7280;
        background: $background;
        scrollbar-size: 1 1;
        scrollbar-color: $primary $surface;
        scrollbar-background: $surface;
    }
    #chat > Static {
        width: 1fr;
        height: auto;
        margin-bottom: 1;
    }
    #chat > LoadingIndicator {
        width: 1fr;
        height: 3;
        margin-bottom: 1;
        color: $text-muted;
        padding: 1 2;
    }
    #input_area {
        height: auto;
        margin: 1 2 0 2;
    }
    #input {
        height: auto;
        min-height: 3;
        max-height: 8;
        border: round #6b7280;
        background: $surface;
        padding: 0 1;
        scrollbar-size-vertical: 1;
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
        Binding("ctrl+n", "new_conversation", "New", priority=True),
        Binding("ctrl+r", "reload_tools", "Reload tools", priority=True),
        Binding("ctrl+l", "clear_chat", "Clear view", priority=True),
        Binding("ctrl+end", "jump_to_end", "Latest", priority=True),
        ("escape", "interrupt", "Interrupt"),
        ("ctrl+q", "quit", "Quit"),
    ]

    def __init__(
        self,
        agent: Agent,
        models: list[str] | None = None,
        prompt_history_path: Path | str | None = None,
    ):
        super().__init__()
        self.agent = agent
        self.models = list(dict.fromkeys([agent.model, *(models or [])]))
        self.prompt_history_path = (
            Path(prompt_history_path) if prompt_history_path else None
        )
        self.tool_count = len(agent.tool_manager.list_tools()) if agent.tool_manager else 0
        self._tools_loading = False
        self._sending = False
        self._interrupted = False
        self._active_tool: str | None = None
        self._tool_views: list[ToolCallView] = []
        self._response_worker: Worker[None] | None = None
        self._spinner_index = 0
        setter = getattr(self.agent, "set_listener", None)
        if callable(setter):
            setter(self._on_tool_event)

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static(id="status")
        yield ChatScroll(id="chat")
        with Vertical(id="input_area"):
            yield PromptInput(
                placeholder="Ask a question or describe a task... (Shift+Enter for a new line)",
                id="input",
                history_path=self.prompt_history_path,
            )
            yield Static(id="input_hint")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#input").focus()
        self.query_one("#chat", ChatScroll).border_title = "Conversation"
        self.sub_title = f"{self.agent.model}  ·  {Path.cwd().name}"
        self._load_conversation()
        self._update_status()
        self._initialize_tools()
        self._fetch_models()
        self.set_interval(0.1, self._tick_spinner)

    def _tick_spinner(self) -> None:
        if self._sending or self._tools_loading or self._active_tool:
            self._spinner_index = (self._spinner_index + 1) % len(SPINNER_FRAMES)
            try:
                if self._active_tool and self._tool_views:
                    self._tool_views[-1]._refresh_panel()
                self._update_status()
            except Exception:
                pass

    def _message_renderable(self, role: str, content: str):
        color = ROLE_STYLES.get(role, "white")
        body = Markdown(content) if role in ("You", "Assistant") else Text(content)
        border = f"bold {color}" if role in ("You", "Assistant") else color
        return Panel(
            body,
            title=Text(role, style=f"bold {color}"),
            title_align="left",
            subtitle=Text(datetime.now().strftime("%H:%M"), style="dim"),
            subtitle_align="right",
            border_style=border,
            padding=(1, 2),
        )

    def _write_message(self, role: str, content: str, *, force_scroll: bool = False) -> Static:
        chat = self.query_one("#chat", ChatScroll)
        widget = Static(self._message_renderable(role, content))
        chat.mount(widget)
        if force_scroll:
            chat.jump_to_end()
        else:
            chat.follow_tail()
        return widget

    def _show_thinking(self) -> None:
        chat = self.query_one("#chat", ChatScroll)
        if not chat.query("#thinking"):
            indicator = LoadingIndicator(id="thinking")
            chat.mount(indicator)
            chat.follow_tail()

    def _hide_thinking(self) -> None:
        chat = self.query_one("#chat", ChatScroll)
        matches = chat.query("#thinking")
        if matches:
            matches.remove()

    def _load_conversation(self) -> None:
        history = getattr(self.agent, "history", None) or []
        if not history:
            self._write_message(
                "System",
                "Welcome to CodeTUI\n\n"
                "Ask a question, explore your code, or describe what you want to build.\n"
                "Choose a model with Ctrl+P, then send your first message below.",
                force_scroll=True,
            )
            return
        self._write_message(
            "System",
            f"Resumed conversation ({len(history)} messages). "
            "Press Ctrl+N to start a new one.",
            force_scroll=True,
        )
        for message in history:
            role = "You" if message.get("role") == "user" else "Assistant"
            content = message.get("content")
            if isinstance(content, str):
                self._write_message(role, content, force_scroll=True)

    def action_clear_chat(self) -> None:
        self.query_one("#chat", ChatScroll).remove_children()
        self._tool_views = []
        self._write_message(
            "System", "View cleared. Conversation context is preserved.", force_scroll=True
        )

    def action_new_conversation(self) -> None:
        clear = getattr(self.agent, "clear_history", None)
        if callable(clear):
            clear()
        self.query_one("#input", PromptInput).clear_history()
        self.query_one("#chat", ChatScroll).remove_children()
        self._tool_views = []
        self._write_message("System", "Started a new conversation.", force_scroll=True)
        self._update_status()

    def action_jump_to_end(self) -> None:
        self.query_one("#chat", ChatScroll).jump_to_end()

    def action_interrupt(self) -> None:
        if not self._sending:
            return
        self._interrupted = True
        worker = self._response_worker
        if worker is not None and worker.is_running:
            worker.cancel()

    def _on_tool_event(self, event: dict[str, Any]) -> None:
        kind = event.get("type")
        name = event.get("name", "?")
        if kind == "tool_start":
            self._active_tool = name
            self._update_status()
            view = ToolCallView(name, event.get("arguments"))
            self._tool_views.append(view)
            chat = self.query_one("#chat", ChatScroll)
            chat.mount(view)
            chat.follow_tail()
        elif kind == "tool_end":
            result = event.get("result", "")
            for view in reversed(self._tool_views):
                if view.tool_name == name and view.result_text is None:
                    view.set_result(result)
                    break
            if self._active_tool == name:
                self._active_tool = None
            self._update_status()

    def _initialize_tools(self) -> None:
        manager = self.agent.tool_manager
        if manager is None:
            return
        if manager.list_tools():
            self.tool_count = len(manager.list_tools())
            self._update_status()
            return
        self._tools_loading = True
        self._update_status()
        self.run_worker(self._init_tools_worker(), name="init-tools", exclusive=True)

    def _fetch_models(self) -> None:
        models = getattr(getattr(self.agent, "client", None), "models", None)
        if not hasattr(models, "list"):
            return
        self.run_worker(self._fetch_models_worker(), name="fetch-models")

    async def _fetch_models_worker(self) -> None:
        try:
            result = await self.agent.client.models.list()
        except Exception as exc:
            self._write_message(
                "System", f"Could not fetch models: {exc}", force_scroll=True
            )
            return
        ids: list[str] = []
        for model in result:
            model_id = getattr(model, "id", None)
            if isinstance(model_id, str):
                ids.append(model_id)
        if ids:
            self.models = list(dict.fromkeys([self.agent.model, *ids]))

    async def _init_tools_worker(self) -> None:
        manager = self.agent.tool_manager
        if manager is None:
            return
        try:
            await manager.initialize()
            self.tool_count = len(manager.list_tools())
        except Exception as e:
            self._write_message("Error", f"Failed to initialize tools: {e}", force_scroll=True)
        finally:
            self._tools_loading = False
            self._update_status()

    def _update_status(self) -> None:
        busy = self._sending or self._tools_loading or bool(self._active_tool)
        spinner = SPINNER_FRAMES[self._spinner_index] if busy else ""
        if self._active_tool:
            state = f"{spinner} Running: {self._active_tool}..."
        elif self._sending:
            state = f"{spinner} Thinking..."
        else:
            state = "Ready"
        if self._tools_loading:
            tools = "Loading tools..."
        else:
            tools = f"{self.tool_count} tools"
        count = len(getattr(self.agent, "history", None) or [])
        status = Text()
        status.append(self.agent.model, style="bold cyan")
        status.append("  ·  ")
        status.append(tools, style="yellow" if self._tools_loading else "green")
        status.append("  ·  ")
        status.append(f"{count} msgs", style="yellow")
        status.append("  ·  ")
        status.append(state, style="bold green" if state == "Ready" else "bold white")
        self.query_one("#status", Static).update(status)
        if self._sending:
            hint = "Esc to interrupt  ·  Ctrl+End latest  ·  scroll up to pause auto-follow"
        else:
            hint = (
                "Enter send  ·  Shift+Enter newline  ·  ↑/↓ history  ·  "
                "Ctrl+P model  ·  Ctrl+N new  ·  Ctrl+L clear view"
            )
        self.query_one("#input_hint", Static).update(Text(hint, style="dim italic"))

    def action_open_models(self) -> None:
        if not self.models:
            self._write_message("System", "No models available.", force_scroll=True)
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
            self._write_message("System", "No tool manager configured.", force_scroll=True)
            return
        self._write_message("System", "Reloading tools...", force_scroll=True)
        self._initialize_tools()

    def _switch_model(self, model: str) -> None:
        self.agent.model = model
        self.sub_title = f"{self.agent.model}  ·  {Path.cwd().name}"
        self._write_message("System", f"Switched model to: {model}", force_scroll=True)
        self._update_status()

    def on_prompt_input_submitted(self, event: PromptInput.Submitted) -> None:
        content = event.value.strip()
        if not content:
            return
        if self._tools_loading:
            self._write_message(
                "System", "Tools are still loading, please wait...", force_scroll=True
            )
            return
        if self._sending:
            self._write_message(
                "System", "Please wait for the current response.", force_scroll=True
            )
            return
        input_widget = self.query_one("#input", PromptInput)
        input_widget.value = ""
        input_widget.push_history(content)
        self._write_message("You", content, force_scroll=True)
        self._sending = True
        self._interrupted = False
        self._response_worker = self.run_worker(
            self._respond(content), name="respond", group="respond", exclusive=True
        )
        self._update_status()

    async def _respond(self, content: str) -> None:
        try:
            stream = getattr(self.agent, "stream_message", None)
            if stream is None:
                self._show_thinking()
                try:
                    response = await self.agent.send_message(content)
                finally:
                    self._hide_thinking()
                self._write_message("Assistant", response, force_scroll=True)
                return
            chat = self.query_one("#chat", ChatScroll)
            self._show_thinking()
            widget: Static | None = None
            buffer = ""
            last_refresh = 0.0
            last_rendered_len = 0
            first_token = True
            agen = stream(content)
            try:
                async for token in agen:
                    if first_token:
                        first_token = False
                        self._hide_thinking()
                        widget = Static(self._message_renderable("Assistant", ""))
                        chat.mount(widget)
                        chat.follow_tail()
                    buffer += token
                    now = time.monotonic()
                    if widget is not None and (
                        now - last_refresh >= STREAM_UPDATE_INTERVAL
                        or len(buffer) - last_rendered_len >= 512
                    ):
                        last_refresh = now
                        last_rendered_len = len(buffer)
                        widget.update(
                            self._message_renderable("Assistant", buffer + STREAMING_CURSOR)
                        )
                        chat.follow_tail()
            except asyncio.CancelledError:
                self._interrupted = True
            finally:
                self._hide_thinking()
                if self._interrupted:
                    try:
                        await agen.aclose()
                    except BaseException:
                        pass
                    buffer = f"{buffer}\n\n*[interrupted]*" if buffer else "*[interrupted]*"
                elif not buffer:
                    buffer = "(no response)"
                try:
                    if widget is None:
                        widget = Static(self._message_renderable("Assistant", buffer))
                        chat.mount(widget)
                    else:
                        widget.update(self._message_renderable("Assistant", buffer))
                    chat.follow_tail()
                except Exception:
                    pass
        except asyncio.CancelledError:
            self._interrupted = True
            raise
        except Exception as exc:
            self._hide_thinking()
            try:
                self._write_message(
                    "Error", f"Request failed: {exc}", force_scroll=True
                )
            except Exception:
                pass
        finally:
            self._sending = False
            try:
                self._update_status()
            except Exception:
                pass
            try:
                self.query_one("#input", PromptInput).focus()
            except Exception:
                pass
