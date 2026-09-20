# CodeTUI

A simple terminal UI coding agent that connects to any OpenAI-compatible API and extends it with tool use via MCP servers.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Textual](https://img.shields.io/badge/Textual-0.44%2B-purple)
![License](https://img.shields.io/badge/license-MIT-green)

## Features

- **Chat interface** — built with [Textual](https://textual.textualize.io/), optimized for terminal workflows.
- **OpenAI-compatible backends** — works with Ollama, OpenAI, OpenRouter, vLLM, LM Studio, and other chat-compatible APIs.
- **Tool calling via MCP** — load external tools from `mcp.json` and let the model call them automatically.
- **Streaming responses** — real-time token streaming with auto-follow chat behavior.
- **Conversation history** — persistent chat history and prompt history.
- **Model switching** — browse and switch between available models with `Ctrl+P`.
- **Local development ready** — async test suite with `pytest-asyncio`.

## Installation

```bash
git clone https://github.com/elchaschab/codetui.git
cd codetui
python -m venv .venv
source .venv/bin/activate
pip install .
```

### Development install

```bash
pip install -e ".[dev]"
```

## Usage

Start the TUI:

```bash
codetui
```

By default, it connects to `http://localhost:11434/v1` with model `llama3`. Override with CLI flags:

```bash
codetui \
  --base-url https://openrouter.ai/api/v1 \
  --api-key sk-... \
  --model mistralai/mistral-7b-instruct \
  --mcp-config ./mcp.json
```

### CLI arguments

| Argument | Default | Description |
| --- | --- | --- |
| `--base-url` | `http://localhost:11434/v1` | OpenAI-compatible API base URL |
| `--api-key` | `None` | API key |
| `--model` | `llama3` | Model name |
| `--mcp-config` | `None` | Path to MCP config JSON |
| `--history-path` | `~/.cache/codetui/history.json` | Conversation history file |
| `--prompt-history-path` | `~/.cache/codetui/prompt_history.json` | Prompt history file |
| `--no-history` | `False` | Disable loading and saving history |

## Configuration

### MCP tools

Create an `mcp.json` in your project root or at `~/.config/codetui/mcp.json`:

```json
{
  "mcpServers": {
    "ddg-search": {
      "command": "uvx",
      "args": ["duckduckgo-mcp-server"]
    }
  }
}
```

Environment variables in MCP configs support `${VAR_NAME}` interpolation.

Reload tools inside the app with `Ctrl+R`.

### Key bindings

| Key | Action |
| --- | --- |
| `Enter` | Send message |
| `Shift+Enter` | New line |
| `Ctrl+P` | Switch model |
| `Ctrl+N` | New conversation |
| `Ctrl+R` | Reload tools |
| `Ctrl+L` | Clear view |
| `Ctrl+End` | Jump to latest |
| `Esc` | Interrupt |
| `Ctrl+Q` | Quit |

## Development

```bash
# run tests
pytest

# run app from source
python -m codetui.main
```

## License

MIT
