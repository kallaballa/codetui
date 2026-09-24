import argparse
import asyncio
import logging

from .agent import Agent
from .history import default_history_path, default_prompt_history_path
from .tools import MCPToolManager
from .ui import TUI


def main():
    logging.basicConfig(level=logging.WARNING, format="[%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="TUI Coding Agent")
    parser.add_argument(
        "--base-url",
        default="http://localhost:11434/v1",
        help="OpenAI compatible API base URL",
    )
    parser.add_argument("--api-key", default=None, help="API key")
    parser.add_argument("--model", default="llama3", help="Default model name")
    parser.add_argument("--mcp-config", default=None, help="Path to MCP config JSON")
    parser.add_argument(
        "--history-path",
        default=str(default_history_path()),
        help="Where to persist conversation history",
    )
    parser.add_argument(
        "--no-history",
        action="store_true",
        help="Do not load or save conversation history",
    )
    parser.add_argument(
        "--prompt-history-path",
        default=str(default_prompt_history_path()),
        help="Where to persist prompt input history",
    )
    args = parser.parse_args()

    tool_manager = MCPToolManager(config_path=args.mcp_config)
    agent = Agent(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        tool_manager=tool_manager,
        history_path=None if args.no_history else args.history_path,
    )
    app = TUI(
        agent=agent,
        prompt_history_path=None if args.no_history else args.prompt_history_path,
    )
    try:
        app.run()
    finally:
        try:
            asyncio.run(tool_manager.shutdown())
        except Exception:
            pass


if __name__ == "__main__":
    main()
