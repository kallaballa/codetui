import argparse
import asyncio

from openai import OpenAI

from .agent import Agent
from .history import default_history_path, default_prompt_history_path
from .tools import MCPToolManager
from .ui import TUI


def fetch_models(base_url: str, api_key: str) -> list[str]:
    client = OpenAI(base_url=base_url, api_key=api_key)
    try:
        models = client.models.list()
        return [model.id for model in models.data]
    except Exception as exc:
        if api_key and "401" in str(exc):
            raise SystemExit(
                f"Authentication failed for {base_url}. Check your API key."
            ) from exc
        return []


def main():
    parser = argparse.ArgumentParser(description="TUI Coding Agent")
    parser.add_argument("--base-url", default="http://localhost:11434/v1", help="OpenAI compatible API base URL")
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
    models = fetch_models(args.base_url, args.api_key)
    app = TUI(
        agent=agent,
        models=models,
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
