import asyncio
import argparse
from .agent import Agent
from .tools import MCPToolManager
from .ui import TUI
from openai import OpenAI


def fetch_models(base_url: str, api_key: str) -> list[str]:
    client = OpenAI(base_url=base_url, api_key=api_key)
    try:
        models = client.models.list()
        return [model.id for model in models.data]
    except Exception:
        return []


def main():
    parser = argparse.ArgumentParser(description="TUI Coding Agent")
    parser.add_argument("--base-url", default="http://localhost:11434/v1", help="OpenAI compatible API base URL")
    parser.add_argument("--api-key", default="sk-dummy", help="API key")
    parser.add_argument("--model", default="llama3", help="Default model name")
    parser.add_argument("--mcp-config", default=None, help="Path to MCP config JSON")
    args = parser.parse_args()

    tool_manager = MCPToolManager(config_path=args.mcp_config)
    agent = Agent(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        tool_manager=tool_manager,
    )
    models = fetch_models(args.base_url, args.api_key)
    app = TUI(agent=agent, models=models)
    try:
        app.run()
    finally:
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(tool_manager.shutdown())
        except Exception:
            pass


if __name__ == "__main__":
    main()
