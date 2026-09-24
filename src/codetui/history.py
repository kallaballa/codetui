import json
from pathlib import Path

VALID_ROLES = {"user", "assistant", "tool", "system"}


def default_history_path() -> Path:
    return Path.home() / ".cache" / "codetui" / "history.json"


def default_prompt_history_path() -> Path:
    return Path.home() / ".cache" / "codetui" / "prompt_history.json"


def _sanitize(messages: list[dict]) -> list[dict]:
    cleaned: list[dict] = []
    tool_ids: set[str] = set()
    pending_ids: list[str] = []
    for message in messages or []:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role not in VALID_ROLES:
            continue
        if role == "assistant":
            entry: dict = {"role": role}
            content = message.get("content")
            if isinstance(content, str):
                entry["content"] = content
            tool_calls = message.get("tool_calls")
            if isinstance(tool_calls, list):
                valid = []
                for tc in tool_calls:
                    if not isinstance(tc, dict):
                        continue
                    fn = tc.get("function")
                    if not (isinstance(tc.get("id"), str) and isinstance(fn, dict)):
                        continue
                    valid.append(
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": fn.get("name", ""),
                                "arguments": fn.get("arguments", ""),
                            },
                        }
                    )
                    tool_ids.add(tc["id"])
                    pending_ids.append(tc["id"])
                if valid:
                    entry["tool_calls"] = valid
            if "content" not in entry and "tool_calls" not in entry:
                continue
            cleaned.append(entry)
        elif role == "tool":
            content = message.get("content")
            if not isinstance(content, str):
                continue
            tool_call_id = message.get("tool_call_id")
            if isinstance(tool_call_id, str) and tool_call_id in tool_ids:
                if tool_call_id in pending_ids:
                    pending_ids.remove(tool_call_id)
                cleaned.append(
                    {"role": "tool", "tool_call_id": tool_call_id, "content": content}
                )
            elif not isinstance(tool_call_id, str) and pending_ids:
                tool_call_id = pending_ids.pop(0)
                cleaned.append(
                    {"role": "tool", "tool_call_id": tool_call_id, "content": content}
                )
        else:
            content = message.get("content")
            if isinstance(content, str):
                cleaned.append({"role": role, "content": content})
    return cleaned


def load_history(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(data, list):
        return []
    return _sanitize(data)


def save_history(path: Path, messages: list[dict]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_sanitize(messages), indent=2), encoding="utf-8")
    except OSError:
        pass


def load_prompt_history(path: Path) -> list[str]:
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(data, list):
        return []
    cleaned: list[str] = []
    for item in data:
        if isinstance(item, str):
            value = item.strip()
            if value and (not cleaned or cleaned[-1] != value):
                cleaned.append(value)
    return cleaned


def save_prompt_history(path: Path, prompts: list[str]) -> None:
    cleaned: list[str] = []
    for item in prompts or []:
        if not isinstance(item, str):
            continue
        value = item.strip()
        if not value or (cleaned and cleaned[-1] == value):
            continue
        cleaned.append(value)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cleaned, indent=2), encoding="utf-8")
    except OSError:
        pass