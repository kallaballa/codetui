import json
from pathlib import Path

VALID_ROLES = {"user", "assistant", "tool", "system"}


def default_history_path() -> Path:
    return Path.home() / ".cache" / "codetui" / "history.json"


def default_prompt_history_path() -> Path:
    return Path.home() / ".cache" / "codetui" / "prompt_history.json"


def _sanitize(messages: list[dict]) -> list[dict]:
    cleaned = []
    for message in messages or []:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        if role not in VALID_ROLES or not isinstance(content, str):
            continue
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