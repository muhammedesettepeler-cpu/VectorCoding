"""Helper utility functions."""

from typing import Any


def truncate_text(text: str, max_length: int = 100) -> str:
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def format_score(score: float) -> str:
    return f"{score:.4f}"


def safe_dict_get(dictionary: dict[str, Any], key: str, default: Any = None) -> Any:  # noqa: ANN401
    return dictionary.get(key, default)
