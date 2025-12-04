"""Helper utility functions."""

from typing import Any


def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text to maximum length with ellipsis.

    Args:
        text: Text to truncate
        max_length: Maximum length before truncation

    Returns:
        Truncated text with ellipsis if needed
    """
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def format_score(score: float) -> str:
    """Format similarity score for display.

    Args:
        score: Similarity score between 0 and 1

    Returns:
        Formatted score string
    """
    return f"{score:.4f}"


def safe_dict_get(dictionary: dict[str, Any], key: str, default: Any = None) -> Any:  # noqa: ANN401
    """Safely get value from dictionary with default.

    Args:
        dictionary: Dictionary to query
        key: Key to look up
        default: Default value if key not found

    Returns:
        Value from dictionary or default
    """
    return dictionary.get(key, default)
