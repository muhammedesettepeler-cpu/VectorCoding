from typing import Any

from loguru import logger
from qdrant_client import QdrantClient

from vector_sentiment.config.settings import QdrantSettings, get_settings
from vector_sentiment.utils.logger import setup_logging as configure_logging
from vector_sentiment.vectordb.client import QdrantClientWrapper


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration."""
    configure_logging(level=level)


def get_qdrant_client() -> tuple[QdrantClient, QdrantSettings]:
    """Get Qdrant client and settings.

    Returns:
        Tuple of (QdrantClient, QdrantSettings)
    """
    settings = get_settings()
    wrapper = QdrantClientWrapper(settings.qdrant)
    return wrapper.client, settings.qdrant


def print_results(results: list, title: str = "Results") -> None:
    """Print search/recommendation results.

    Args:
        results: List of result objects (must have score, label, text attributes)
        title: Title for the results section
    """
    print(f"\n{title}:")
    print("=" * 80)

    if not results:
        print("No results found.")
        return

    for i, result in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(f"  Score: {result.score:.4f}")
        print(f"  Label: {result.label}")
        if hasattr(result, "text") and result.text:
            # Truncate text if too long
            text = result.text
            if len(text) > 200:
                text = text[:197] + "..."
            print(f"  Text:  {text}")
        print("-" * 40)


def print_stats(title: str, **kwargs: Any) -> None:
    """Print statistics.

    Args:
        title: Title for the stats section
        **kwargs: Key-value pairs to print
    """
    print(f"\n{title}:")
    print("-" * 40)
    for key, value in kwargs.items():
        print(f"  {key}: {value}")
    print("-" * 40)
