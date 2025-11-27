"""Advanced search examples demonstrating filters and recommendations.

This script shows advanced usage patterns including:
- Filtered searches
- Score thresholds
- Recommendation with positive/negative examples
"""

from loguru import logger

from vector_sentiment.config.settings import get_settings
from vector_sentiment.embeddings.service import EmbeddingService
from vector_sentiment.search.recommender import RecommendationService
from vector_sentiment.search.searcher import SearchService
from vector_sentiment.utils.logger import setup_logging
from vector_sentiment.vectordb.client import QdrantClientWrapper


def demo_filtered_search() -> None:
    """Demonstrate search with label filters."""
    logger.info("=== Filtered Search Demo ===")

    settings = get_settings()
    embedding_service = EmbeddingService(settings.embedding.model_name)

    with QdrantClientWrapper(settings.qdrant) as qdrant:
        search_service = SearchService(
            client=qdrant.client,
            collection_name=settings.collection.name,
            embedding_service=embedding_service,
            vector_name=settings.embedding.model_name,
        )

        query = "Excellent product"

        # Search only in positive reviews
        logger.info(f"\nSearching for '{query}' in POSITIVE reviews:")
        results = search_service.search(
            query_text=query,
            filter_label="positive",
            limit=5,
        )

        for i, result in enumerate(results, 1):
            logger.info(f"{i}. Score: {result.score:.4f} - {result.label}")


def demo_score_threshold() -> None:
    """Demonstrate search with score threshold."""
    logger.info("\n=== Score Threshold Demo ===")

    settings = get_settings()
    embedding_service = EmbeddingService(settings.embedding.model_name)

    with QdrantClientWrapper(settings.qdrant) as qdrant:
        search_service = SearchService(
            client=qdrant.client,
            collection_name=settings.collection.name,
            embedding_service=embedding_service,
            vector_name=settings.embedding.model_name,
        )

        query = "Good quality"
        threshold = 0.8

        logger.info(f"\nSearching for '{query}' with threshold >= {threshold}:")
        results = search_service.search(
            query_text=query,
            score_threshold=threshold,
            limit=10,
        )

        logger.info(f"Found {len(results)} results with score >= {threshold}")
        for result in results:
            logger.info(f"Score: {result.score:.4f} - {result.label}")


def demo_recommendations() -> None:
    """Demonstrate recommendation with positive/negative examples."""
    logger.info("\n=== Recommendations Demo ===")

    settings = get_settings()

    with QdrantClientWrapper(settings.qdrant) as qdrant:
        recommend_service = RecommendationService(
            client=qdrant.client,
            collection_name=settings.collection.name,
            vector_name=settings.embedding.model_name,
        )

        # Recommend based on labels
        logger.info("\nRecommending based on labels:")
        logger.info("Positive examples: positive reviews")
        logger.info("Negative examples: negative reviews")

        results = recommend_service.recommend_by_label(
            positive_label="positive",
            negative_label="negative",
            limit=5,
        )

        logger.info(f"\nFound {len(results)} recommendations:")
        for i, result in enumerate(results, 1):
            logger.info(f"{i}. Score: {result.score:.4f} - Label: {result.label}")
            if result.text:
                text_preview = result.text[:80] + "..." if len(result.text) > 80 else result.text
                logger.info(f"   {text_preview}")


def main() -> None:
    """Run all advanced search examples."""
    setup_logging(level="INFO")

    try:
        demo_filtered_search()
        demo_score_threshold()
        demo_recommendations()

        logger.info("\n✓ All examples completed successfully")

    except Exception as e:
        logger.error(f"Error running examples: {e}")
        raise


if __name__ == "__main__":
    main()
