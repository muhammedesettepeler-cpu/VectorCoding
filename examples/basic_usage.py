"""Basic usage example for Vector Sentiment Search.

This script demonstrates how to use the library programmatically for
loading data, generating embeddings, and performing searches.
"""

from pathlib import Path

from loguru import logger

from vector_sentiment.config.settings import get_settings
from vector_sentiment.data.loader import ParquetDataLoader
from vector_sentiment.data.preprocessor import TextPreprocessor
from vector_sentiment.embeddings.service import EmbeddingService
from vector_sentiment.search.searcher import SearchService
from vector_sentiment.utils.logger import setup_logging
from vector_sentiment.vectordb.client import QdrantClientWrapper
from vector_sentiment.vectordb.collection import CollectionManager
from vector_sentiment.vectordb.upserter import VectorUpserter


def main() -> None:
    """Run basic usage example."""
    # Setup logging
    setup_logging(level="INFO")
    logger.info("Starting basic usage example")

    # Load settings
    settings = get_settings()

    # Initialize embedding service
    logger.info("Loading embedding model...")
    embedding_service = EmbeddingService(
        model_name=settings.embedding.model_name,
        batch_size=settings.embedding.batch_size,
    )

    # Initialize Qdrant client
    logger.info("Connecting to Qdrant...")
    qdrant_wrapper = QdrantClientWrapper(settings.qdrant)

    if not qdrant_wrapper.health_check():
        logger.error("Qdrant is not healthy")
        return

    client = qdrant_wrapper.client
    collection_name = settings.collection.name
    vector_name = settings.embedding.model_name

    # Create collection if it doesn't exist
    collection_manager = CollectionManager(client)

    if not qdrant_wrapper.collection_exists(collection_name):
        logger.info(f"Creating collection '{collection_name}'...")
        collection_manager.create_collection(
            collection_name=collection_name,
            vector_size=settings.embedding.vector_size,
            vector_name=vector_name,
            distance=settings.collection.distance_metric,
        )
    else:
        logger.info(f"Collection '{collection_name}' already exists")

    # Example: Insert some sample data
    logger.info("Inserting sample data...")
    sample_texts = [
        "This product is amazing! I love it.",
        "Terrible experience, would not recommend.",
        "It's okay, nothing special.",
    ]
    sample_labels = ["positive", "negative", "neutral"]

    # Preprocess
    preprocessor = TextPreprocessor()
    processed_texts = preprocessor.preprocess_batch(sample_texts)

    # Generate embeddings
    embeddings = embedding_service.encode(processed_texts)

    # Upload to Qdrant
    upserter = VectorUpserter(client, collection_name)
    payloads = [
        {"label": label, "text": text}
        for label, text in zip(sample_labels, sample_texts, strict=True)
    ]

    upserter.upsert_batch(
        vectors=embeddings,
        payloads=payloads,
        vector_name=vector_name,
        start_id=0,
    )

    logger.info("Sample data inserted successfully")

    # Example: Search
    logger.info("Performing search...")
    search_service = SearchService(
        client=client,
        collection_name=collection_name,
        embedding_service=embedding_service,
        vector_name=vector_name,
    )

    query = "Great quality product"
    results = search_service.search(query_text=query, limit=3)

    logger.info(f"\nSearch results for: '{query}'")
    for i, result in enumerate(results, 1):
        logger.info(f"{i}. Score: {result.score:.4f}, Label: {result.label}")
        if result.text:
            logger.info(f"   Text: {result.text}")

    # Cleanup
    qdrant_wrapper.close()
    logger.info("Example completed")


if __name__ == "__main__":
    main()
