"""CLI commands for vector sentiment search operations."""

import sys
from pathlib import Path

import click
from loguru import logger

from vector_sentiment.config.settings import get_settings
from vector_sentiment.data.loader import ParquetDataLoader
from vector_sentiment.data.preprocessor import TextPreprocessor
from vector_sentiment.embeddings.service import EmbeddingService
from vector_sentiment.search.recommender import RecommendationService
from vector_sentiment.search.searcher import SearchService
from vector_sentiment.utils.logger import setup_logging
from vector_sentiment.vectordb.client import QdrantClientWrapper
from vector_sentiment.vectordb.collection import CollectionManager
from vector_sentiment.vectordb.upserter import VectorUpserter


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def cli(verbose: bool) -> None:
    """Vector Sentiment Search CLI.

    A command-line interface for managing sentiment vector embeddings
    and performing similarity search operations with Qdrant.
    """
    settings = get_settings()

    log_level = "DEBUG" if verbose else settings.logging.level
    setup_logging(
        level=log_level,
        log_file=settings.logging.file_path,
        rotation=settings.logging.rotation,
        retention=settings.logging.retention,
    )


@cli.command()
@click.option(
    "--data-path",
    type=click.Path(exists=True, path_type=Path),
    help="Path to parquet data file",
)
@click.option(
    "--batch-size",
    type=int,
    default=128,
    help="Batch size for processing",
)
@click.option(
    "--recreate",
    is_flag=True,
    help="Recreate collection if it exists",
)
def ingest(
    data_path: Path | None,
    batch_size: int,
    recreate: bool,
) -> None:
    """Ingest data into Qdrant vector database.

    This command loads sentiment data from a parquet file, generates embeddings,
    and uploads them to Qdrant with metadata.

    Example:
        vector-sentiment ingest --data-path data/sentiment.parquet --batch-size 128
    """
    settings = get_settings()

    # Use settings path if not provided
    if data_path is None:
        data_path = settings.data.parquet_path

    logger.info(f"Starting data ingestion from {data_path}")

    # Initialize services
    logger.info("Initializing services...")
    embedding_service = EmbeddingService(
        model_name=settings.embedding.model_name,
        batch_size=settings.embedding.batch_size,
        normalize=settings.embedding.normalize,
    )

    with QdrantClientWrapper(settings.qdrant) as qdrant_wrapper:
        # Health check
        if not qdrant_wrapper.health_check():
            logger.error("Qdrant health check failed")
            sys.exit(1)

        client = qdrant_wrapper.client
        collection_manager = CollectionManager(client)

        # Create or recreate collection
        collection_name = settings.collection.name
        vector_name = settings.embedding.model_name

        if recreate:
            logger.info("Recreating collection...")
            collection_manager.recreate_collection(
                collection_name=collection_name,
                vector_size=settings.embedding.vector_size,
                vector_name=vector_name,
                distance=settings.collection.distance_metric,
            )
        elif not qdrant_wrapper.collection_exists(collection_name):
            logger.info("Creating new collection...")
            collection_manager.create_collection(
                collection_name=collection_name,
                vector_size=settings.embedding.vector_size,
                vector_name=vector_name,
                distance=settings.collection.distance_metric,
            )
        else:
            logger.info(f"Using existing collection '{collection_name}'")

        # Create payload index on label field
        try:
            collection_manager.create_payload_index(
                collection_name=collection_name,
                field_name="label",
                field_schema="keyword",
            )
        except Exception as e:
            logger.warning(f"Could not create payload index (may already exist): {e}")

        # Initialize preprocessor and upserter
        preprocessor = TextPreprocessor(
            lowercase=settings.data.lowercase,
            remove_stopwords=settings.data.remove_stopwords,
            remove_punctuation=settings.data.remove_punctuation,
        )

        upserter = VectorUpserter(client, collection_name)

        # Load and process data
        logger.info(f"Loading data from {data_path}")
        with ParquetDataLoader(data_path, batch_size=batch_size) as loader:
            total_rows = loader.get_total_rows()
            logger.info(f"Total rows to process: {total_rows}")

            current_id = 0
            processed = 0

            for batch_df in loader.iter_batches():
                # Extract text and labels
                text_col = "text" if "text" in batch_df.columns else "sentence"
                texts = batch_df[text_col].tolist()
                labels = batch_df["label"].tolist()

                # Preprocess texts
                processed_texts = preprocessor.preprocess_batch(texts)

                # Generate embeddings
                embeddings = embedding_service.encode(processed_texts, batch_size=batch_size)

                # Create payloads
                payloads = [
                    {"label": str(label).lower(), "text": str(text)}
                    for text, label in zip(texts, labels, strict=True)
                ]

                # Upload to Qdrant
                count = upserter.upsert_batch(
                    vectors=embeddings,
                    payloads=payloads,
                    vector_name=vector_name,
                    start_id=current_id,
                )

                current_id += count
                processed += count

                logger.info(
                    f"Progress: {processed}/{total_rows} ({processed / total_rows * 100:.1f}%)"
                )

            logger.info(f"Ingestion complete! Total vectors uploaded: {processed}")


@cli.command()
@click.argument("query")
@click.option("--label", type=str, help="Filter by label (positive/negative/neutral)")
@click.option("--score-threshold", type=float, help="Minimum score threshold (0.0-1.0)")
@click.option("--limit", type=int, default=10, help="Maximum number of results")
def search(
    query: str,
    label: str | None,
    score_threshold: float | None,
    limit: int,
) -> None:
    """Search for similar vectors using text query.

    Example:
        vector-sentiment search "Great product quality" --label positive --limit 5
    """
    settings = get_settings()

    logger.info(f"Searching for: {query}")

    # Initialize services
    embedding_service = EmbeddingService(
        model_name=settings.embedding.model_name,
    )

    with QdrantClientWrapper(settings.qdrant) as qdrant_wrapper:
        if not qdrant_wrapper.health_check():
            logger.error("Qdrant health check failed")
            sys.exit(1)

        search_service = SearchService(
            client=qdrant_wrapper.client,
            collection_name=settings.collection.name,
            embedding_service=embedding_service,
            vector_name=settings.embedding.model_name,
        )

        # Perform search
        results = search_service.search(
            query_text=query,
            filter_label=label,
            score_threshold=score_threshold,
            limit=limit,
        )

        # Display results
        click.echo(f"\nFound {len(results)} results:\n")
        for i, result in enumerate(results, 1):
            click.echo(f"{i}. [Score: {result.score:.4f}] Label: {result.label}")
            if result.text:
                text_preview = result.text[:100] + "..." if len(result.text) > 100 else result.text
                click.echo(f"   Text: {text_preview}")
            click.echo()


@cli.command()
@click.option("--positive-ids", type=str, help="Comma-separated positive example IDs")
@click.option("--negative-ids", type=str, help="Comma-separated negative example IDs")
@click.option("--positive-label", type=str, help="Use examples with this label as positive")
@click.option("--negative-label", type=str, help="Use examples with this label as negative")
@click.option("--limit", type=int, default=10, help="Maximum number of results")
def recommend(
    positive_ids: str | None,
    negative_ids: str | None,
    positive_label: str | None,
    negative_label: str | None,
    limit: int,
) -> None:
    """Get recommendations based on positive/negative examples.

    Example:
        vector-sentiment recommend --positive-ids "1,2,3" --negative-ids "10,11" --limit 5
        vector-sentiment recommend --positive-label positive --negative-label negative
    """
    settings = get_settings()

    with QdrantClientWrapper(settings.qdrant) as qdrant_wrapper:
        if not qdrant_wrapper.health_check():
            logger.error("Qdrant health check failed")
            sys.exit(1)

        recommend_service = RecommendationService(
            client=qdrant_wrapper.client,
            collection_name=settings.collection.name,
            vector_name=settings.embedding.model_name,
        )

        # Get recommendations
        if positive_label:
            # Use label-based recommendation
            results = recommend_service.recommend_by_label(
                positive_label=positive_label,
                negative_label=negative_label,
                limit=limit,
            )
        elif positive_ids:
            # Use ID-based recommendation
            pos_ids = [int(x.strip()) for x in positive_ids.split(",")]
            neg_ids = [int(x.strip()) for x in negative_ids.split(",")] if negative_ids else None

            results = recommend_service.recommend(
                positive_ids=pos_ids,
                negative_ids=neg_ids,
                limit=limit,
            )
        else:
            click.echo("Error: Must provide either --positive-ids or --positive-label")
            sys.exit(1)

        # Display results
        click.echo(f"\nFound {len(results)} recommendations:\n")
        for i, result in enumerate(results, 1):
            click.echo(f"{i}. [Score: {result.score:.4f}] Label: {result.label}")
            if result.text:
                text_preview = result.text[:100] + "..." if len(result.text) > 100 else result.text
                click.echo(f"   Text: {text_preview}")
            click.echo()


@cli.command()
def status() -> None:
    """Check Qdrant connection and collection status.

    Example:
        vector-sentiment status
    """
    settings = get_settings()

    click.echo("Checking Qdrant connection...")

    with QdrantClientWrapper(settings.qdrant) as qdrant_wrapper:
        if qdrant_wrapper.health_check():
            click.echo("✓ Qdrant is healthy\n")

            # Get collection info
            collection_name = settings.collection.name
            if qdrant_wrapper.collection_exists(collection_name):
                info = qdrant_wrapper.get_collection_info(collection_name)
                if info:
                    click.echo(f"Collection: {collection_name}")
                    click.echo(f"Points count: {info.points_count}")
                    click.echo(f"Vectors config: {info.config.params.vectors}")
            else:
                click.echo(f"Collection '{collection_name}' does not exist")
        else:
            click.echo("✗ Qdrant health check failed")
            sys.exit(1)


def main() -> None:
    """Main entrypoint for CLI."""
    cli()


if __name__ == "__main__":
    main()
