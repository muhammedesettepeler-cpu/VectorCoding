import argparse
import sys
from collections import Counter
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from qdrant_client.http.exceptions import UnexpectedResponse

from scenarios.utils.common import get_qdrant_client, print_results, setup_logging
from vector_sentiment.config.settings import get_settings
from vector_sentiment.vectordb.operations import CollectionManager
from vector_sentiment.vectordb.operations.recommend import VectorRecommender

# Constants
SEPARATOR_LINE_LENGTH = 80
DEFAULT_POSITIVE_LABEL = "positive"
DEFAULT_NEGATIVE_LABEL = "negative"
DEFAULT_POSITIVE_IDS = "1,2,3,5,8,13,21"
DEFAULT_NEGATIVE_IDS = "10,15,20,25"
DEFAULT_RESULT_LIMIT = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate recommendations based on positive/negative examples"
    )

    project_root = Path(__file__).parent.parent
    default_config = project_root / "data_dir" / "master_config.yaml"

    parser.add_argument(
        "--config",
        type=str,
        default=str(default_config),
        help=f"Path to config YAML file (default: {default_config})",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        help="Scenario name to override active_scenario in master_config",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="labels",
        choices=["labels", "ids"],
        help="Recommendation mode: 'labels' or 'ids'",
    )
    parser.add_argument(
        "--positive-label",
        type=str,
        default=DEFAULT_POSITIVE_LABEL,
        help="Positive example label (for labels mode)",
    )
    parser.add_argument(
        "--negative-label",
        type=str,
        default=DEFAULT_NEGATIVE_LABEL,
        help="Negative example label (for labels mode)",
    )
    parser.add_argument(
        "--positive-ids",
        type=str,
        default=DEFAULT_POSITIVE_IDS,
        help="Comma-separated positive point IDs (for ids mode)",
    )
    parser.add_argument(
        "--negative-ids",
        type=str,
        default=DEFAULT_NEGATIVE_IDS,
        help="Comma-separated negative point IDs (for ids mode)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_RESULT_LIMIT,
        help="Maximum number of recommendations",
    )
    parser.add_argument(
        "--filter-label",
        type=str,
        default=None,
        help="Filter results by label",
    )

    return parser.parse_args()


def parse_ids(ids_str: str) -> list[int]:
    if not ids_str:
        return []

    try:
        return [int(x.strip()) for x in ids_str.split(",")]
    except ValueError as e:
        raise ValueError(f"Invalid ID format in '{ids_str}': {e}") from e


def log_collection_error(collection_name: str, is_empty: bool = False) -> None:
    if is_empty:
        logger.error(f"Collection '{collection_name}' exists but is empty!")
    else:
        logger.error(f"Collection '{collection_name}' does not exist!")
    logger.error("Please run 'python scenarios/ingest.py' first to load data")


def display_statistics(results: list) -> None:
    if not results:
        return

    # Calculate and display score statistics
    avg_score = sum(r.score for r in results) / len(results)
    logger.info(f"\nAverage recommendation score: {avg_score:.4f}")
    logger.info(f"Top recommendation score: {results[0].score:.4f}")

    # Display label distribution using Counter (more efficient)
    label_counts = Counter(r.label for r in results)
    logger.info("\nRecommended label distribution:")
    for label, count in sorted(label_counts.items()):
        pct = (count / len(results)) * 100
        logger.info(f"  {label}: {count} ({pct:.1f}%)")


def main() -> None:
    """Run the recommendation scenario."""
    args = parse_args()
    setup_logging(level="INFO")

    # Load settings from .env
    settings = get_settings()

    logger.info("=" * SEPARATOR_LINE_LENGTH)
    logger.info("RECOMMENDATIONS")
    logger.info("=" * SEPARATOR_LINE_LENGTH)

    # Load config from master_config.yaml
    from vector_sentiment.config.dataset_config import DatasetConfig

    logger.info("MODE: Configuration-Driven (master_config.yaml)")
    logger.info(f"Config file: {args.config}")

    try:
        config = DatasetConfig.from_master_config(args.config, scenario=args.scenario)
        collection_name = config.collection_name
        logger.info(f"Dataset: {config.name}")
        logger.info(f"Description: {config.description}")
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return
    # Use actual model name from settings
    model_name = settings.embedding.model_name
    dense_vector_name = model_name  # Use model name as vector field name

    logger.info(f"Collection: {collection_name}")
    logger.info(f"Mode: {args.mode}")

    if args.mode == "labels":
        logger.info(f"Positive label: {args.positive_label}")
        logger.info(f"Negative label: {args.negative_label or 'None'}")
    else:
        logger.info(f"Positive IDs: {args.positive_ids}")
        logger.info(f"Negative IDs: {args.negative_ids or 'None'}")

    logger.info(f"Result limit: {args.limit}")

    try:
        # Initialize services
        logger.info("\nConnecting to Qdrant...")
        client, _ = get_qdrant_client()

        # Check if collection exists
        collection_mgr = CollectionManager(client)

        if not collection_mgr.collection_exists(collection_name):
            log_collection_error(collection_name, is_empty=False)
            return

        # Check if collection has data
        info = collection_mgr.get_collection_info(collection_name)
        if info and info.points_count == 0:
            log_collection_error(collection_name, is_empty=True)
            return

        logger.info(f"Found collection '{collection_name}' with {info.points_count:,} points")

        # Generate recommendations
        logger.info("\nGenerating recommendations...")
        recommender = VectorRecommender(
            client=client,
            collection_name=collection_name,
            vector_name=dense_vector_name,
        )

        if args.mode == "labels":
            results = recommender.recommend_by_label(
                positive_label=args.positive_label,
                negative_label=args.negative_label,
                limit=args.limit,
            )
        else:
            positive_ids = parse_ids(args.positive_ids)
            negative_ids = parse_ids(args.negative_ids) if args.negative_ids else None
            results = recommender.recommend(
                positive_ids=positive_ids,
                negative_ids=negative_ids,
                filter_label=args.filter_label,
                limit=args.limit,
            )

        # Display results
        logger.info(f"\n✓ Generated {len(results)} recommendations")
        print_results(results, title="Recommendations")

        # Display statistics
        display_statistics(results)

    except ValueError as e:
        logger.error(f"Invalid input: {e}")
        sys.exit(1)
    except UnexpectedResponse as e:
        logger.error(f"Qdrant connection error: {e}")
        logger.error("Please check your Qdrant connection settings in .env")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
