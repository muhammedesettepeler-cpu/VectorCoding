import argparse
import sys
from collections import Counter
from pathlib import Path

from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scenarios.utils.common import get_qdrant_client, print_results, setup_logging
from vector_sentiment.config.dataset_config import DatasetConfig
from vector_sentiment.config.settings import get_settings
from vector_sentiment.embeddings.service import EmbeddingService
from vector_sentiment.vectordb.operations import CollectionManager
from vector_sentiment.vectordb.operations.search import VectorSearcher

SPARSE_VECTOR_NAME = "sparse"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search for similar vectors")

    project_root = Path(__file__).parent.parent
    default_config = project_root / "data_dir" / "master_config.yaml"

    parser.add_argument(
        "--config",
        type=str,
        default=str(default_config),
        help=f"Path to config YAML file (default: {default_config})",
    )
    parser.add_argument("--scenario", type=str, help="Scenario name override")
    parser.add_argument(
        "--query",
        type=str,
        default="Great product quality and excellent customer service",
        help="Search query text",
    )
    parser.add_argument("--label", type=str, help="Filter results by label")
    parser.add_argument(
        "--threshold", type=float, default=0.2, help="Minimum similarity score (0.0 to 1.0)"
    )
    parser.add_argument("--limit", type=int, default=10, help="Max results")
    parser.add_argument("--hybrid", action="store_true", help="Enable hybrid search")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(level="INFO")
    settings = get_settings()

    logger.info("Starting Search Scenario")
    if args.hybrid:
        logger.info("Mode: HYBRID (Dense + Sparse)")
    else:
        logger.info("Mode: SEMANTIC (Dense only)")

    logger.info(f"Configuration: {args.config}")

    # Load configuration
    try:
        config = DatasetConfig.from_master_config(args.config, scenario=args.scenario)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return

    collection_name = config.collection_name
    model_name = settings.embedding.model_name

    logger.info(f"Dataset: {config.name}")
    logger.info(f"Collection: {collection_name}")
    logger.info(f"Query: {args.query}")

    # Initialize services
    try:
        client, _ = get_qdrant_client()
        collection_mgr = CollectionManager(client)

        if not collection_mgr.collection_exists(collection_name):
            logger.error(f"Collection '{collection_name}' not found. Please ingest data first.")
            return

        embedding_service = EmbeddingService(model_name=model_name)
        searcher = VectorSearcher(
            client=client,
            collection_name=collection_name,
            embedding_service=embedding_service,
            vector_name=model_name,
        )

        # Execute Search
        if args.hybrid:
            from vector_sentiment.embeddings.sparse import SparseEmbeddingService

            logger.info("Initializing sparse model...")
            sparse_service = SparseEmbeddingService(model_name=settings.embedding.sparse_model_name)

            results = searcher.hybrid_search(
                query_text=args.query,
                sparse_vector_name=SPARSE_VECTOR_NAME,
                sparse_embedding_service=sparse_service,
                filter_label=args.label,
                limit=args.limit,
            )
        else:
            results = searcher.search(
                query_text=args.query,
                filter_label=args.label,
                score_threshold=args.threshold,
                limit=args.limit,
            )

        # Output Results
        print_results(results, title="Search Results")

        if results:
            avg_score = sum(r.score for r in results) / len(results)
            logger.info(f"Average Score: {avg_score:.4f}")

            label_counts = Counter(r.label for r in results)
            logger.info("Label Distribution:")
            for label, count in label_counts.most_common():
                logger.info(f"  {label}: {count}")
        else:
            logger.warning("No results found. Try lowering the --threshold or changing the query.")

    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
