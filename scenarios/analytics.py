import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import ipdb

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from loguru import logger

from scenarios.utils.common import get_qdrant_client, setup_logging
from vector_sentiment.config.settings import get_settings


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Analyze vector collection statistics")

    parser.add_argument(
        "--collection",
        type=str,
        help="Collection name to analyze (overrides config)",
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
        "--top-similar",
        type=int,
        default=0,
        help="Show top N most similar vector pairs (0=disable)",
    )

    return parser.parse_args()


def get_collection_statistics(client, collection_name: str) -> dict[str, Any]:  # noqa: ANN001
    """Get comprehensive collection statistics."""
    info = client.get_collection(collection_name)

    stats = {
        "collection_name": collection_name,
        "total_points": info.points_count,
        "vectors_count": info.vectors_count
        if hasattr(info, "vectors_count")
        else info.points_count,
        "indexed_vectors": info.indexed_vectors_count
        if hasattr(info, "indexed_vectors_count")
        else 0,
        "status": info.status,
    }

    # Get vector config info
    if hasattr(info.config, "params") and hasattr(info.config.params, "vectors"):
        vectors_config = info.config.params.vectors
        if isinstance(vectors_config, dict):
            for vector_name, vector_params in vectors_config.items():
                stats[f"vector_{vector_name}_size"] = vector_params.size
                stats[f"vector_{vector_name}_distance"] = vector_params.distance.name

    return stats


def analyze_label_distribution(client, collection_name: str) -> dict[str, int]:  # noqa: ANN001
    """Analyze label distribution in the collection."""
    labels: list[str] = []

    # Scroll through all points
    offset = None
    while True:
        result = client.scroll(
            collection_name=collection_name,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        points, offset = result

        if not points:
            break

        for point in points:
            if point.payload and "label" in point.payload:
                labels.append(str(point.payload["label"]))

        if offset is None:
            break

    return dict(Counter(labels))


def find_similar_pairs(client, collection_name: str, top_n: int = 5) -> list[tuple]:
    """Find most similar vector pairs in the collection."""
    logger.info(f"Analyzing top {top_n} similar pairs...")

    all_points = []
    offset = None

    while True:
        result = client.scroll(
            collection_name=collection_name,
            limit=100,
            offset=offset,
            with_payload=False,
            with_vectors=True,
        )

        points, offset = result

        if not points:
            break

        all_points.extend(points)

        if offset is None:
            break

    if len(all_points) < 2:
        return []

    # Calculate similarities (sample if too many)
    max_compare = min(len(all_points), 200)  # Limit to avoid O(n^2) explosion
    sample_points = all_points[:max_compare]

    similarities = []
    for i in range(len(sample_points)):
        for j in range(i + 1, len(sample_points)):
            # Get vectors - handle named vectors
            vec_i = sample_points[i].vector
            vec_j = sample_points[j].vector

            # If named vectors, get the first one
            if isinstance(vec_i, dict):
                vec_i = list(vec_i.values())[0]
            if isinstance(vec_j, dict):
                vec_j = list(vec_j.values())[0]

            # Calculate cosine similarity
            vec_i_np = np.array(vec_i)
            vec_j_np = np.array(vec_j)

            similarity = np.dot(vec_i_np, vec_j_np) / (
                np.linalg.norm(vec_i_np) * np.linalg.norm(vec_j_np)
            )

            similarities.append((sample_points[i].id, sample_points[j].id, float(similarity)))

    # Sort by similarity and return top N
    similarities.sort(key=lambda x: x[2], reverse=True)
    return similarities[:top_n]


def main() -> None:
    """Run analytics scenario."""
    args = parse_args()
    setup_logging(level="INFO")

    logger.info("COLLECTION ANALYTICS")

    # Determine collection name
    if args.collection:
        # Direct collection name override
        collection_name = args.collection
        logger.info(f"Using collection: {collection_name}")
    else:
        # Load from master config
        from vector_sentiment.config.dataset_config import DatasetConfig

        logger.info(f"Loading config: {args.config}")
        try:
            config = DatasetConfig.from_master_config(args.config, scenario=args.scenario)
            collection_name = config.collection_name
            logger.info(f"Dataset: {config.name}")
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            settings = get_settings()
            collection_name = settings.collection.name
            logger.warning(f"Falling back to settings: {collection_name}")

    logger.info(f"Analyzing collection: {collection_name}\n")

    # Connect to Qdrant
    client, _ = get_qdrant_client()

    # Get statistics
    logger.info("[1/3] Collecting statistics...")
    try:
        stats = get_collection_statistics(client, collection_name)

        logger.info("COLLECTION STATISTICS")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")

    except Exception as e:
        logger.error(f"Failed to get statistics: {e}")
        return

    # Analyze labels
    logger.info("\n[2/3] Analyzing label distribution...")
    try:
        label_dist = analyze_label_distribution(client, collection_name)

        total_labeled = sum(label_dist.values())

        logger.info("LABEL DISTRIBUTION")

        for label, count in sorted(label_dist.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_labeled * 100) if total_labeled > 0 else 0
            logger.info(f"  {label}: {count:,} ({percentage:.1f}%)")

        logger.info(f"\nTotal labeled points: {total_labeled:,}")
        ipdb.set_trace()
    except Exception as e:
        logger.error(f"Failed to analyze labels: {e}")

    # Find similar pairs
    if args.top_similar > 0:
        logger.info(f"\n[3/3] Finding top {args.top_similar} similar pairs...")
        try:
            similar_pairs = find_similar_pairs(client, collection_name, args.top_similar)

            logger.info(f"TOP {args.top_similar} SIMILAR PAIRS")

            for rank, (id1, id2, similarity) in enumerate(similar_pairs, 1):
                logger.info(f"{rank}. ID {id1} ↔ ID {id2}: {similarity:.4f}")

        except Exception as e:
            logger.error(f"Failed to find similar pairs: {e}")
    else:
        logger.info("\n[3/3] Skipping similarity analysis (use --top-similar to enable)")

    logger.info("ANALYTICS COMPLETE")


if __name__ == "__main__":
    main()
