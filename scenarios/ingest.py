import argparse
import sys
from pathlib import Path

from loguru import logger

# Add project root to path to make scenarios importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from scenarios.utils.common import get_qdrant_client, print_stats, setup_logging
from vector_sentiment.config.dataset_config import DatasetConfig
from vector_sentiment.config.settings import get_settings
from vector_sentiment.data.loader import ParquetDataLoader
from vector_sentiment.embeddings.service import EmbeddingService
from vector_sentiment.vectordb.operations import (
    CollectionManager,
    IndexManager,
    PointCreator,
)

SPARSE_VECTOR_NAME = "sparse"


def parse_args():  # noqa: ANN201
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Ingest data from Parquet to Qdrant")

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
        "--recreate",
        action="store_true",
        help="Force recreation of collection (deletes existing data)",
    )

    return parser.parse_args()


def main() -> None:
    """Run the data ingestion scenario."""
    args = parse_args()
    setup_logging(level="INFO")

    # Load settings from .env
    settings = get_settings()
    logger.info("DATA INGESTION (Scenario-Driven)")
    logger.info(f"Config file: {args.config}")

    # Load config
    try:
        # Use from_master_config which auto-detects master config format
        config = DatasetConfig.from_master_config(args.config, scenario=args.scenario)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return

    config_dir = Path(args.config).parent
    data_path = config.get_data_path(config_dir)
    collection_name = config.collection_name
    model_name = settings.embedding.model_name
    dense_vector_name = model_name  # Sentence transformer model name
    vector_size = settings.collection.vector_size
    batch_size = config.batch_size or settings.embedding.batch_size
    enable_sparse = config.enable_sparse
    recreate = config.recreate or args.recreate

    logger.info(f"Dataset: {config.name}")
    logger.info(f"Description: {config.description}")
    logger.info(f"Data path: {data_path}")
    logger.info(f"Collection: {collection_name}")
    logger.info(f"Text column: {config.text_column}")
    logger.info(f"Label column: {config.label_column or 'None (no labels)'}")
    logger.info(f"Metadata columns: {config.metadata_columns or 'None'}")
    logger.info(f"Embedding model: {model_name}")
    logger.info(f"Sparse vectors: {enable_sparse}")
    logger.info(f"Recreate collection: {recreate}")

    # Validate data path
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        logger.error("Please provide a valid data file path in master_config.yaml")
        return

    # Initialize services
    logger.info("\n[1/5] Connecting to Qdrant...")
    client, _ = get_qdrant_client()

    logger.info("\n[2/5] Initializing embedding service...")
    embedding_service = EmbeddingService(model_name=model_name)

    # Initialize sparse embedding service if enabled
    sparse_service = None
    if enable_sparse:
        from vector_sentiment.embeddings.sparse import SparseEmbeddingService

        logger.info("Initializing SPLADE sparse embedding service...")
        sparse_service = SparseEmbeddingService(model_name=settings.embedding.sparse_model_name)

    # Setup collection
    logger.info("\n[3/5] Setting up collection...")
    collection_mgr = CollectionManager(client)

    # Determine sparse vector name
    sparse_vector_name = SPARSE_VECTOR_NAME if enable_sparse else None

    if recreate:
        logger.warning("Recreating collection (existing data will be deleted)")
        collection_mgr.recreate_collection(
            collection_name=collection_name,
            vector_size=vector_size,
            vector_name=dense_vector_name,
            sparse_vector_name=sparse_vector_name,
        )
    elif not collection_mgr.collection_exists(collection_name):
        logger.info(f"Creating new collection: {collection_name}")
        collection_mgr.create_collection(
            collection_name=collection_name,
            vector_size=vector_size,
            vector_name=dense_vector_name,
            sparse_vector_name=sparse_vector_name,
        )
    else:
        # Collection exists - check if it has data
        info = collection_mgr.get_collection_info(collection_name)
        if info and info.points_count > 0:
            logger.info(
                f"✓ Collection '{collection_name}' already exists with "
                f"{info.points_count:,} points - skipping ingestion"
            )
            logger.info("To re-ingest data, use --recreate flag or set recreate: true in config")
            return
        else:
            logger.info(f"Collection '{collection_name}' exists but is empty - proceeding")

    # Create payload index for efficient filtering
    logger.info("\n[4/5] Creating payload indexes...")
    index_mgr = IndexManager(client)

    # Create index on label field if it exists
    if config.label_column:
        try:
            index_mgr.create_payload_index(
                collection_name=collection_name,
                field_name="label",
                field_schema="keyword",
            )
            logger.info("✓ Created index on 'label' field")
        except Exception as e:
            logger.debug(f"Index already exists or creation failed: {e}")

    # Create indexes on metadata fields
    for meta_col in config.metadata_columns:
        try:
            index_mgr.create_payload_index(
                collection_name=collection_name,
                field_name=meta_col,
                field_schema="keyword",
            )
            logger.info(f"✓ Created index on '{meta_col}' field")
        except Exception as e:
            logger.debug(f"Index on '{meta_col}' already exists or failed: {e}")

    # Load and ingest data
    logger.info("\n[5/5] Ingesting data...")
    creator = PointCreator(client, collection_name)

    with ParquetDataLoader(data_path, batch_size=batch_size) as loader:
        total_rows = loader.get_total_rows()
        logger.info(f"Total rows to process: {total_rows:,}")

        current_id = 0
        processed = 0

        for batch_df in loader.iter_batches():
            # Extract data using config-driven column mappings
            texts, labels, metadata = loader.extract_batch(
                batch_df,
                text_column=config.text_column,
                label_column=config.label_column,
                metadata_columns=config.metadata_columns,
            )

            # Generate dense embeddings
            embeddings = embedding_service.encode(texts)

            # Generate sparse embeddings if enabled
            sparse_vectors = None
            if sparse_service:
                sparse_vectors = sparse_service.encode(texts)

            # Prepare payloads
            payloads = []
            for idx, text in enumerate(texts):
                payload = {"text": text}

                # Add label if available
                if labels:
                    payload["label"] = str(labels[idx])

                # Add metadata
                for meta_key, meta_values in metadata.items():
                    payload[meta_key] = str(meta_values[idx])

                payloads.append(payload)

            # Upload to Qdrant
            count = creator.upsert_points(
                vectors=embeddings,
                payloads=payloads,
                vector_name=dense_vector_name,
                start_id=current_id,
                sparse_vectors=sparse_vectors,
                sparse_vector_name=sparse_vector_name,
            )

            current_id += count
            processed += count

            # Progress update
            progress_pct = (processed / total_rows) * 100
            logger.info(f"Progress: {processed:,}/{total_rows:,} ({progress_pct:.1f}%)")

    # Summary
    logger.info("✓ INGESTION COMPLETE")

    print_stats(
        "Summary",
        Dataset=config.name,
        Collection=collection_name,
        EmbeddingModel=model_name,
        TotalVectors=f"{processed:,}",
        VectorDimension=vector_size,
        BatchSize=batch_size,
        SparseVectors="Enabled" if enable_sparse else "Disabled",
        TextColumn=config.text_column,
        LabelColumn=config.label_column or "None",
        MetadataColumns=", ".join(config.metadata_columns) if config.metadata_columns else "None",
    )


if __name__ == "__main__":
    main()
