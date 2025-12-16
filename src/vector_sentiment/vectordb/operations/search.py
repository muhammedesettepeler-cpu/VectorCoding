"""Vector similarity search operations for Qdrant.

This module handles vector similarity search with filtering and ranking.
Moved from search/ module to vectordb/operations/ for better organization.
"""

from typing import TYPE_CHECKING

from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http import models

from vector_sentiment.embeddings.service import EmbeddingService
from vector_sentiment.models.schemas import FilterOptions, SearchQuery, SearchResult

if TYPE_CHECKING:
    from vector_sentiment.embeddings.sparse import SparseEmbeddingService


class VectorSearcher:
    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        embedding_service: EmbeddingService,
        vector_name: str,
    ) -> None:
        self.client = client
        self.collection_name = collection_name
        self.embedding_service = embedding_service
        self.vector_name = vector_name

        logger.info(
            f"Initialized VectorSearcher for collection '{collection_name}' "
            f"with vector '{vector_name}'"
        )

    def search(
        self,
        query_text: str,
        filter_label: str | None = None,
        score_threshold: float | None = None,
        limit: int = 10,
        shard_key_selector: str | int | None = None,
    ) -> list[SearchResult]:
        logger.info(
            f"Searching for '{query_text[:50]}...' with "
            f"filter_label={filter_label}, score_threshold={score_threshold}, limit={limit}"
        )

        # Generate query embedding
        query_embedding = self.embedding_service.encode_single(query_text)

        # Build filter if label specified
        query_filter = None
        if filter_label is not None:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="label",
                        match=models.MatchValue(value=filter_label),
                    )
                ]
            )
            logger.debug(f"Applied label filter: {filter_label}")

        # Execute search with shard key filtering
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding.tolist(),
            using=self.vector_name,
            query_filter=query_filter,
            limit=limit,
            score_threshold=score_threshold,
            with_payload=True,
            shard_key_selector=shard_key_selector,  # Filter by shard
        )

        search_results = response.points

        logger.info(f"Found {len(search_results)} results")

        # Convert to SearchResult objects
        results = []
        for hit in search_results:
            result = SearchResult(
                id=hit.id,
                score=hit.score,
                label=hit.payload.get("label", "unknown"),
                text=hit.payload.get("text", None),
            )
            results.append(result)

        return results

    def search_with_options(
        self,
        query: SearchQuery,
    ) -> list[SearchResult]:
        filters = query.filters or FilterOptions()

        return self.search(
            query_text=query.query_text,
            filter_label=filters.label,
            score_threshold=filters.score_threshold,
            limit=filters.limit,
        )

    def hybrid_search(
        self,
        query_text: str,
        sparse_vector_name: str,
        sparse_embedding_service: "SparseEmbeddingService",
        filter_label: str | None = None,
        limit: int = 10,
    ) -> list[SearchResult]:
        logger.info(f"Hybrid search for '{query_text[:50]}...' with limit={limit}")

        # Generate embeddings
        dense_embedding = self.embedding_service.encode_single(query_text)
        sparse_embedding = sparse_embedding_service.encode_single(query_text)

        # Build filter if label specified
        query_filter = None
        if filter_label is not None:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="label",
                        match=models.MatchValue(value=filter_label),
                    )
                ]
            )

        # Hybrid search with prefetch and RRF fusion
        response = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                # Dense vector search
                models.Prefetch(
                    query=dense_embedding.tolist(),
                    using=self.vector_name,
                    limit=limit * 2,  # Over-fetch for better fusion
                ),
                # Sparse vector search
                models.Prefetch(
                    query=models.SparseVector(
                        indices=sparse_embedding.indices,
                        values=sparse_embedding.values,
                    ),
                    using=sparse_vector_name,
                    limit=limit * 2,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        )

        search_results = response.points
        logger.info(f"Hybrid search found {len(search_results)} results")

        # Convert to SearchResult objects
        results = []
        for hit in search_results:
            result = SearchResult(
                id=hit.id,
                score=hit.score,
                label=hit.payload.get("label", "unknown"),
                text=hit.payload.get("text", None),
            )
            results.append(result)

        return results
