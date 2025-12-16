"""Recommendation operations for Qdrant.

This module handles recommendation queries using positive and negative examples.
Moved from search/ module to vectordb/operations/ for better organization.
"""

from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http import models

from vector_sentiment.models.schemas import SearchResult


class VectorRecommender:
    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        vector_name: str,
    ) -> None:
        self.client = client
        self.collection_name = collection_name
        self.vector_name = vector_name

        logger.info(
            f"Initialized VectorRecommender for collection '{collection_name}' "
            f"with vector '{vector_name}'"
        )

    def recommend(
        self,
        positive_ids: list[int],
        negative_ids: list[int] | None = None,
        filter_label: str | None = None,
        limit: int = 10,
        score_threshold: float | None = None,
        shard_key_selector: str | int | None = None,
    ) -> list[SearchResult]:
        logger.info(
            f"Generating recommendations: positive={len(positive_ids)}, "
            f"negative={len(negative_ids) if negative_ids else 0}, "
            f"filter_label={filter_label}, limit={limit}"
        )

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

        # Execute recommendation query
        query = models.RecommendQuery(
            recommend=models.RecommendInput(
                positive=positive_ids,
                negative=negative_ids or [],
            )
        )

        # Execute recommendation with shard key
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query,
            using=self.vector_name,
            query_filter=query_filter,
            limit=limit,
            score_threshold=score_threshold,
            with_payload=True,
            shard_key_selector=shard_key_selector,  # Filter by shard
        )

        recommendations = response.points

        logger.info(f"Generated {len(recommendations)} recommendations")

        # Convert to SearchResult objects
        results = []
        for rec in recommendations:
            result = SearchResult(
                id=rec.id,
                score=rec.score,
                label=rec.payload.get("label", "unknown"),
                text=rec.payload.get("text", None),
            )
            results.append(result)

        return results

    def recommend_by_label(
        self,
        positive_label: str,
        negative_label: str | None = None,
        limit: int = 10,
    ) -> list[SearchResult]:
        # Scroll to get example IDs with positive label
        positive_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="label",
                    match=models.MatchValue(value=positive_label),
                )
            ]
        )

        positive_points, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=positive_filter,
            limit=5,  # Get 5 examples
            with_payload=False,
        )

        positive_ids = [int(p.id) for p in positive_points]  # type: ignore[arg-type]

        # Get negative examples if specified
        negative_ids = None
        if negative_label:
            negative_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="label",
                        match=models.MatchValue(value=negative_label),
                    )
                ]
            )

            negative_points, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=negative_filter,
                limit=5,
                with_payload=False,
            )

            negative_ids = [int(p.id) for p in negative_points]  # type: ignore[arg-type]

        logger.info(
            f"Using label-based examples: positive={len(positive_ids)}, "
            f"negative={len(negative_ids) if negative_ids else 0}"
        )

        if not positive_ids:
            logger.warning(
                f"No points found with label '{positive_label}'. "
                f"Please check available labels using `analytics.py`."
            )
            # You might want to raise an error or just return empty
            raise ValueError(f"No points found with positive label: {positive_label}")

        # Generate recommendations
        return self.recommend(
            positive_ids=positive_ids,
            negative_ids=negative_ids,
            limit=limit,
        )
