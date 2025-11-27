"""Recommendation service using Qdrant's recommend API.

This module provides recommendation functionality using positive and negative
examples to find similar or dissimilar vectors.
"""

from typing import Optional

from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http import models

from vector_sentiment.models.schemas import SearchResult


class RecommendationService:
    """Vector recommendation service.

    This class provides recommendation operations using Qdrant's recommend API,
    which finds vectors similar to positive examples and dissimilar to negative examples.

    Attributes:
        client: QdrantClient instance
        collection_name: Name of the collection
        vector_name: Name of the vector in the collection

    Example:
        >>> service = RecommendationService(
        ...     client=client,
        ...     collection_name="sentiment_vectors",
        ...     vector_name="all-MiniLM-L6-v2"
        ... )
        >>> # Find vectors similar to positive examples
        >>> results = service.recommend(positive_ids=[1, 2, 3], limit=10)
    """

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        vector_name: str,
    ) -> None:
        """Initialize recommendation service.

        Args:
            client: QdrantClient instance
            collection_name: Name of the collection
            vector_name: Name of the vector in the collection
        """
        self.client = client
        self.collection_name = collection_name
        self.vector_name = vector_name

        logger.info(
            f"Initialized RecommendationService for collection '{collection_name}' "
            f"with vector '{vector_name}'"
        )

    def recommend(
        self,
        positive_ids: list[int],
        negative_ids: Optional[list[int]] = None,
        filter_label: Optional[str] = None,
        limit: int = 10,
        score_threshold: Optional[float] = None,
    ) -> list[SearchResult]:
        """Get recommendations based on positive and negative examples.

        This method uses Qdrant's recommend API to find vectors that are:
        - Similar to positive examples
        - Dissimilar to negative examples (if provided)

        Args:
            positive_ids: List of point IDs to use as positive examples
            negative_ids: Optional list of point IDs to use as negative examples
            filter_label: Optional label to filter results by
            limit: Maximum number of recommendations
            score_threshold: Minimum similarity score

        Returns:
            List of SearchResult objects

        Example:
            >>> # Recommend similar to positive examples
            >>> results = service.recommend(
            ...     positive_ids=[1, 5, 10],
            ...     limit=5
            ... )
            >>>
            >>> # Recommend using both positive and negative examples
            >>> results = service.recommend(
            ...     positive_ids=[1, 2],  # Similar to these
            ...     negative_ids=[50, 51],  # But not similar to these
            ...     filter_label="positive",
            ...     limit=10
            ... )
        """
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

        # Execute recommend query
        recommendations = self.client.recommend(
            collection_name=self.collection_name,
            positive=positive_ids,
            negative=negative_ids or [],
            query_filter=query_filter,
            limit=limit,
            score_threshold=score_threshold,
            using=self.vector_name,  # Specify which named vector to use
            with_payload=True,
        )

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
        negative_label: Optional[str] = None,
        limit: int = 10,
    ) -> list[SearchResult]:
        """Get example point IDs by label and generate recommendations.

        This is a convenience method that first finds points with the specified
        labels and uses them for recommendations.

        Args:
            positive_label: Label for positive examples
            negative_label: Optional label for negative examples
            limit: Maximum number of recommendations

        Returns:
            List of SearchResult objects

        Example:
            >>> # Find more examples like "positive" but unlike "negative"
            >>> results = service.recommend_by_label(
            ...     positive_label="positive",
            ...     negative_label="negative",
            ...     limit=10
            ... )
        """
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

        positive_ids = [p.id for p in positive_points]

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

            negative_ids = [p.id for p in negative_points]

        logger.info(
            f"Using label-based examples: positive={len(positive_ids)}, "
            f"negative={len(negative_ids) if negative_ids else 0}"
        )

        # Generate recommendations
        return self.recommend(
            positive_ids=positive_ids,
            negative_ids=negative_ids,
            limit=limit,
        )
