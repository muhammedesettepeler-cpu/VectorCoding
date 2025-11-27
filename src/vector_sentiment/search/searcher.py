"""Search service for vector similarity search with filtering.

This module provides search functionality with support for query filters,
score thresholds, and named vector queries.
"""

from typing import Any, Optional

from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http import models

from vector_sentiment.embeddings.service import EmbeddingService
from vector_sentiment.models.schemas import FilterOptions, SearchQuery, SearchResult


class SearchService:
    """Vector similarity search service.

    This class provides search operations on Qdrant collections with support
    for filtering by payload, score thresholds, and result limits.

    Attributes:
        client: QdrantClient instance
        collection_name: Name of the collection to search
        embedding_service: Service for generating query embeddings
        vector_name: Name of the vector in the collection

    Example:
        >>> service = SearchService(
        ...     client=client,
        ...     collection_name="sentiment_vectors",
        ...     embedding_service=embedding_service,
        ...     vector_name="all-MiniLM-L6-v2"
        ... )
        >>> results = service.search("Great product!", limit=10)
    """

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        embedding_service: EmbeddingService,
        vector_name: str,
    ) -> None:
        """Initialize search service.

        Args:
            client: QdrantClient instance
            collection_name: Name of the collection
            embedding_service: Service for generating embeddings
            vector_name: Name of the vector in the collection
        """
        self.client = client
        self.collection_name = collection_name
        self.embedding_service = embedding_service
        self.vector_name = vector_name

        logger.info(
            f"Initialized SearchService for collection '{collection_name}' "
            f"with vector '{vector_name}'"
        )

    def search(
        self,
        query_text: str,
        filter_label: Optional[str] = None,
        score_threshold: Optional[float] = None,
        limit: int = 10,
    ) -> list[SearchResult]:
        """Search for similar vectors using text query.

        This method:
        1. Converts query text to embedding vector
        2. Creates NamedVector with model name
        3. Applies optional filters (label, score threshold)
        4. Returns formatted results

        Args:
            query_text: Text query to search for
            filter_label: Optional label to filter by (e.g., 'positive', 'negative')
            score_threshold: Minimum similarity score threshold (0.0 to 1.0)
            limit: Maximum number of results to return

        Returns:
            List of SearchResult objects

        Example:
            >>> # Basic search
            >>> results = service.search("Amazing quality")
            >>>
            >>> # Search with label filter
            >>> results = service.search(
            ...     "Good product",
            ...     filter_label="positive",
            ...     limit=5
            ... )
            >>>
            >>> # Search with score threshold
            >>> results = service.search(
            ...     "Excellent",
            ...     score_threshold=0.8,
            ...     limit=10
            ... )
        """
        logger.info(
            f"Searching for '{query_text[:50]}...' with "
            f"filter_label={filter_label}, score_threshold={score_threshold}, limit={limit}"
        )

        # Generate query embedding
        query_embedding = self.embedding_service.encode_single(query_text)

        # Create NamedVector (required for collections with named vectors)
        query_vector = models.NamedVector(
            name=self.vector_name,
            vector=query_embedding.tolist(),
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
            logger.debug(f"Applied label filter: {filter_label}")

        # Execute search
        search_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            query_filter=query_filter,
            limit=limit,
            score_threshold=score_threshold,
            with_payload=True,
        )

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
        """Search using SearchQuery model.

        This method accepts a validated SearchQuery Pydantic model.

        Args:
            query: SearchQuery model with query text and filters

        Returns:
            List of SearchResult objects

        Example:
            >>> from vector_sentiment.models import SearchQuery, FilterOptions
            >>> query = SearchQuery(
            ...     query_text="Great product",
            ...     filters=FilterOptions(label="positive", limit=5)
            ... )
            >>> results = service.search_with_options(query)
        """
        filters = query.filters or FilterOptions()

        return self.search(
            query_text=query.query_text,
            filter_label=filters.label,
            score_threshold=filters.score_threshold,
            limit=filters.limit,
        )

    def get_point_by_id(self, point_id: int) -> Optional[dict[str, Any]]:
        """Retrieve a point by its ID.

        Args:
            point_id: ID of the point to retrieve

        Returns:
            Point data as dictionary, or None if not found

        Example:
            >>> point = service.get_point_by_id(42)
            >>> if point:
            ...     print(point['payload'])
        """
        try:
            points = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[point_id],
                with_payload=True,
                with_vectors=False,
            )

            if points:
                return {
                    "id": points[0].id,
                    "payload": points[0].payload,
                }
            return None

        except Exception as e:
            logger.error(f"Error retrieving point {point_id}: {e}")
            return None
