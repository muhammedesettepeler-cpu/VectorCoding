"""Advanced Qdrant concepts documentation.

This module documents advanced Qdrant features that are important to understand
for production deployments, even though they are not fully implemented in this project.

Topics covered:
- Quantization strategies for memory optimization
- Shard key selectors for horizontal scaling (CRITICAL)
- Payload indexing for query optimization
- Batch operations for performance

Note: This file contains long documentation lines which are intentionally ignored.
"""
# ruff: noqa: E501


def quantization_concepts() -> str:
    """Document quantization strategies for vector storage.

    Quantization reduces memory usage by compressing vector representations
    with minimal accuracy loss. This is critical for large-scale deployments.

    Returns:
        Detailed documentation string about quantization
    """
    return """
QUANTIZATION STRATEGIES
=======================

Quantization compresses vector representations to reduce memory usage while maintaining
acceptable search accuracy. This is crucial for production deployments with millions
or billions of vectors.

1. SCALAR QUANTIZATION
   --------------------
   Converts float32 vectors (4 bytes per dimension) to int8 (1 byte per dimension).

   Memory Savings: 4x reduction (e.g., 384-dim vector: 1.5KB → 384 bytes)

   Accuracy Impact: Minimal (typically <1% reduction in recall@10)

   Configuration Example:
   ```python
   from qdrant_client.http import models

   client.create_collection(
       collection_name="vectors",
       vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
       quantization_config=models.ScalarQuantization(
           scalar=models.ScalarQuantizationConfig(
               type=models.ScalarType.INT8,
               quantile=0.99,  # Outlier handling
               always_ram=True,  # Keep quantized vectors in RAM
           )
       )
   )
   ```

   When to Use:
   - Large collections (>1M vectors)
   - Memory-constrained environments
   - Acceptable ~1% accuracy tradeoff

2. PRODUCT QUANTIZATION
   ---------------------
   Advanced compression technique that splits vectors into sub-vectors and
   quantizes each separately. Higher compression ratio at cost of more accuracy loss.

   Memory Savings: 8x-64x reduction (configurable)

   Accuracy Impact: Moderate (5-10% reduction depending on settings)

   Configuration Example:
   ```python
   client.create_collection(
       collection_name="vectors",
       vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
       quantization_config=models.ProductQuantization(
           product=models.ProductQuantizationConfig(
               compression=models.CompressionRatio.X16,
               always_ram=True,
           )
       )
   )
   ```

   When to Use:
   - Extremely large collections (>10M vectors)
   - Severe memory constraints
   - Can tolerate accuracy loss

3. TRADE-OFFS SUMMARY
   ------------------
   | Method              | Memory | Speed  | Accuracy |
   |---------------------|--------|--------|----------|
   | No Quantization     | 1x     | 1x     | 100%     |
   | Scalar (int8)       | 0.25x  | 1.2x   | 99%      |
   | Product (x16)       | 0.06x  | 1.5x   | 90-95%   |
   | Product (x64)       | 0.02x  | 2x     | 85-90%   |

RECOMMENDATION: Start with scalar quantization for production. Only move to product
quantization if memory is critically constrained.
"""


def shard_key_selector_concepts() -> str:
    """Document shard key selectors for horizontal scaling.

    CRITICAL: Understanding this concept is essential for production deployments
    with multiple nodes and horizontal scaling requirements.

    Returns:
        Comprehensive documentation about shard key selectors
    """
    return """
SHARD KEY SELECTORS - CRITICAL CONCEPT
=======================================

Shard key selectors enable horizontal partitioning of data across multiple nodes
for scalability, isolation, and performance. This is one of the most important
concepts for production Qdrant deployments.

WHY THIS MATTERS
----------------
In production, you often need to:
1. Scale beyond a single node's capacity
2. Isolate data by tenant/customer for compliance
3. Balance load across multiple servers
4. Enable geographic data distribution

Without sharding, all data lives on one node, creating a single point of failure
and capacity bottleneck.

HOW SHARD KEYS WORK
-------------------
A shard key is a field in the payload used to determine which shard (partition)
stores the data. All vectors with the same shard key value are stored together
on the same shard.

Example: In a multi-tenant application, use tenant_id as the shard key:
- All vectors for tenant "companyA" → Shard 1
- All vectors for tenant "companyB" → Shard 2
- All vectors for tenant "companyC" → Shard 3

CONFIGURATION EXAMPLE
--------------------
```python
from qdrant_client.http import models

# Create collection with sharding enabled
client.create_collection(
    collection_name="multi_tenant_vectors",
    vectors_config={"embedding": models.VectorParams(size=384, distance=models.Distance.COSINE)},
    shard_number=6,  # Number of shards
    sharding_method=models.ShardingMethod.CUSTOM,  # Enable custom sharding
)

# Create shard key for tenant isolation
client.create_shard_key(
    collection_name="multi_tenant_vectors",
    shard_key="tenant_id"
)
```

USING SHARD KEY DURING UPLOAD
-----------------------------
```python
# Upload vectors with shard key selector
client.upsert(
    collection_name="multi_tenant_vectors",
    points=[
        models.PointStruct(
            id=1,
            vector={"embedding": [0.1, 0.2, ...]},
            payload={"tenant_id": "companyA", "text": "..."}
        ),
        # ... more points ...
    ],
    shard_key_selector="companyA",  # All these points go to companyA's shard
)
```

USING SHARD KEY DURING SEARCH
----------------------------
```python
# Search only within one tenant's data
results = client.search(
    collection_name="multi_tenant_vectors",
    query_vector=[0.1, 0.2, ...],
    shard_key_selector="companyA",  # Only search companyA's shard
    limit=10
)
```

BENEFITS:
- Faster search (only search relevant shard)
- Data isolation (tenant data never mixes)
- Compliance (can isolate by geography)

COMMON USE CASES
----------------

1. MULTI-TENANT SaaS APPLICATIONS
   Shard Key: tenant_id or customer_id

   Why: Each customer's data is isolated. Searches only touch one shard,
   reducing query time. Easy to comply with data residency requirements.

   Example:
   ```python
   payload = {
       "tenant_id": "acme_corp",
       "document": "Contract for Acme Corporation",
       "label": "contract"
   }
   ```

2. GEOGRAPHIC DISTRIBUTION
   Shard Key: country or region

   Why: Store EU data in EU nodes, US data in US nodes. Reduces latency
   and ensures GDPR compliance.

   Example:
   ```python
   payload = {
       "region": "eu-west",
       "user_query": "product search",
       "label": "positive"
   }
   ```

3. WORKLOAD ISOLATION
   Shard Key: priority or department
   Why: High-priority workloads get dedicated resources. One department's
   heavy usage doesn't impact others.
   Example:
   ```python
   payload = {
       "department": "engineering",
       "document": "Technical specification",
       "priority": "high"
   }
   ```

4. TIME-BASED PARTITIONING
   Shard Key: year_month

   Why: Archive old data to cheaper storage. Recent data on fast SSDs.
   Easy to delete old partitions.

   Example:
   ```python
   payload = {
       "year_month": "2024_03",
       "event": "user interaction",
       "label": "positive"
   }
   ```

WHEN TO USE SHARD KEYS
----------------------
USE shard keys when:
✓ Multi-tenant application with many customers
✓ Need data isolation for compliance
✓ Dataset too large for single node (>10M vectors)
✓ Want to optimize search by partitioning
✓ Need geographic data distribution

DON'T USE shard keys when:
✗ Single tenant or small dataset
✗ All searches need to scan all data
✗ Adding unnecessary complexity to simple use case

IMPORTANT CONSIDERATIONS
------------------------
1. Choose shard keys with good cardinality (many unique values)
   Bad: boolean field (only 2 shards)
   Good: user_id, tenant_id (thousands of shards)

2. Balanced distribution matters
   Bad: 90% of data in one tenant, 10% in others (imbalanced load)
   Good: Evenly distributed data across tenants

3. Shard keys are immutable
   Cannot change a point's shard key after creation. Must delete and re-insert.

4. Query planning
   Always use shard_key_selector in queries when possible for performance.

PRODUCTION ARCHITECTURE EXAMPLE
-------------------------------
E-commerce platform with 1000 merchants:

Collection: product_embeddings
Shard Key: merchant_id
Shards: 12 (distributed across 4 nodes)

Benefits:
- Each merchant's products isolated (security)
- Product search only within merchant (fast)
- Can scale by adding more nodes
- Can analyze per-merchant usage

Code:
```python
# Insert products
client.upsert(
    collection_name="product_embeddings",
    points=[...],
    shard_key_selector=merchant_id
)

# Search within merchant
client.search(
    collection_name="product_embeddings",
    query_vector=query_embedding,
    shard_key_selector=merchant_id,  # Only search this merchant's shard
    limit=20
)
```

CONCLUSION
----------
Shard key selectors are essential for:
- Horizontal scaling across multiple nodes
- Multi-tenant data isolation
- Query performance optimization
- Compliance with data residency laws

Understanding and properly implementing sharding is the difference between a
prototype and a production-ready vector database deployment.
"""


def payload_indexing_concepts() -> str:
    """Document payload indexing for query optimization.

    Returns:
        Documentation about payload indexing strategies
    """
    return """
PAYLOAD INDEXING
================

Payload indexes speed up filtered searches by creating indexes on metadata fields.
Without indexes, Qdrant must scan all points to apply filters.

INDEX TYPES
-----------

1. KEYWORD INDEX
   For exact string matching (e.g., labels, categories, IDs)

   Example:
   ```python
   client.create_payload_index(
       collection_name="vectors",
       field_name="label",
       field_schema=models.PayloadSchemaType.KEYWORD
   )
   ```

2. INTEGER INDEX
   For numeric fields (e.g., user_id, timestamp)

   Example:
   ```python
   client.create_payload_index(
       collection_name="vectors",
       field_name="user_id",
       field_schema=models.PayloadSchemaType.INTEGER
   )
   ```

3. FLOAT INDEX
   For floating-point numbers (e.g., scores, ratings)

4. GEO INDEX
   For geographic coordinates (lat/lon)

WHEN TO CREATE INDEXES
----------------------
Create indexes for fields that:
✓ Are frequently used in filters
✓ Have high cardinality (many unique values)
✓ Are used in range queries

For example, in sentiment analysis:
- Index "label" field (positive/negative/neutral)
- Index "timestamp" for time-range filtering

Query Performance:
- Without index: O(N) - scan all points
- With index: O(log N) - index lookup
"""


def batch_operations_concepts() -> str:
    """Document batch search and recommend operations.

    Returns:
        Documentation about batch operations for performance
    """
    return """
BATCH OPERATIONS
================

Batch operations process multiple queries in a single API call, improving
throughput and reducing network overhead.

SEARCH BATCH
------------
Execute multiple search queries at once.

Example:
```python
from qdrant_client.http import models

queries = [
    models.SearchRequest(
        vector=models.NamedVector(name="embedding", vector=[0.1, 0.2, ...]),
        limit=10,
        filter=models.Filter(
            must=[models.FieldCondition(key="label", match=models.MatchValue(value="positive"))]
        )
    ),
    models.SearchRequest(
        vector=models.NamedVector(name="embedding", vector=[0.3, 0.4, ...]),
        limit=10,
    ),
]

results = client.search_batch(
    collection_name="vectors",
    requests=queries
)
```

Benefits:
- Reduced network round-trips
- Better resource utilization
- Higher throughput

RECOMMEND BATCH
---------------
Execute multiple recommendation queries simultaneously.

Example:
```python
requests = [
    models.RecommendRequest(
        positive=[1, 2, 3],  # IDs of positive examples
        negative=[10, 11],   # IDs of negative examples
        limit=5
    ),
    # ... more requests
]

results = client.recommend_batch(
    collection_name="vectors",
    requests=requests
)
```

Use Cases:
- Bulk recommendation generation
- A/B testing multiple strategies
- Batch processing pipelines
"""


def print_all_concepts() -> None:
    """Print all concept documentation.

    This function can be called to display all advanced Qdrant concepts
    for educational purposes.
    """
    print(quantization_concepts())
    print("\n" + "=" * 80 + "\n")
    print(shard_key_selector_concepts())
    print("\n" + "=" * 80 + "\n")
    print(payload_indexing_concepts())
    print("\n" + "=" * 80 + "\n")
    print(batch_operations_concepts())


if __name__ == "__main__":
    # When run as script, display all concepts
    print_all_concepts()
