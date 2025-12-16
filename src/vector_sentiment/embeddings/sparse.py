from dataclasses import dataclass

from fastembed import SparseTextEmbedding
from loguru import logger


@dataclass
class SparseVector:
    indices: list[int]
    values: list[float]


class SparseEmbeddingService:
    def __init__(self, model_name: str) -> None:

        self.model_name = model_name

        logger.info(f"Loading SPLADE sparse model: {model_name}")
        self.model = SparseTextEmbedding(model_name=model_name)
        logger.info(f"SPLADE model loaded: {model_name}")

    def encode(self, texts: list[str]) -> list[SparseVector]:
        if not texts:
            return []

        logger.debug(f"Generating sparse embeddings for {len(texts)} texts")

        # fastembed returns generator, convert to list
        embeddings = list(self.model.embed(texts))

        results = []
        for emb in embeddings:
            sparse_vec = SparseVector(
                indices=emb.indices.tolist(),
                values=emb.values.tolist(),
            )
            results.append(sparse_vec)

        logger.debug(f"Generated {len(results)} sparse vectors")
        return results

    def encode_single(self, text: str) -> SparseVector:
        return self.encode([text])[0]
