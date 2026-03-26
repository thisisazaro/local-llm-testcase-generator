from __future__ import annotations

import hashlib
import logging
import math
import random
from typing import List

from app.config import SETTINGS

logger = logging.getLogger(__name__)


class HashEmbedder:
    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def _embed_one(self, text: str) -> List[float]:
        seed_hex = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        seed = int(seed_hex, 16)
        rng = random.Random(seed)
        vec = [rng.uniform(-1.0, 1.0) for _ in range(self.dim)]
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_one(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed_one(text)


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed. "
                "Set EMBEDDING_PROVIDER=hash or install sentence-transformers."
            ) from exc

        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        vectors = self.model.encode(texts, normalize_embeddings=True)
        return [vec.tolist() for vec in vectors]

    def embed_query(self, text: str) -> List[float]:
        vec = self.model.encode([text], normalize_embeddings=True)[0]
        return vec.tolist()


def get_embedder():
    provider = SETTINGS.embedding_provider.lower().strip()
    if provider == "sentence_transformers":
        try:
            logger.info("Using sentence-transformers embedder: %s", SETTINGS.embedding_model)
            return SentenceTransformerEmbedder(SETTINGS.embedding_model)
        except Exception as exc:
            if not SETTINGS.allow_mock_fallback:
                raise
            logger.warning("Embedding provider fallback to hash due to: %s", exc)

    logger.info("Using hash embedder (dim=%s)", SETTINGS.embedding_dim)
    return HashEmbedder(dim=SETTINGS.embedding_dim)
