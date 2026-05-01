"""Embedding utilities — encode text chunks using sentence-transformers."""

import logging
from typing import Optional

import numpy as np

from utils.config import SEED

logger = logging.getLogger(__name__)

# Lazy-loaded model singleton
_model: Optional[object] = None
_MODEL_NAME = "all-MiniLM-L6-v2"


def _get_model() -> object:
    """Lazy-load the sentence-transformers model (heavy import)."""
    global _model  # noqa: PLW0603
    if _model is None:
        logger.info("Loading embedding model: %s", _MODEL_NAME)
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(_MODEL_NAME)
        logger.info("Embedding model loaded.")
    return _model


def embed_texts(texts: list[str], batch_size: int = 64) -> np.ndarray:
    """Encode a list of texts into dense vectors.

    Args:
        texts: List of text strings to embed.
        batch_size: Encoding batch size.

    Returns:
        2-D numpy array of shape ``(len(texts), dim)``.
    """
    model = _get_model()
    np.random.seed(SEED)  # noqa: NPY002
    embeddings: np.ndarray = model.encode(  # type: ignore[union-attr]
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    logger.info("Embedded %d texts → shape %s", len(texts), embeddings.shape)
    return embeddings


def embed_query(query: str) -> np.ndarray:
    """Encode a single query string into a dense vector.

    Args:
        query: The query text.

    Returns:
        1-D numpy array of shape ``(dim,)``.
    """
    model = _get_model()
    vec: np.ndarray = model.encode(  # type: ignore[union-attr]
        [query],
        normalize_embeddings=True,
    )
    return vec[0]
