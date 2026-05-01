"""Hybrid retriever — BM25 keyword search + semantic similarity with RRF fusion."""

import logging
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

from retrieval.corpus_loader import Chunk, Corpus
from retrieval.embeddings import embed_query, embed_texts
from utils.config import SEED, TOP_K

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """A single retrieval hit with score and metadata."""

    chunk: Chunk
    score: float
    method: str  # "bm25", "semantic", or "hybrid"


class HybridRetriever:
    """Combines BM25 keyword search with dense semantic similarity.

    Uses Reciprocal Rank Fusion (RRF) to merge rankings.
    """

    def __init__(self, corpus: Corpus) -> None:
        """Initialise the retriever by building BM25 and embedding indices.

        Args:
            corpus: The loaded and chunked corpus.
        """
        self._corpus = corpus
        self._chunks = corpus.chunks
        self._texts = corpus.texts

        np.random.seed(SEED)  # noqa: NPY002

        # --- BM25 index ---
        logger.info("Building BM25 index over %d chunks …", len(self._texts))
        tokenised = [text.lower().split() for text in self._texts]
        self._bm25 = BM25Okapi(tokenised)

        # --- Dense embeddings ---
        logger.info("Computing dense embeddings for %d chunks …", len(self._texts))
        self._embeddings: np.ndarray = embed_texts(self._texts)

        logger.info("HybridRetriever ready.")

    # ------------------------------------------------------------------
    # Internal ranking helpers
    # ------------------------------------------------------------------

    def _bm25_rank(self, query: str, top_n: int) -> list[tuple[int, float]]:
        """Return top-N (index, score) pairs from BM25."""
        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_n]
        return [(int(i), float(scores[i])) for i in top_indices]

    def _semantic_rank(self, query: str, top_n: int) -> list[tuple[int, float]]:
        """Return top-N (index, score) pairs from cosine similarity."""
        q_vec = embed_query(query)
        # Embeddings are already L2-normalised → dot product = cosine sim
        scores = self._embeddings @ q_vec
        top_indices = np.argsort(scores)[::-1][:top_n]
        return [(int(i), float(scores[i])) for i in top_indices]

    @staticmethod
    def _rrf_fuse(
        *rankings: list[tuple[int, float]],
        k: int = 60,
    ) -> list[tuple[int, float]]:
        """Reciprocal Rank Fusion across multiple ranked lists.

        Args:
            *rankings: Each is a list of (doc_index, score) sorted descending.
            k: RRF constant (default 60).

        Returns:
            Fused ranking as list of (doc_index, rrf_score) sorted descending.
        """
        rrf_scores: dict[int, float] = {}
        for ranking in rankings:
            for rank, (doc_idx, _score) in enumerate(ranking):
                rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + 1.0 / (k + rank + 1)
        sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_items

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K,
        domain_filter: str | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve the top-K most relevant chunks for a query.

        Args:
            query: The search query string.
            top_k: Number of results to return.
            domain_filter: Optional domain to restrict results to.

        Returns:
            List of :class:`RetrievalResult` sorted by relevance.
        """
        # Expand candidate pool so we have room after domain filtering
        pool_size = top_k * 4

        bm25_hits = self._bm25_rank(query, pool_size)
        sem_hits = self._semantic_rank(query, pool_size)
        fused = self._rrf_fuse(bm25_hits, sem_hits)

        results: list[RetrievalResult] = []
        for doc_idx, rrf_score in fused:
            chunk = self._chunks[doc_idx]
            if domain_filter and chunk.domain != domain_filter.lower():
                continue
            results.append(
                RetrievalResult(chunk=chunk, score=rrf_score, method="hybrid")
            )
            if len(results) >= top_k:
                break

        logger.debug(
            "Query: %s | domain=%s | returned %d results",
            query[:60],
            domain_filter,
            len(results),
        )
        return results
