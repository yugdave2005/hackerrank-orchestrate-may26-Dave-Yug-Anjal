"""Corpus loader — walks data/ directories, parses documents, and chunks them."""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from utils.config import CHUNK_OVERLAP_TOKENS, CHUNK_SIZE_TOKENS, DATA_DIR, DOMAINS

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A chunked fragment of a corpus document."""

    text: str
    domain: str  # hackerrank | claude | visa
    source_file: str  # relative path within data/
    chunk_index: int

    def __repr__(self) -> str:
        return f"Chunk(domain={self.domain!r}, src={self.source_file!r}, idx={self.chunk_index})"


@dataclass
class Corpus:
    """The full indexed corpus ready for retrieval."""

    chunks: list[Chunk] = field(default_factory=list)

    @property
    def texts(self) -> list[str]:
        """Return plain-text list aligned with chunks for vectorisation."""
        return [c.text for c in self.chunks]

    def filter_by_domain(self, domain: str) -> list[Chunk]:
        """Return only chunks belonging to a given domain."""
        return [c for c in self.chunks if c.domain == domain.lower()]


# ---------------------------------------------------------------------------
# Tokenisation helper (simple whitespace-based, good enough for chunking)
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"\S+")


def _estimate_tokens(text: str) -> int:
    """Rough token count by whitespace splitting (≈ 1.3x BPE tokens)."""
    return len(_WORD_RE.findall(text))


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def _chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE_TOKENS,
    overlap: int = CHUNK_OVERLAP_TOKENS,
) -> list[str]:
    """Split text into overlapping chunks of approximately *chunk_size* tokens.

    Args:
        text: The full document text.
        chunk_size: Target tokens per chunk.
        overlap: Token overlap between consecutive chunks.

    Returns:
        List of chunk strings.
    """
    words = _WORD_RE.findall(text)
    if not words:
        return []

    chunks: list[str] = []
    start = 0
    step = max(chunk_size - overlap, 1)

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        if end >= len(words):
            break
        start += step

    return chunks


# ---------------------------------------------------------------------------
# File parsing
# ---------------------------------------------------------------------------

def _read_file(path: Path) -> str:
    """Read a text/markdown file and return its content.

    Args:
        path: Absolute path to the file.

    Returns:
        File content as a string, or empty string on read failure.
    """
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        logger.warning("Could not read %s: %s", path, exc)
        return ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_corpus(data_dir: Path | None = None) -> Corpus:
    """Walk data directories, parse files, chunk, and return a Corpus.

    Args:
        data_dir: Root data directory. Defaults to ``DATA_DIR`` from config.

    Returns:
        A fully populated :class:`Corpus` instance.
    """
    data_dir = data_dir or DATA_DIR
    corpus = Corpus()
    total_files = 0

    for domain in DOMAINS:
        domain_dir = data_dir / domain
        if not domain_dir.exists():
            logger.warning("Domain directory missing: %s", domain_dir)
            continue

        # Collect all readable files (md, txt, etc.)
        files = sorted(
            p for p in domain_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in {".md", ".txt", ".html", ".csv"}
        )

        for fpath in files:
            content = _read_file(fpath)
            if not content.strip():
                continue

            relative = fpath.relative_to(data_dir).as_posix()
            text_chunks = _chunk_text(content)

            for idx, chunk_text in enumerate(text_chunks):
                corpus.chunks.append(
                    Chunk(
                        text=chunk_text,
                        domain=domain,
                        source_file=relative,
                        chunk_index=idx,
                    )
                )
            total_files += 1

    logger.info(
        "Corpus loaded: %d files → %d chunks across %s",
        total_files,
        len(corpus.chunks),
        DOMAINS,
    )
    return corpus
