"""Configuration module — environment variables, constants, and seed management."""

import logging
import os
import random
from pathlib import Path
from typing import Final

import numpy as np
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Repo root is two levels up from this file (code/utils/config.py → repo root)
REPO_ROOT: Final[Path] = Path(__file__).resolve().parent.parent.parent
CODE_DIR: Final[Path] = REPO_ROOT / "code"
DATA_DIR: Final[Path] = REPO_ROOT / "data"
SUPPORT_TICKETS_DIR: Final[Path] = REPO_ROOT / "support_tickets"

DEFAULT_INPUT_CSV: Final[Path] = SUPPORT_TICKETS_DIR / "support_tickets.csv"
DEFAULT_OUTPUT_CSV: Final[Path] = SUPPORT_TICKETS_DIR / "output.csv"

# ---------------------------------------------------------------------------
# Load .env (from repo root)
# ---------------------------------------------------------------------------
_env_path = REPO_ROOT / ".env"
load_dotenv(dotenv_path=_env_path)

# ---------------------------------------------------------------------------
# API keys (read-only from env)
# ---------------------------------------------------------------------------

def get_anthropic_api_key() -> str:
    """Return the Anthropic API key from environment variables.

    Raises:
        EnvironmentError: If ANTHROPIC_API_KEY is not set.
    """
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key or key == "your_key_here":
        raise EnvironmentError(
            "ANTHROPIC_API_KEY is not set. "
            "Copy .env.example to .env and add your key."
        )
    return key


# ---------------------------------------------------------------------------
# Model & retrieval constants
# ---------------------------------------------------------------------------
CLAUDE_MODEL: Final[str] = "claude-sonnet-4-20250514"
CLAUDE_TEMPERATURE: Final[float] = 0.0
CLAUDE_MAX_TOKENS: Final[int] = 1024

CHUNK_SIZE_TOKENS: Final[int] = 300
CHUNK_OVERLAP_TOKENS: Final[int] = 50
TOP_K: Final[int] = 5

SEED: Final[int] = 42

# Domain names
DOMAINS: Final[list[str]] = ["hackerrank", "claude", "visa"]

# Allowed output values
VALID_STATUSES: Final[set[str]] = {"replied", "escalated"}
VALID_REQUEST_TYPES: Final[set[str]] = {
    "product_issue", "feature_request", "bug", "invalid",
}

# ---------------------------------------------------------------------------
# Seed everything
# ---------------------------------------------------------------------------

def seed_all(seed: int = SEED) -> None:
    """Set deterministic seeds for all random sources."""
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    os.environ["PYTHONHASHSEED"] = str(seed)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger with a consistent format."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)-28s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
