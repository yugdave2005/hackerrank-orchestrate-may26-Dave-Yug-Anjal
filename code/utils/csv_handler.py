"""CSV I/O utilities — read support tickets and write triage output."""

import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Input columns we expect
# ---------------------------------------------------------------------------
INPUT_COLUMNS: list[str] = ["Issue", "Subject", "Company"]

# Output columns in the exact order required by the evaluator
OUTPUT_COLUMNS: list[str] = [
    "status",
    "product_area",
    "response",
    "justification",
    "request_type",
]


def read_tickets(path: Path) -> pd.DataFrame:
    """Read support tickets CSV and return a DataFrame.

    Args:
        path: Path to the input CSV file.

    Returns:
        DataFrame with columns Issue, Subject, Company.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If expected columns are missing.
    """
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")

    df = pd.read_csv(path, encoding="utf-8")
    logger.info("Loaded %d tickets from %s", len(df), path)

    # Normalise column names (strip whitespace)
    df.columns = [c.strip() for c in df.columns]

    missing = [c for c in INPUT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    # Fill NaN with empty strings for text fields
    for col in INPUT_COLUMNS:
        df[col] = df[col].fillna("").astype(str).str.strip()

    return df


def write_output(rows: list[dict[str, Any]], path: Path) -> None:
    """Write triage results to output CSV.

    Args:
        rows: List of dicts, each with keys matching OUTPUT_COLUMNS.
        path: Destination path for the output CSV.
    """
    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)

    # Sanity checks
    for col in OUTPUT_COLUMNS:
        empty_count = (df[col].astype(str).str.strip() == "").sum()
        if empty_count > 0:
            logger.warning("Column '%s' has %d empty values", col, empty_count)

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    logger.info("Wrote %d rows to %s", len(df), path)
