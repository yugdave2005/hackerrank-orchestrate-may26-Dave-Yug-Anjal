"""CLI entry point — runs the support triage pipeline on a CSV of tickets."""

import argparse
import logging
import sys
from pathlib import Path

# Ensure code/ is on Python path so sub-packages resolve
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent.triage_agent import TriageAgent
from retrieval.corpus_loader import load_corpus
from retrieval.retriever import HybridRetriever
from utils.config import (
    DEFAULT_INPUT_CSV,
    DEFAULT_OUTPUT_CSV,
    seed_all,
    setup_logging,
)
from utils.csv_handler import read_tickets, write_output

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(
        description="AI Support Triage Agent — process support tickets and generate responses.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help=f"Path to input CSV (default: {DEFAULT_INPUT_CSV})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help=f"Path to output CSV (default: {DEFAULT_OUTPUT_CSV})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process only the first 5 rows (for testing)",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point: load corpus, initialise agent, process tickets."""
    args = parse_args()

    # --- Setup ---
    setup_logging()
    seed_all()
    logger.info("=" * 60)
    logger.info("AI Support Triage Agent — Starting")
    logger.info("=" * 60)

    # --- Load corpus and build retriever ---
    logger.info("Loading corpus from data/ …")
    corpus = load_corpus()
    retriever = HybridRetriever(corpus)

    # --- Initialise triage agent ---
    agent = TriageAgent(retriever)

    # --- Read tickets ---
    logger.info("Reading tickets from %s", args.input)
    df = read_tickets(args.input)

    if args.dry_run:
        logger.info("DRY RUN mode — processing only first 5 rows")
        df = df.head(5)

    # --- Process each ticket ---
    results: list[dict[str, str]] = []

    try:
        from tqdm import tqdm
        iterator = tqdm(df.iterrows(), total=len(df), desc="Processing tickets")
    except ImportError:
        logger.warning("tqdm not installed — running without progress bar")
        iterator = df.iterrows()

    for idx, row in iterator:
        ticket_num = idx + 1
        issue = str(row.get("Issue", ""))
        subject = str(row.get("Subject", ""))
        company = str(row.get("Company", ""))

        logger.info("--- Ticket %d/%d ---", ticket_num, len(df))
        logger.info("Issue: %.80s…", issue)

        try:
            result = agent.process_ticket(
                issue=issue,
                subject=subject,
                company=company,
            )
            results.append(result)
            logger.info(
                "Result: status=%s, type=%s, area=%s",
                result["status"],
                result["request_type"],
                result["product_area"],
            )

        except Exception as exc:
            # Per-row error handling: escalate with error justification
            logger.error("Error processing ticket %d: %s", ticket_num, exc)
            results.append({
                "status": "escalated",
                "product_area": "General Support",
                "response": (
                    "We apologize for the inconvenience. Your request has been "
                    "recorded and a support agent will follow up shortly."
                ),
                "justification": f"Processing error: {exc}. Escalated for human review.",
                "request_type": "product_issue",
            })

    # --- Write output ---
    write_output(results, args.output)

    logger.info("=" * 60)
    logger.info("Done! Processed %d tickets → %s", len(results), args.output)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
