# AI Support Triage Agent

A production-quality terminal-based support triage agent that processes multi-domain support tickets across HackerRank, Claude, and Visa ecosystems using RAG (Retrieval-Augmented Generation).

## Prerequisites

- Python 3.10+
- pip

## Installation

```bash
pip install -r code/requirements.txt
```

## Setup

```bash
cp .env.example .env
# Edit .env and add your Anthropic API key:
# ANTHROPIC_API_KEY=your_key_here
```

## Run

```bash
# Process all tickets
python code/main.py

# Dry run — process only first 5 rows (for testing)
python code/main.py --dry-run

# Custom input/output paths
python code/main.py --input path/to/tickets.csv --output path/to/output.csv
```

## Architecture Overview

### `retrieval/corpus_loader.py` — Corpus Loading
Walks all files in `data/hackerrank/`, `data/claude/`, and `data/visa/`. Parses markdown documents and chunks them into ~300-token segments with 50-token overlap. Each chunk is tagged with its source domain and file path for provenance tracking.

### `retrieval/embeddings.py` — Embedding Layer
Uses `sentence-transformers` with the `all-MiniLM-L6-v2` model to encode corpus chunks and queries into dense vectors. The model is lazy-loaded to avoid slow startup when embeddings aren't needed. All vectors are L2-normalised for efficient cosine similarity via dot product.

### `retrieval/retriever.py` — Hybrid Retrieval
Combines BM25 keyword search (via `rank_bm25`) with dense semantic similarity search. Rankings are merged using **Reciprocal Rank Fusion (RRF)** to leverage the strengths of both methods. Supports optional domain filtering to restrict results to a specific ecosystem.

### `agent/classifier.py` — Classification
Uses Claude API to classify each ticket into a `request_type` (product_issue / feature_request / bug / invalid) and `product_area` (e.g. Assessments, Privacy, Card Disputes). Also infers the domain when the company field is empty.

### `agent/escalation.py` — Escalation Logic
Rule-based escalation engine using 30+ regex patterns covering fraud, legal threats, GDPR requests, security vulnerabilities, abuse, assessment integrity, and urgency signals. Also escalates when the corpus has no relevant grounding (low retrieval score) — the agent never guesses.

### `agent/response_generator.py` — Response Generation
Uses Claude API with a strict system prompt that forces responses to be grounded exclusively in retrieved corpus chunks. Generates both a user-facing `response` and an internal `justification`. Separate prompts for reply vs. escalation scenarios.

### `agent/triage_agent.py` — Pipeline Orchestration
Chains the full pipeline: classify → retrieve → escalate → generate response. Handles domain resolution and passes context between stages.

### `code/main.py` — CLI Entry Point
Argparse-based CLI with `--dry-run` (first 5 rows), `--input`, and `--output` flags. Uses tqdm for progress tracking and handles per-row errors gracefully by escalating failed tickets rather than crashing.

## Design Decisions

1. **Hybrid retrieval (BM25 + semantic)**: BM25 excels at exact keyword matching (useful for product names, error codes), while semantic search captures meaning. RRF fusion combines both without requiring score calibration.

2. **Escalation-first approach**: The escalation check runs BEFORE response generation. This ensures high-risk tickets (fraud, legal, security) are never answered with potentially incorrect information — they're always routed to humans.

3. **Claude Sonnet for generation**: Claude Sonnet 4 provides the best balance of quality and speed for grounded response generation. Temperature=0 and seed=42 ensure deterministic outputs.

4. **Per-row error handling**: Instead of crashing on a single bad ticket, errors are caught per-row and the ticket is escalated with an error justification. This ensures the output CSV is always complete.

5. **Corpus-only grounding**: The system prompt explicitly prohibits the model from using parametric knowledge. Responses must cite information from the retrieved chunks only, preventing hallucinated policies or fabricated steps.
