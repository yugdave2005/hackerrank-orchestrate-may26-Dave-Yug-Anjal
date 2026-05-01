"""Triage agent — orchestrates the full classify → retrieve → escalate → respond pipeline."""

import logging
from typing import Any

from agent.classifier import classify_ticket
from agent.escalation import should_escalate
from agent.response_generator import generate_response
from retrieval.retriever import HybridRetriever

logger = logging.getLogger(__name__)


class TriageAgent:
    """End-to-end support ticket triage agent.

    For each ticket, the agent:
    1. Classifies request_type and product_area
    2. Retrieves relevant corpus chunks
    3. Evaluates escalation triggers
    4. Generates a grounded response + justification
    """

    def __init__(self, retriever: HybridRetriever) -> None:
        """Initialise the triage agent.

        Args:
            retriever: A fully initialised HybridRetriever.
        """
        self._retriever = retriever
        logger.info("TriageAgent initialised.")

    def process_ticket(self, issue: str, subject: str, company: str) -> dict[str, str]:
        """Process a single support ticket through the full pipeline.

        Args:
            issue: The ticket body text.
            subject: The ticket subject line.
            company: Company name (HackerRank, Claude, Visa, or empty).

        Returns:
            Dict with keys: status, product_area, response, justification, request_type.
        """
        # --- Step 1: Classify ---
        classification = classify_ticket(issue, subject, company)
        request_type = classification["request_type"]
        product_area = classification["product_area"]
        inferred_domain = classification["inferred_domain"]

        logger.info(
            "Classified: type=%s, area=%s, domain=%s",
            request_type, product_area, inferred_domain,
        )

        # --- Step 2: Determine domain for retrieval filtering ---
        domain_filter = _resolve_domain(company, inferred_domain)

        # --- Step 3: Retrieve ---
        query = f"{subject} {issue}".strip()
        retrieval_results = self._retriever.retrieve(
            query=query,
            domain_filter=domain_filter,
        )
        logger.info("Retrieved %d chunks", len(retrieval_results))

        # --- Step 4: Escalation check ---
        is_escalated, escalation_reason = should_escalate(
            issue=issue,
            subject=subject,
            request_type=request_type,
            retrieval_results=retrieval_results,
        )
        status = "escalated" if is_escalated else "replied"
        logger.info("Escalation decision: %s (%s)", status, escalation_reason)

        # --- Step 5: Generate response ---
        gen_result = generate_response(
            issue=issue,
            subject=subject,
            company=company,
            request_type=request_type,
            product_area=product_area,
            retrieval_results=retrieval_results,
            is_escalated=is_escalated,
            escalation_reason=escalation_reason,
        )

        return {
            "status": status,
            "product_area": product_area,
            "response": gen_result["response"],
            "justification": gen_result["justification"],
            "request_type": request_type,
        }


def _resolve_domain(company: str, inferred_domain: str) -> str | None:
    """Resolve the domain filter for retrieval.

    Args:
        company: Explicit company field from the ticket.
        inferred_domain: Domain inferred by the classifier.

    Returns:
        Domain string for filtering, or None for no filter.
    """
    c = company.lower().strip() if company else ""

    if "hackerrank" in c:
        return "hackerrank"
    if "claude" in c:
        return "claude"
    if "visa" in c:
        return "visa"

    # Fall back to classifier inference
    if inferred_domain in {"hackerrank", "claude", "visa"}:
        return inferred_domain

    # Unknown domain — search all
    return None
