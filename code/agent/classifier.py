"""Classifier — determines request_type and product_area for support tickets."""

import logging
from typing import Any

import anthropic

from utils.config import (
    CLAUDE_MAX_TOKENS,
    CLAUDE_MODEL,
    CLAUDE_TEMPERATURE,
    get_anthropic_api_key,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Domain → typical product areas
# ---------------------------------------------------------------------------
PRODUCT_AREA_HINTS: dict[str, list[str]] = {
    "hackerrank": [
        "Assessments", "Screen", "Interview", "Library", "Billing",
        "Account Access", "API", "Integrations", "Subscription",
        "Community", "SkillUp", "Certification", "Settings",
    ],
    "claude": [
        "Conversation Management", "Privacy", "Account Access", "Billing",
        "API", "Safety", "Desktop App", "Mobile App", "Team Management",
        "Enterprise", "Education", "Connectors", "Claude Code", "Bedrock",
    ],
    "visa": [
        "Card Disputes", "Payments", "Travel Support", "General Support",
        "Fraud", "Account Access", "Card Benefits", "Merchant Issues",
    ],
}

CLASSIFY_SYSTEM_PROMPT = """You are a support ticket classifier. Classify the ticket into:

1. **request_type** — one of: product_issue, feature_request, bug, invalid
   - "invalid" = spam, gibberish, offensive, completely out of scope, or irrelevant to support
   - "bug" = something is broken / not working as expected
   - "feature_request" = user wants something new added
   - "product_issue" = legitimate support question about existing behavior

2. **product_area** — the most relevant support category. Choose from common areas like:
   Assessments, Screen, Interview, Billing, Account Access, API, Subscription,
   Community, Certification, Conversation Management, Privacy, Safety, Team Management,
   Card Disputes, Payments, Travel Support, General Support, Fraud, etc.

3. **inferred_domain** — if company is None or empty, infer which domain this belongs to:
   hackerrank, claude, visa, or unknown.

Respond in EXACTLY this format (no markdown, no extra text):
request_type: <value>
product_area: <value>
inferred_domain: <value>
"""


def classify_ticket(
    issue: str,
    subject: str,
    company: str,
) -> dict[str, str]:
    """Classify a support ticket's request_type and product_area.

    Args:
        issue: The main ticket body.
        subject: The ticket subject line.
        company: The company name (HackerRank, Claude, Visa, or empty).

    Returns:
        Dict with keys: request_type, product_area, inferred_domain.
    """
    user_msg = (
        f"Company: {company or 'None'}\n"
        f"Subject: {subject or '(no subject)'}\n"
        f"Issue:\n{issue}"
    )

    try:
        client = anthropic.Anthropic(api_key=get_anthropic_api_key())
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=256,
            temperature=CLAUDE_TEMPERATURE,
            system=CLASSIFY_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        text = response.content[0].text.strip()
        return _parse_classification(text, company)

    except anthropic.APIError as exc:
        logger.error("Claude API error during classification: %s", exc)
        return {
            "request_type": "product_issue",
            "product_area": "General Support",
            "inferred_domain": _guess_domain(company),
        }


def _parse_classification(text: str, company: str) -> dict[str, str]:
    """Parse the structured classifier response.

    Args:
        text: Raw text from the Claude response.
        company: Original company field for fallback.

    Returns:
        Parsed classification dict.
    """
    result: dict[str, str] = {
        "request_type": "product_issue",
        "product_area": "General Support",
        "inferred_domain": _guess_domain(company),
    }

    for line in text.splitlines():
        line = line.strip()
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip().lower().replace(" ", "_")
        value = value.strip()

        if key == "request_type" and value in {
            "product_issue", "feature_request", "bug", "invalid",
        }:
            result["request_type"] = value
        elif key == "product_area":
            result["product_area"] = value
        elif key == "inferred_domain":
            result["inferred_domain"] = value.lower()

    return result


def _guess_domain(company: str) -> str:
    """Best-effort domain guess from the company field.

    Args:
        company: The company string.

    Returns:
        Domain string (hackerrank, claude, visa, or unknown).
    """
    c = company.lower().strip() if company else ""
    if "hackerrank" in c:
        return "hackerrank"
    if "claude" in c:
        return "claude"
    if "visa" in c:
        return "visa"
    return "unknown"
