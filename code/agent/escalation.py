"""Escalation logic — decides whether a ticket should be escalated or replied to."""

import logging
import re

from retrieval.retriever import RetrievalResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Escalation keyword / pattern sets
# ---------------------------------------------------------------------------

# High-urgency signals → always escalate
URGENCY_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bfraud\b",
        r"\bunauthori[sz]ed\s+transaction",
        r"\baccount\s+(hack|compromise|breach|stolen)",
        r"\bmy\s+account\s+was\s+hack",
        r"\bidentity\s+(theft|stolen)\b",
        r"\bcharge\s+I\s+don'?t\s+recogni[sz]e",
        r"\blegal\s+(action|threat|notice|complaint)",
        r"\blawsuit\b",
        r"\bregulator",
        r"\bGDPR\b",
        r"\bDPDP\b",
        r"\bdata\s+(removal|deletion|erasure)\b",
        r"\bdelete\s+(my|all)\s+(account|data)",
        r"\bchargeback",
        r"\bbilling\s+dispute",
        r"\bsecurity\s+vulnerability",
        r"\bcredential\s+(leak|expos)",
        r"\bpassword\s+(leak|expos|breach)",
        r"\bharass(ment)?\b",
        r"\bthreat(s|en)?\b",
        r"\babuse\b",
        r"\bcheating\b",
        r"\bscore\s+manipulat",
        r"\bassessment\s+integrity",
        r"\burgent(ly)?\b",
        r"\bimmediately\b",
        r"\bsite\s+is\s+down\b",
        r"\bcompletely\s+(down|broken|failing)\b",
        r"\bstopped\s+working\s+completely\b",
        r"\ball\s+requests?\s+(are\s+)?failing\b",
        r"\brefund\b",
        r"\bbug\s+bounty\b",
        r"\bsecurity\b.*\bvulnerability\b",
    ]
]

# Feature / FAQ patterns → typically reply-safe
REPLY_SAFE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bhow\s+(do|can|to)\b",
        r"\bwhat\s+is\b",
        r"\bwhere\s+(can|do)\b",
        r"\bstep(s|-by-step)\b",
        r"\bguide\b",
        r"\btutorial\b",
        r"\bfeature\s+request\b",
        r"\bwould\s+like\s+to\s+see\b",
        r"\bcan\s+you\s+add\b",
    ]
]

# Minimum relevance score threshold — below this, corpus has no answer
MIN_RETRIEVAL_SCORE: float = 0.01


def should_escalate(
    issue: str,
    subject: str,
    request_type: str,
    retrieval_results: list[RetrievalResult],
) -> tuple[bool, str]:
    """Decide whether a ticket must be escalated to a human.

    Args:
        issue: The main ticket body.
        subject: The ticket subject line.
        request_type: Already-classified request type.
        retrieval_results: Retrieved corpus chunks with scores.

    Returns:
        Tuple of (should_escalate: bool, reason: str).
    """
    combined_text = f"{subject} {issue}".lower()

    # 1. Check urgency / escalation patterns
    for pattern in URGENCY_PATTERNS:
        match = pattern.search(combined_text)
        if match:
            reason = f"Escalation trigger matched: '{match.group()}'"
            logger.info("ESCALATE — %s", reason)
            return True, reason

    # 2. Check if corpus has NO relevant grounding
    if not retrieval_results:
        reason = "No relevant corpus documents found — cannot ground a response"
        logger.info("ESCALATE — %s", reason)
        return True, reason

    # Check if all retrieval scores are below threshold
    max_score = max(r.score for r in retrieval_results)
    if max_score < MIN_RETRIEVAL_SCORE:
        reason = (
            f"Best retrieval score ({max_score:.4f}) below threshold "
            f"({MIN_RETRIEVAL_SCORE}) — corpus has no relevant answer"
        )
        logger.info("ESCALATE — %s", reason)
        return True, reason

    # 3. If it's an invalid request, we can still reply (out of scope message)
    # But check if it looks like a prompt injection attempt
    injection_patterns = [
        r"(display|show|reveal|give).*(internal|rules|logic|system\s+prompt)",
        r"ignore\s+(previous|all)\s+(instructions|rules)",
        r"delete\s+all\s+files",
    ]
    for pat_str in injection_patterns:
        if re.search(pat_str, combined_text, re.IGNORECASE):
            reason = f"Possible prompt injection or malicious request detected"
            logger.info("ESCALATE — %s", reason)
            return True, reason

    # 4. Default: reply is safe
    logger.debug("REPLY — no escalation triggers found")
    return False, "No escalation triggers detected; corpus has relevant grounding"
