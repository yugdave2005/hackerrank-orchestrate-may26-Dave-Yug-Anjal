"""Response generator — produces grounded responses using Claude API."""

import logging

import anthropic

from retrieval.retriever import RetrievalResult
from utils.config import (
    CLAUDE_MAX_TOKENS,
    CLAUDE_MODEL,
    CLAUDE_TEMPERATURE,
    get_anthropic_api_key,
)

logger = logging.getLogger(__name__)

RESPONSE_SYSTEM_PROMPT = """You are a professional customer support agent. Your task is to help the user based ONLY on the support documentation provided below.

CRITICAL RULES:
1. Answer ONLY from the retrieved corpus chunks provided in the <context> tags.
2. NEVER fabricate policies, steps, URLs, phone numbers, or information not explicitly stated in the provided context.
3. If the corpus does not contain enough information to answer fully, clearly state what you can answer from the corpus and acknowledge what you cannot.
4. Be concise, professional, and user-facing.
5. Do not mention internal systems, retrieval processes, or that you are an AI.
6. Format your response clearly with numbered steps if providing instructions.
7. If the ticket is about something completely unrelated to the support domains (HackerRank, Claude, Visa), politely say it is out of scope.

You will also produce a JUSTIFICATION — a 1-2 sentence internal note explaining why you chose this response and which corpus documents grounded it.

Respond in EXACTLY this format (no markdown fences):
RESPONSE:
<your user-facing response>

JUSTIFICATION:
<1-2 sentence internal justification>
"""

ESCALATION_SYSTEM_PROMPT = """You are a professional customer support agent. This ticket has been flagged for escalation to a human agent.

Write a brief, empathetic response acknowledging the user's concern and letting them know a human agent will follow up. Be specific about what you understood their issue to be.

Also produce a JUSTIFICATION — a 1-2 sentence internal note explaining the escalation reason.

Respond in EXACTLY this format (no markdown fences):
RESPONSE:
<your user-facing response>

JUSTIFICATION:
<1-2 sentence internal justification with escalation reason>
"""


def generate_response(
    issue: str,
    subject: str,
    company: str,
    request_type: str,
    product_area: str,
    retrieval_results: list[RetrievalResult],
    is_escalated: bool,
    escalation_reason: str,
) -> dict[str, str]:
    """Generate a grounded response and justification for a support ticket.

    Args:
        issue: The main ticket body.
        subject: The ticket subject line.
        company: Company name or empty.
        request_type: Classified request type.
        product_area: Classified product area.
        retrieval_results: Retrieved corpus chunks.
        is_escalated: Whether this ticket is being escalated.
        escalation_reason: Reason for escalation (if applicable).

    Returns:
        Dict with keys: response, justification.
    """
    # Build context from retrieval results
    context_parts: list[str] = []
    for i, result in enumerate(retrieval_results, 1):
        context_parts.append(
            f"[Source {i}: {result.chunk.source_file} (domain: {result.chunk.domain})]"
            f"\n{result.chunk.text}\n"
        )
    context_str = "\n".join(context_parts) if context_parts else "(No relevant documents found)"

    user_msg = (
        f"Company: {company or 'None'}\n"
        f"Subject: {subject or '(no subject)'}\n"
        f"Request Type: {request_type}\n"
        f"Product Area: {product_area}\n"
    )

    if is_escalated:
        user_msg += f"Escalation Reason: {escalation_reason}\n"

    user_msg += (
        f"\n<context>\n{context_str}\n</context>\n\n"
        f"Issue:\n{issue}"
    )

    system_prompt = ESCALATION_SYSTEM_PROMPT if is_escalated else RESPONSE_SYSTEM_PROMPT

    try:
        client = anthropic.Anthropic(api_key=get_anthropic_api_key())
        api_response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=CLAUDE_MAX_TOKENS,
            temperature=CLAUDE_TEMPERATURE,
            system=system_prompt,
            messages=[{"role": "user", "content": user_msg}],
        )
        text = api_response.content[0].text.strip()
        return _parse_response(text, is_escalated, escalation_reason)

    except anthropic.APIError as exc:
        logger.error("Claude API error during response generation: %s", exc)
        return {
            "response": (
                "We apologize for the inconvenience. Your request has been "
                "recorded and a support agent will follow up shortly."
            ),
            "justification": f"API error occurred: {exc}. Defaulting to escalation response.",
        }


def _parse_response(
    text: str,
    is_escalated: bool,
    escalation_reason: str,
) -> dict[str, str]:
    """Parse Claude's structured response into response + justification.

    Args:
        text: Raw response text from Claude.
        is_escalated: Whether the ticket is escalated.
        escalation_reason: Escalation reason for fallback justification.

    Returns:
        Dict with 'response' and 'justification' keys.
    """
    response = ""
    justification = ""

    # Split on RESPONSE: and JUSTIFICATION: markers
    if "RESPONSE:" in text and "JUSTIFICATION:" in text:
        parts = text.split("JUSTIFICATION:")
        response_part = parts[0]
        justification = parts[1].strip() if len(parts) > 1 else ""

        # Remove the RESPONSE: prefix
        if "RESPONSE:" in response_part:
            response = response_part.split("RESPONSE:", 1)[1].strip()
        else:
            response = response_part.strip()
    else:
        # Fallback: use the whole text as response
        response = text
        justification = (
            f"Escalated: {escalation_reason}" if is_escalated
            else "Response generated from corpus context."
        )

    # Ensure non-empty
    if not response.strip():
        response = (
            "Your request has been received and a support agent will follow up shortly."
        )
    if not justification.strip():
        justification = (
            f"Escalated: {escalation_reason}" if is_escalated
            else "Response grounded in retrieved corpus documents."
        )

    return {"response": response, "justification": justification}
