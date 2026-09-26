"""Pattern 2 - Routing.

Classify an input first, then send it to a specialised handler. Each route
gets its own prompt (and could get its own model or tools), so one handler's
instructions never dilute another's.

Example: customer-support ticket router.

Usage:
    python 02_routing.py "I was charged twice for my subscription this month"
"""

import sys

from common import banner, call, call_json

ROUTES = {
    "billing": (
        "You are a billing specialist. Acknowledge the charge issue, explain "
        "what information you need (invoice ID, last 4 card digits) and the "
        "refund timeline. Never promise a refund before verification."
    ),
    "technical": (
        "You are a senior support engineer. Ask for reproduction steps, "
        "environment details and error messages, then give the most likely "
        "fix first."
    ),
    "account": (
        "You are an account-security specialist. For login or access issues, "
        "walk through recovery steps and never ask for the user's password."
    ),
    "general": "You are a friendly support agent. Answer concisely.",
}

ROUTER_SCHEMA = {
    "type": "object",
    "properties": {
        "route": {"type": "string", "enum": list(ROUTES)},
        "urgency": {"type": "string", "enum": ["low", "medium", "high"]},
        "reason": {"type": "string"},
    },
    "required": ["route", "urgency", "reason"],
    "additionalProperties": False,
}


def route(ticket: str) -> str:
    banner("Classifying")
    decision = call_json(
        f"Classify this support ticket.\n\n<ticket>\n{ticket}\n</ticket>",
        ROUTER_SCHEMA,
        effort="low",  # classification is cheap; save effort for the handler
    )
    banner(f"Route: {decision['route']} | urgency: {decision['urgency']} | {decision['reason']}")

    return call(
        f"<ticket>\n{ticket}\n</ticket>\n\nWrite the reply to the customer.",
        system=ROUTES[decision["route"]],
    )


if __name__ == "__main__":
    print(route(" ".join(sys.argv[1:]) or "I was charged twice for my subscription this month"))
