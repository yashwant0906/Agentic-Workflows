"""Pattern 1 - Prompt chaining.

Break a task into fixed sequential steps, each consuming the previous step's
output, with a programmatic gate between steps to catch problems early.

Example: turn a rough topic into a polished technical blog post.
    outline -> gate (does it cover the requirements?) -> draft -> edit

Usage:
    python 01_prompt_chaining.py "Why vector databases matter for RAG"
"""

import sys

from common import banner, call, call_json

GATE_SCHEMA = {
    "type": "object",
    "properties": {
        "passes": {"type": "boolean"},
        "missing": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["passes", "missing"],
    "additionalProperties": False,
}

REQUIREMENTS = [
    "a concrete motivating example",
    "at least one trade-off or limitation",
    "a practical takeaway the reader can act on",
]


def chain(topic: str, max_outline_attempts: int = 2) -> str:
    banner("Step 1: outline")
    outline = call(f"Write a 5-7 point outline for a technical blog post on: {topic}")

    # Gate: verify the outline before spending tokens on a full draft.
    for attempt in range(max_outline_attempts):
        check = call_json(
            f"Outline:\n{outline}\n\nDoes this outline include ALL of: "
            f"{'; '.join(REQUIREMENTS)}? List anything missing.",
            GATE_SCHEMA,
            effort="low",
        )
        if check["passes"]:
            break
        banner(f"Gate failed (attempt {attempt + 1}); revising outline")
        outline = call(
            f"Revise this outline so it also covers: {', '.join(check['missing'])}\n\n{outline}"
        )
    else:
        raise SystemExit("Outline never passed the gate; stopping before drafting.")

    banner("Step 2: draft")
    draft = call(
        f"Write a ~600 word blog post following this outline exactly:\n{outline}",
        effort="high",
    )

    banner("Step 3: edit")
    return call(
        "Edit this post for clarity and concision. Cut filler, keep technical "
        f"accuracy, keep Markdown headings. Return only the post.\n\n{draft}"
    )


if __name__ == "__main__":
    print(chain(" ".join(sys.argv[1:]) or "Why vector databases matter for RAG"))
