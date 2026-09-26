"""Pattern 5 - Evaluator-optimizer.

One call generates, another evaluates against explicit criteria, and the
feedback loops back until the evaluator passes it or the budget runs out.
Works when there are clear success criteria and iteration measurably helps.

Example: write a Python function and iterate until a reviewer accepts it.

Usage:
    python 05_evaluator_optimizer.py "parse ISO-8601 durations like P3DT4H into seconds"
"""

import sys

from common import banner, call, call_json

CRITERIA = [
    "correct for all valid inputs, including edge cases",
    "raises ValueError with a helpful message on invalid input",
    "has type hints and a docstring",
    "uses only the standard library",
]

EVAL_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["PASS", "NEEDS_WORK"]},
        "feedback": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["verdict", "feedback"],
    "additionalProperties": False,
}


def optimize(task: str, max_rounds: int = 4) -> str:
    criteria = "\n".join(f"- {c}" for c in CRITERIA)
    code = call(
        f"Write a Python function to {task}.\nRequirements:\n{criteria}\n"
        "Return only the code in one ```python block.",
        effort="high",
    )

    for round_no in range(1, max_rounds + 1):
        result = call_json(
            f"Evaluate this code strictly against the requirements.\n"
            f"Requirements:\n{criteria}\n\n{code}\n\n"
            "Mentally run edge cases. PASS only if every requirement is met.",
            EVAL_SCHEMA,
            effort="high",
            system="You are a meticulous senior reviewer. Be specific; no praise.",
        )
        banner(f"Round {round_no}: {result['verdict']}")
        if result["verdict"] == "PASS":
            return code

        feedback = "\n".join(f"- {f}" for f in result["feedback"])
        print(feedback, file=sys.stderr)
        code = call(
            f"Task: {task}\nRequirements:\n{criteria}\n\nCurrent code:\n{code}\n\n"
            f"Reviewer feedback:\n{feedback}\n\nFix every point. Return only the "
            "code in one ```python block.",
            effort="high",
        )

    banner(f"Stopped after {max_rounds} rounds without a PASS - returning best effort")
    return code


if __name__ == "__main__":
    print(optimize(" ".join(sys.argv[1:]) or "parse ISO-8601 durations like P3DT4H into seconds"))
