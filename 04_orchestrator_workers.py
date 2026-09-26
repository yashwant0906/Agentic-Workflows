"""Pattern 4 - Orchestrator-workers.

Unlike parallelization, the subtasks are not known in advance: an
orchestrator decides them at runtime from the input, workers execute them
concurrently, and a synthesizer merges the results.

Example: produce a research brief on any question.

Usage:
    python 04_orchestrator_workers.py "Should a 10-person startup adopt Kubernetes?"
"""

import sys
from concurrent.futures import ThreadPoolExecutor

from common import banner, call, call_json

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "subtasks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "instructions": {"type": "string"},
                },
                "required": ["title", "instructions"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["subtasks"],
    "additionalProperties": False,
}


def research(question: str, max_workers: int = 5) -> str:
    banner("Orchestrator: planning")
    plan = call_json(
        f"Question: {question}\n\nBreak this into 3-5 independent research "
        "subtasks that together answer it. Each must be answerable on its own.",
        PLAN_SCHEMA,
        effort="high",
    )
    subtasks = plan["subtasks"][:max_workers]
    for t in subtasks:
        banner(f"  planned: {t['title']}")

    banner(f"Workers: running {len(subtasks)} subtasks in parallel")

    def work(task: dict) -> str:
        findings = call(
            f"Overall question: {question}\n\nYour subtask: {task['title']}\n"
            f"{task['instructions']}\n\nAnswer in <=200 words. Flag uncertainty.",
            effort="low",
        )
        return f"### {task['title']}\n{findings}"

    with ThreadPoolExecutor() as pool:
        sections = list(pool.map(work, subtasks))

    banner("Synthesizer: merging")
    return call(
        f"Question: {question}\n\nWorker findings:\n\n" + "\n\n".join(sections) + "\n\n"
        "Write a decision brief: a one-paragraph answer first, then key "
        "reasons, then open risks. Resolve contradictions between workers explicitly.",
        effort="high",
    )


if __name__ == "__main__":
    print(research(" ".join(sys.argv[1:]) or "Should a 10-person startup adopt Kubernetes?"))
