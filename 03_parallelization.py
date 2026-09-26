"""Pattern 3 - Parallelization.

Two flavours, both run concurrently:
  * Sectioning - split a task into independent parts reviewed side by side.
  * Voting     - ask the same question several times and aggregate.

Example: review a code snippet from three independent angles (sectioning),
then vote on whether it is safe to merge.

Usage:
    python 03_parallelization.py path/to/file.py
"""

import sys
from concurrent.futures import ThreadPoolExecutor

from common import banner, call, call_json

REVIEWERS = {
    "security": "Review ONLY for security issues (injection, secrets, unsafe input handling).",
    "correctness": "Review ONLY for logic bugs and unhandled edge cases.",
    "readability": "Review ONLY for naming, structure and maintainability.",
}

VOTE_SCHEMA = {
    "type": "object",
    "properties": {"safe_to_merge": {"type": "boolean"}, "reason": {"type": "string"}},
    "required": ["safe_to_merge", "reason"],
    "additionalProperties": False,
}

SAMPLE = '''\
import sqlite3

def find_user(name):
    conn = sqlite3.connect("app.db")
    rows = conn.execute(f"SELECT * FROM users WHERE name = '{name}'").fetchall()
    return rows[0]
'''


def review(code: str, voters: int = 3) -> str:
    wrapped = f"<code>\n{code}\n</code>"

    banner("Sectioning: 3 focused reviews in parallel")
    with ThreadPoolExecutor() as pool:
        futures = {
            name: pool.submit(call, f"{focus}\nBe brief: bullet points only.\n\n{wrapped}")
            for name, focus in REVIEWERS.items()
        }
        reviews = {name: f.result() for name, f in futures.items()}

    banner(f"Voting: {voters} independent merge decisions")
    with ThreadPoolExecutor() as pool:
        votes = list(
            pool.map(
                lambda _: call_json(f"Is this code safe to merge?\n\n{wrapped}", VOTE_SCHEMA),
                range(voters),
            )
        )
    approvals = sum(v["safe_to_merge"] for v in votes)
    # Conservative aggregation: merging needs a unanimous yes.
    verdict = "APPROVE" if approvals == voters else "REQUEST CHANGES"

    report = [f"## {name.title()}\n{text}" for name, text in reviews.items()]
    report.append(f"## Verdict: {verdict} ({approvals}/{voters} votes to merge)")
    report += [f"- {'yes' if v['safe_to_merge'] else 'no'}: {v['reason']}" for v in votes]
    return "\n\n".join(report)


if __name__ == "__main__":
    source = open(sys.argv[1]).read() if len(sys.argv) > 1 else SAMPLE
    print(review(source))
