"""Pattern 6 - Autonomous agent (tool-use loop).

The model decides which tool to call next, observes the result, and repeats
until the task is done. The loop is written out by hand so every step is
visible; guardrails are a step budget and a sandboxed workspace.

Example: a "data analyst" agent that inspects CSV files in ./sandbox and
answers questions about them with a safe calculator.

Usage:
    python 06_autonomous_agent.py "Which region had the highest total revenue?"
"""

import ast
import csv
import json
import operator
import sys
from pathlib import Path

from common import Conversation, banner

SANDBOX = Path(__file__).parent / "sandbox"

TOOLS = [
    {
        "name": "list_files",
        "description": "List the data files available in the workspace.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "read_csv",
        "description": "Read a CSV file from the workspace. Returns up to `limit` rows as JSON.",
        "input_schema": {
            "type": "object",
            "properties": {
                "filename": {"type": "string", "description": "Name from list_files"},
                "limit": {"type": "integer", "description": "Max rows (default 200)"},
            },
            "required": ["filename"],
            "additionalProperties": False,
        },
    },
    {
        "name": "calculate",
        "description": "Evaluate an arithmetic expression, e.g. '1200.5 + 830 * 2'. "
        "Use this instead of mental arithmetic.",
        "input_schema": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
            "additionalProperties": False,
        },
    },
]

_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg,
}


def _safe_eval(node: ast.AST) -> float:
    """Arithmetic only - no names, calls or attribute access (unlike eval())."""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left, right = _safe_eval(node.left), _safe_eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("exponent too large")
        return _OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("only numbers and + - * / ** are allowed")


# Failures a tool can hit on bad model input; reported back instead of crashing.
TOOL_ERRORS = (ValueError, KeyError, TypeError, OSError, SyntaxError,
               ArithmeticError, RecursionError, csv.Error)


def run_tool(name: str, args: dict) -> str:
    if name == "list_files":
        return json.dumps(sorted(p.name for p in SANDBOX.glob("*.csv")))
    if name == "read_csv":
        path = (SANDBOX / args["filename"]).resolve()
        if path.parent != SANDBOX.resolve() or not path.is_file():
            raise ValueError(f"no such file in workspace: {args['filename']}")
        with path.open(newline="") as f:
            rows = list(csv.DictReader(f))
        return json.dumps(rows[: args.get("limit", 200)])
    if name == "calculate":
        return str(_safe_eval(ast.parse(args["expression"], mode="eval")))
    raise ValueError(f"unknown tool: {name}")


def agent(task: str, max_steps: int = 15) -> str:
    conv = Conversation(
        system=(
            "You are a careful data analyst. Explore the workspace with tools, "
            "use `calculate` for every arithmetic step, and cite the numbers "
            "behind your conclusion."
        ),
        tools=TOOLS,
    )
    conv.add_user(task)

    for step in range(1, max_steps + 1):
        reply = conv.send()
        if not reply.tool_calls:
            return reply.text

        results = []
        for call in reply.tool_calls:
            banner(f"Step {step}: {call.name}({json.dumps(call.input)})")
            try:
                if call.error:
                    raise ValueError(call.error)
                results.append((call.id, run_tool(call.name, call.input), False))
            except TOOL_ERRORS as exc:  # report tool errors back so the model can recover
                results.append((call.id, f"Error: {exc}", True))
        conv.add_tool_results(results)

    return f"Stopped: step budget of {max_steps} exhausted before the task finished."


if __name__ == "__main__":
    print(agent(" ".join(sys.argv[1:]) or "Which region had the highest total revenue, and by how much?"))
