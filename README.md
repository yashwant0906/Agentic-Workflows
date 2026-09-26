# Agentic Workflows

Six runnable Python examples of the core building blocks of LLM agent systems,
from fixed pipelines to a fully autonomous tool-using agent. They run on
**Claude, OpenAI, or a local model**, with the same code for all three.

| # | Pattern | Example | Use it when |
|---|---------|---------|-------------|
| 1 | [Prompt chaining](01_prompt_chaining.py) | Topic → outline → **gate** → draft → edit | The task splits cleanly into fixed steps |
| 2 | [Routing](02_routing.py) | Support ticket → classifier → specialist prompt | Inputs fall into distinct categories that need different handling |
| 3 | [Parallelization](03_parallelization.py) | 3 focused code reviews + 3-way merge vote | Independent subtasks, or you want confidence via multiple samples |
| 4 | [Orchestrator-workers](04_orchestrator_workers.py) | Planner decides subtasks → parallel workers → synthesizer | You can't know the subtasks until you see the input |
| 5 | [Evaluator-optimizer](05_evaluator_optimizer.py) | Generate code → strict review → revise, until PASS | Clear success criteria, and iteration measurably helps |
| 6 | [Autonomous agent](06_autonomous_agent.py) | Data-analyst agent with sandboxed file + calculator tools | Open-ended tasks where the model must choose its own steps |

Patterns 1–5 are **workflows**: your code controls the flow. Pattern 6 is an
**agent**: the model controls the flow. Start with the simplest one that works;
agents cost more and are harder to debug.

## Quick start

Requires Python 3.10+.

```bash
pip install -r requirements.txt
```

Then pick **one** backend:

| Backend | Set | Default model |
|---------|-----|---------------|
| Claude (Anthropic) | `ANTHROPIC_API_KEY=sk-ant-...` | `claude-opus-5` |
| OpenAI | `OPENAI_API_KEY=sk-...` | `gpt-5` |
| Local / self-hosted | `LLM_BASE_URL=http://localhost:11434/v1` and `LLM_MODEL=<name>` | none (required) |

If you set more than one, the order above decides which is used. To choose
explicitly, set `LLM_PROVIDER=anthropic|openai|local`. `LLM_MODEL` overrides
the model for any backend. If your self-hosted server needs a key, set
`LLM_API_KEY`.

```bash
python 01_prompt_chaining.py "Why vector databases matter for RAG"
python 02_routing.py "I was charged twice this month"
python 03_parallelization.py path/to/some_file.py
python 04_orchestrator_workers.py "Should a 10-person startup adopt Kubernetes?"
python 05_evaluator_optimizer.py "parse ISO-8601 durations into seconds"
python 06_autonomous_agent.py "Which region had the highest total revenue?"
```

Progress goes to stderr and the final result to stdout, so
`python 04_orchestrator_workers.py "..." > brief.md` works.

### Running a local model

Any server with an OpenAI-compatible `/v1/chat/completions` endpoint works.

**Ollama**

```bash
ollama pull qwen3:1.7b
export LLM_BASE_URL=http://localhost:11434/v1 LLM_MODEL=qwen3:1.7b
```

**vLLM**, e.g. serving [Erawan 1](https://huggingface.co/Airavat-ai/erawan-1):

```bash
vllm serve Airavat-ai/erawan-1 --enable-auto-tool-choice --tool-call-parser hermes
export LLM_BASE_URL=http://localhost:8000/v1 LLM_MODEL=Airavat-ai/erawan-1
```

**LM Studio**: start its local server, then set
`LLM_BASE_URL=http://localhost:1234/v1` and `LLM_MODEL` to the loaded model.

Expectations for small local models:

- **Pattern 6 needs tool calling.** The server has to support the `tools`
  parameter; in vLLM that means the flags shown above.
- **Structured output depends on the server.** Some local servers only
  approximate JSON-schema output. `call_json()` always parses and validates the
  result, and sends one repair request back to the model if it fails.
- **Context windows are short.** Small models often have 4K–8K tokens.
  Patterns 4 and 5 build long prompts, so expect more failures there than
  with hosted models.

## Design notes

- **One interface, three backends.** [`common.py`](common.py) exposes
  `call()`, `call_json()` and a `Conversation` class. Each backend keeps
  history in its native format, so Anthropic's thinking blocks and OpenAI's
  tool-call messages round-trip correctly. OpenAI and local servers go through
  Chat Completions, because that is the API local servers implement.
- **Structured outputs for control flow.** Every decision the code branches on
  (gate pass/fail, route, votes, plans, verdicts) is JSON validated against a
  schema, with no brittle string parsing.
- **Effort matched to the step.** Claude runs classification and gates at
  `low` effort, and drafting, planning and evaluation at `high`. Other
  backends ignore the effort hint.
- **Guardrails on the agent.** It has a step budget, file access confined to
  `sandbox/`, and an AST-based calculator instead of `eval()`. Tool errors,
  including malformed arguments, go back to the model so it can recover.
- **Bounded loops.** The gate, evaluator and JSON-repair loops have attempt
  limits, so a stubborn case ends instead of burning tokens.
- **Refusals are errors.** A declined request raises `RefusalError` instead of
  passing an empty string to the next step. On Claude, refused requests are
  retried on a fallback model server-side first.

## Live agents on this repository

Two of these ideas run on real GitHub events via
[`.github/workflows/`](.github/workflows):

- **Issue triage** ([`issue-triage.yml`](.github/workflows/issue-triage.yml))
  labels new issues and asks for missing details.
- **PR review** ([`pr-review.yml`](.github/workflows/pr-review.yml)) posts a
  review, and updates that same comment on each push instead of adding a new
  one. It uses `pull_request`, not `pull_request_target`, so fork PRs never
  run with your secrets.

To turn them on, add **one** secret under *Settings → Secrets and variables →
Actions*: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, or `LLM_BASE_URL` (a
self-hosted server reachable from GitHub). Optionally, add `LLM_PROVIDER` and
`LLM_MODEL` as repository **variables**. With nothing configured, the agents
skip and stay green.

## Author

Built by [Yashwant](https://github.com/yashwant0906), who also created
[Erawan 1](https://huggingface.co/Airavat-ai/erawan-1), a 1.7B model you can
run these workflows against locally (see *Running a local model* above).
