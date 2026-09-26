"""Provider-neutral LLM helpers shared by every workflow.

Pick a backend with environment variables (first match wins):

    LLM_PROVIDER=anthropic | openai | local     explicit choice, or auto-detect:
    ANTHROPIC_API_KEY=...                       -> anthropic (Claude)
    OPENAI_API_KEY=...                          -> openai
    LLM_BASE_URL=http://localhost:11434/v1      -> local (any OpenAI-compatible
                                                   server: Ollama, vLLM, LM Studio,
                                                   llama.cpp)

    LLM_MODEL=...     model name (required for local; optional otherwise)
    LLM_API_KEY=...   key for a local/self-hosted server, if it needs one

The workflows only use `call()`, `call_json()` and `Conversation`, so they run
unchanged on any backend.
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, field
from functools import cache
from typing import Any

DEFAULT_MODELS = {"anthropic": "claude-opus-5", "openai": "gpt-5"}


class RefusalError(RuntimeError):
    """Raised when the model declines the request."""


class ConfigError(RuntimeError):
    """Raised when no usable provider is configured."""


def provider() -> str:
    explicit = os.environ.get("LLM_PROVIDER", "").strip().lower()
    if explicit:
        if explicit not in ("anthropic", "openai", "local"):
            raise ConfigError(f"LLM_PROVIDER must be anthropic, openai or local, not {explicit!r}")
        return explicit
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if os.environ.get("LLM_BASE_URL"):
        return "local"
    raise ConfigError(
        "No LLM configured. Set ANTHROPIC_API_KEY, OPENAI_API_KEY, or "
        "LLM_BASE_URL (+ LLM_MODEL) for a local server. See README.md."
    )


def is_configured() -> bool:
    try:
        provider()
        return True
    except ConfigError:
        return False


def model() -> str:
    name = os.environ.get("LLM_MODEL") or DEFAULT_MODELS.get(provider())
    if not name:
        raise ConfigError("Set LLM_MODEL to the model name your local server serves.")
    return name


@cache
def _client():
    if provider() == "anthropic":
        import anthropic

        return anthropic.Anthropic()
    from openai import OpenAI

    if provider() == "openai":
        return OpenAI(base_url=os.environ.get("LLM_BASE_URL") or None)
    base_url = os.environ.get("LLM_BASE_URL")
    if not base_url:
        raise ConfigError("LLM_PROVIDER=local needs LLM_BASE_URL, e.g. http://localhost:11434/v1")
    # Most local servers ignore the key, but the client requires a non-empty one.
    return OpenAI(base_url=base_url, api_key=os.environ.get("LLM_API_KEY") or "not-needed")


# --------------------------------------------------------------------------- #
# Conversation: one multi-turn exchange, optionally with tools.
# --------------------------------------------------------------------------- #


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict
    error: str | None = None  # set when the model sent unparseable arguments


@dataclass
class Reply:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)


class Conversation:
    """Keeps history in the active provider's native format.

    `tools` use Anthropic's shape ({name, description, input_schema}); they are
    translated for OpenAI-compatible backends.
    """

    def __init__(self, system: str | None = None, tools: list[dict] | None = None,
                 effort: str = "medium"):
        self.provider = provider()
        self.system = system
        self.tools = tools or []
        self.effort = effort
        self.messages: list[dict] = []
        if system and self.provider != "anthropic":
            self.messages.append({"role": "system", "content": system})

    def add_user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def add_tool_results(self, results: list[tuple[str, str, bool]]) -> None:
        """results: (tool_call_id, content, is_error) for every call in the last reply."""
        if self.provider == "anthropic":
            # All results for one turn go back in a single user message.
            self.messages.append({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": cid, "content": content, "is_error": err}
                for cid, content, err in results
            ]})
        else:
            for cid, content, _ in results:
                self.messages.append({"role": "tool", "tool_call_id": cid, "content": content})

    def send(self, schema: dict | None = None) -> Reply:
        if self.provider == "anthropic":
            return self._send_anthropic(schema)
        return self._send_openai(schema)

    def _send_anthropic(self, schema: dict | None) -> Reply:
        output_config: dict[str, Any] = {"effort": self.effort}
        if schema:
            output_config["format"] = {"type": "json_schema", "schema": schema}
        params: dict[str, Any] = {
            "model": model(),
            "max_tokens": 16000,
            "thinking": {"type": "adaptive"},
            "output_config": output_config,
            "messages": self.messages,
            # Server-side fallback: if a safety classifier declines, the API
            # re-runs the request on Anthropic's recommended fallback model.
            "extra_headers": {"anthropic-beta": "server-side-fallback-2026-07-01"},
            "extra_body": {"fallbacks": "default"},
        }
        if self.system:
            params["system"] = self.system
        if self.tools:
            params["tools"] = self.tools
        response = _client().messages.create(**params)
        if response.stop_reason == "refusal":
            details = response.stop_details
            raise RefusalError(f"Request declined (category: {details.category if details else None})")
        # Append the full content (incl. thinking blocks), not just the text.
        self.messages.append({"role": "assistant", "content": response.content})
        return Reply(
            text="".join(b.text for b in response.content if b.type == "text").strip(),
            tool_calls=[ToolCall(b.id, b.name, b.input)
                        for b in response.content if b.type == "tool_use"],
        )

    def _send_openai(self, schema: dict | None) -> Reply:
        # Chat Completions, because it is what local servers implement.
        params: dict[str, Any] = {"model": model(), "messages": self.messages}
        if self.tools:
            params["tools"] = [{"type": "function", "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t["input_schema"],
            }} for t in self.tools]
        if schema:
            params["response_format"] = {"type": "json_schema", "json_schema": {
                "name": "output", "schema": schema, "strict": True}}
        msg = _client().chat.completions.create(**params).choices[0].message
        if getattr(msg, "refusal", None):
            raise RefusalError(f"Request declined: {msg.refusal}")

        calls = []
        for tc in msg.tool_calls or []:
            try:
                calls.append(ToolCall(tc.id, tc.function.name, json.loads(tc.function.arguments or "{}")))
            except json.JSONDecodeError:
                calls.append(ToolCall(tc.id, tc.function.name, {}, "arguments were not valid JSON"))
        assistant: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            assistant["tool_calls"] = [{
                "id": tc.id, "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            } for tc in msg.tool_calls]
        self.messages.append(assistant)
        return Reply(text=(msg.content or "").strip(), tool_calls=calls)


# --------------------------------------------------------------------------- #
# Single-turn helpers used by patterns 1-5.
# --------------------------------------------------------------------------- #


def call(prompt: str, *, system: str | None = None, effort: str = "medium") -> str:
    """Single-turn call that returns plain text."""
    conv = Conversation(system, effort=effort)
    conv.add_user(prompt)
    return conv.send().text


def call_json(prompt: str, schema: dict, *, system: str | None = None,
              effort: str = "medium", retries: int = 1) -> Any:
    """Single-turn call whose output is validated against `schema`.

    Hosted APIs enforce the schema server-side. Some local servers only
    approximate it, so the result is always parsed and checked here, and a
    failure is sent back to the model once to fix.
    """
    conv = Conversation(system, effort=effort)
    conv.add_user(f"{prompt}\n\nRespond with JSON matching this schema:\n{json.dumps(schema)}")
    for attempt in range(retries + 1):
        text = conv.send(schema).text
        try:
            data = _extract_json(text)
            _validate(data, schema)
            return data
        except ValueError as exc:
            if attempt == retries:
                raise ValueError(f"Model did not return valid JSON: {exc}\n---\n{text}") from exc
            conv.add_user(f"That was invalid ({exc}). Reply with only the corrected JSON.")


def _extract_json(text: str) -> Any:
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    candidate = fenced.group(1) if fenced else text
    start = min((i for i in (candidate.find("{"), candidate.find("[")) if i != -1), default=-1)
    if start == -1:
        raise ValueError("no JSON found")
    try:
        return json.JSONDecoder().raw_decode(candidate[start:])[0]
    except json.JSONDecodeError as exc:
        raise ValueError(str(exc)) from exc


_TYPES = {"object": dict, "array": list, "string": str, "boolean": bool,
          "integer": int, "number": (int, float)}


def _validate(value: Any, schema: dict, path: str = "$") -> None:
    """Minimal JSON Schema check: type, enum, required, properties, items."""
    expected = schema.get("type")
    if expected and not isinstance(value, _TYPES[expected]) or (
        expected in ("integer", "number") and isinstance(value, bool)
    ):
        raise ValueError(f"{path} should be {expected}")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{path} must be one of {schema['enum']}")
    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                raise ValueError(f"{path}.{key} is missing")
        for key, sub in schema.get("properties", {}).items():
            if key in value:
                _validate(value[key], sub, f"{path}.{key}")
    if isinstance(value, list) and "items" in schema:
        for i, item in enumerate(value):
            _validate(item, schema["items"], f"{path}[{i}]")


def banner(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}", file=sys.stderr)
