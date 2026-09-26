"""Fail CI if an SDK release drops a parameter or field that common.py relies on.

The workflows only hit the network when run, so plain imports can't catch API
drift. This checks the installed SDKs' signatures and response models instead.
"""

import inspect

import anthropic
import openai
from anthropic.types import Message
from openai.types.chat import ChatCompletionMessage


def require(names: set[str], available, where: str) -> None:
    missing = names - set(available)
    assert not missing, f"{where} is missing {sorted(missing)}"


require(
    {"model", "max_tokens", "messages", "system", "tools", "thinking",
     "output_config", "extra_headers", "extra_body"},
    inspect.signature(anthropic.Anthropic(api_key="x").messages.create).parameters,
    "anthropic messages.create",
)
require({"content", "stop_reason", "stop_details"}, Message.model_fields, "anthropic Message")

require(
    {"model", "messages", "tools", "response_format"},
    inspect.signature(openai.OpenAI(api_key="x").chat.completions.create).parameters,
    "openai chat.completions.create",
)
require({"content", "refusal", "tool_calls"}, ChatCompletionMessage.model_fields,
        "openai ChatCompletionMessage")

print(f"OK: anthropic {anthropic.__version__}, openai {openai.__version__}")
