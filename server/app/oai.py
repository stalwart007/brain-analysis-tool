"""OpenAI client layer + structured-output helpers.

Two adaptations vs a naive port:

1. Lazy client construction. `OpenAI()` raises immediately if no API key is
   configured, so constructing at import time would break the whole app (and
   the key-free test suite). We defer construction to first use; a missing key
   then surfaces as `openai.OpenAIError` at request time, which the endpoints
   map to a clean 503.

2. Strict-schema stripping. OpenAI structured outputs ("strict" json_schema)
   supports only a subset of JSON Schema and rejects constraint keywords
   (`minimum`, `maxLength`, `pattern`, `default`, …). Our Pydantic models carry
   those for *validation*. So we generate a constraint-free strict schema for
   the wire, and re-apply the real Pydantic validation when parsing the reply.
"""

from typing import Any, Type, TypeVar

from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel

_sync: OpenAI | None = None
_async: AsyncOpenAI | None = None


def sync_client() -> OpenAI:
    global _sync
    if _sync is None:
        _sync = OpenAI()  # raises openai.OpenAIError if no API key resolves
    return _sync


def async_client() -> AsyncOpenAI:
    global _async
    if _async is None:
        _async = AsyncOpenAI()
    return _async


# Keywords OpenAI strict structured outputs does not support — stripped from the
# wire schema (still enforced client-side by Pydantic on the returned JSON).
_UNSUPPORTED = {
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
    "minLength",
    "maxLength",
    "pattern",
    "format",
    "minItems",
    "maxItems",
    "default",
}


def _clean(node: Any) -> Any:
    if isinstance(node, dict):
        cleaned = {k: _clean(v) for k, v in node.items() if k not in _UNSUPPORTED}
        if cleaned.get("type") == "object":
            props = cleaned.get("properties", {})
            # strict mode: every property required, no extras
            cleaned["required"] = list(props.keys())
            cleaned["additionalProperties"] = False
        return cleaned
    if isinstance(node, list):
        return [_clean(v) for v in node]
    return node


def strict_schema(model: Type[BaseModel]) -> dict[str, Any]:
    return _clean(model.model_json_schema())


def response_format_for(model: Type[BaseModel]) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": model.__name__,
            "strict": True,
            "schema": strict_schema(model),
        },
    }


T = TypeVar("T", bound=BaseModel)


class Refusal(RuntimeError):
    """The model declined (safety refusal) — carries the refusal text."""


def parse_completion(completion: Any, model: Type[T]) -> T:
    """Extract and validate the structured payload from a chat completion.
    Raises Refusal on a safety refusal, RuntimeError on empty/unparseable."""
    message = completion.choices[0].message
    if getattr(message, "refusal", None):
        raise Refusal(message.refusal)
    if not message.content:
        raise RuntimeError("Model returned no content.")
    return model.model_validate_json(message.content)
