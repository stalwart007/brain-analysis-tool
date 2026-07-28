"""Batch-mode swarm on the OpenAI Batch API.

Large non-interactive runs (thousands of twins) go through OpenAI's file-based
batch flow — 50% of standard token price, 24h completion window — where
wall-clock latency doesn't matter:

    build JSONL requests -> upload file -> create batch -> poll -> download -> aggregate

Request construction and per-line result parsing are pure functions
(unit-tested); `execute_batch_swarm` is the async job executor.
"""

import asyncio
import json
from typing import Any, Literal

from .config import LOAD_TO_TEMPERATURE, TWIN_MODEL
from .oai import async_client, response_format_for, strict_schema
from .schemas import PersonaSeed, SwarmAggregate, TwinReaction
from .swarm import SIMULATION_PREAMBLE, _LOAD_CONDITIONING, aggregate_reactions

POLL_INTERVAL_S = 15
MAX_WAIT_S = 60 * 60  # batches usually finish within the hour
_TERMINAL = {"completed", "failed", "expired", "cancelled"}


def build_batch_requests(
    personas: list[PersonaSeed],
    scenario: str,
    twins_per_persona: int,
    cognitive_load: Literal["low", "medium", "high"],
) -> list[dict[str, Any]]:
    """One JSONL line per twin, in OpenAI Batch API shape. Same prompt layering
    as the live fan-out, so OpenAI prompt caching applies across the batch too."""
    response_format = response_format_for(TwinReaction)
    temperature = LOAD_TO_TEMPERATURE[cognitive_load]
    requests: list[dict[str, Any]] = []
    for p_index, persona in enumerate(personas):
        for t_index in range(twins_per_persona):
            requests.append(
                {
                    "custom_id": f"twin-{p_index}-{t_index}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": TWIN_MODEL,
                        "temperature": temperature,
                        "response_format": response_format,
                        "messages": [
                            {"role": "system", "content": SIMULATION_PREAMBLE},
                            {"role": "system", "content": f"STIMULUS UNDER TEST:\n{scenario}"},
                            {
                                "role": "system",
                                "content": persona.system_prompt
                                + "\n\nCurrent state: "
                                + _LOAD_CONDITIONING[cognitive_load],
                            },
                            {
                                "role": "user",
                                "content": (
                                    "Experience the stimulus as this persona and report "
                                    "your reaction. Write inner_monologue first, in "
                                    "character, then decide the rest."
                                ),
                            },
                        ],
                    },
                }
            )
    return requests


def parse_batch_line(entry: dict[str, Any]) -> TwinReaction | None:
    """Parse one line of the batch output JSONL into a TwinReaction; None on any
    failure (non-200, refusal, malformed JSON)."""
    try:
        response = entry.get("response") or {}
        if response.get("status_code") != 200:
            return None
        message = response["body"]["choices"][0]["message"]
        if message.get("refusal"):
            return None
        content = message.get("content")
        if not content:
            return None
        return TwinReaction.model_validate_json(content)
    except Exception:
        return None


async def execute_batch_swarm(
    personas: list[PersonaSeed],
    scenario: str,
    twins_per_persona: int,
    cognitive_load: Literal["low", "medium", "high"],
) -> SwarmAggregate:
    client = async_client()
    requests = build_batch_requests(personas, scenario, twins_per_persona, cognitive_load)
    jsonl = "\n".join(json.dumps(r) for r in requests).encode("utf-8")

    upload = await client.files.create(
        file=("swarm_batch.jsonl", jsonl), purpose="batch"
    )
    batch = await client.batches.create(
        input_file_id=upload.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )

    waited = 0
    while batch.status not in _TERMINAL:
        if waited >= MAX_WAIT_S:
            raise RuntimeError(f"Batch {batch.id} did not finish within {MAX_WAIT_S}s.")
        await asyncio.sleep(POLL_INTERVAL_S)
        waited += POLL_INTERVAL_S
        batch = await client.batches.retrieve(batch.id)

    if batch.status != "completed" or not batch.output_file_id:
        raise RuntimeError(f"Batch {batch.id} ended with status '{batch.status}'.")

    content = await client.files.content(batch.output_file_id)
    text = content.text

    reactions: list[TwinReaction] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        reaction = parse_batch_line(json.loads(line))
        if reaction is not None:
            reactions.append(reaction)

    return aggregate_reactions(reactions, scenario, cognitive_load)


# re-exported for tests / callers that build schemas directly
__all__ = [
    "build_batch_requests",
    "parse_batch_line",
    "execute_batch_swarm",
    "strict_schema",
]
