"""Layer 4: Heuristic JSON -> BehavioralSignal (OpenAI backend).

Rules enforced here:
- The model only ever sees the dense feature JSON (Layer 3 output) — never raw
  event streams, DOM text, or anything identifying.
- Output is schema-constrained via OpenAI strict structured outputs, then
  re-validated by Pydantic — a malformed response is a retryable error, not
  corrupt data downstream.
- The system prompt is frozen and identical across every request, so OpenAI's
  automatic prompt caching prices the shared prefix at a discount at volume.
"""

import json

from .config import PROFILER_MODEL
from .oai import parse_completion, response_format_for, sync_client
from .schemas import BehavioralSignal, FeaturePayload

# Frozen — keep byte-identical so OpenAI auto-caches the shared prefix.
ANALYST_SYSTEM = """You are a behavioral UX analyst for a simulation platform.

You receive ONLY aggregated, anonymous interaction metrics for one browsing
session segment: scroll kinematics, hesitation counts, rage-click bursts, and
per-zone dwell times. You never see content, text, or anything identifying.

Your job: translate the numbers into SOFT behavioral segment signals that will
seed synthetic research personas. Rules:
- Cite the specific metrics behind every label in `evidence`.
- These are provisional UX segment signals, NOT psychometric, clinical, or
  individual judgments.
- When the data is thin (few events, short segment), say so and report low
  confidence. Never manufacture certainty.
- `price_sensitivity_signal` may only be non-"unknown" when pricing-zone
  metrics actually exist in the input.
- A `cognitive_model_parameters` block may be present: fitted parameters from
  drift-diffusion, hidden-Markov state decoding, spectral and foraging models.
  Treat them as the strongest evidence available — they are estimated, not
  guessed — but still frame conclusions as behavioral signals."""


def profile_features(
    features: FeaturePayload, cognition_summary: dict | None = None
) -> BehavioralSignal:
    payload = features.model_dump(
        exclude={"event_stream", "velocity_series", "decision_latencies_ms"}
    )
    body = "Session segment metrics:\n" + json.dumps(payload, sort_keys=True)
    if cognition_summary:
        body += "\n\ncognitive_model_parameters:\n" + json.dumps(
            cognition_summary, sort_keys=True
        )
    completion = sync_client().chat.completions.create(
        model=PROFILER_MODEL,
        messages=[
            {"role": "system", "content": ANALYST_SYSTEM},
            {"role": "user", "content": body},
        ],
        response_format=response_format_for(BehavioralSignal),
    )
    return parse_completion(completion, BehavioralSignal)
