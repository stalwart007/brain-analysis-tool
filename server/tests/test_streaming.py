"""Tests for the streaming simulation generators.

The per-twin LLM callables are monkeypatched with deterministic stubs, so these
verify the *orchestration*: event ordering, per-agent emission, Thompson-wave
allocation, Bayesian early stopping, per-step walk events, and final aggregates
— without touching the network.
"""

import asyncio

import openai
import pytest

from app import studies, swarm
from app.persona import seed_persona
from app.schemas import BehavioralSignal, TwinObjection, TwinReaction


def _persona(label="p"):
    return seed_persona(
        BehavioralSignal(
            deliberation="medium",
            frustration_signal="none",
            exploration_style="scanner",
            price_sensitivity_signal="unknown",
            friction_hotspot=None,
            evidence="t",
            likely_mindset=label,
            confidence=0.5,
        )
    )


def _reaction(action, intent):
    return TwinReaction(
        engagement=0.5,
        intent_score=intent,
        action=action,
        dropoff_point=None,
        friction_notes="n/a",
        inner_monologue="…",
    )


def collect(gen):
    async def run():
        return [e async for e in gen]

    return asyncio.run(run())


def test_stream_swarm_emits_per_agent_then_done(monkeypatch):
    async def fake_twin(persona, scenario, load, semaphore):
        return _reaction("continue", 0.6)

    monkeypatch.setattr(swarm, "_run_twin", fake_twin)
    events = collect(swarm.stream_swarm([_persona()], "stimulus", 3, "low"))

    assert events[0] == {"type": "start", "total": 3}
    agent_events = [e for e in events if e["type"] == "agent"]
    assert len(agent_events) == 3
    assert events[-1]["type"] == "done"
    assert events[-1]["result"]["twin_count"] == 3


def test_stream_compare_adaptive_early_stops_and_saves_budget(monkeypatch):
    """B always converts, A never — the posterior should become conclusive well
    before the budget is spent, triggering early_stop and reallocation."""

    async def fake_twin(persona, scenario, load, semaphore):
        if "winner" in scenario:
            return _reaction("convert", 0.9)
        return _reaction("abandon", 0.1)

    monkeypatch.setattr(swarm, "_run_twin", fake_twin)
    from app.schemas import ScenarioVariant

    variants = [
        ScenarioVariant(name="A", scenario="loser copy"),
        ScenarioVariant(name="B", scenario="winner copy"),
    ]
    personas = [_persona() for _ in range(10)]
    events = collect(
        swarm.stream_compare(personas, variants, twins_per_persona=10, cognitive_load="low", adaptive=True)
    )

    kinds = [e["type"] for e in events]
    assert kinds[0] == "start" and events[0]["adaptive"] is True
    assert "wave" in kinds and "posterior" in kinds
    stop = next(e for e in events if e["type"] == "early_stop")
    assert stop["saved"] > 0  # stopped before exhausting the 200-agent budget
    done = events[-1]
    assert done["type"] == "done"
    assert done["result"]["winner"] == "B"
    assert done["result"]["bayesian"]["conclusive"] is True
    # Thompson must have routed more agents to the converting arm overall
    agents = [e for e in events if e["type"] == "agent"]
    to_b = sum(1 for a in agents if a["variant"] == "B")
    assert to_b >= len(agents) - to_b


def test_stream_compare_fixed_mode_emits_posterior_checkpoints(monkeypatch):
    async def fake_twin(persona, scenario, load, semaphore):
        return _reaction("continue", 0.5)

    monkeypatch.setattr(swarm, "_run_twin", fake_twin)
    from app.schemas import ScenarioVariant

    variants = [
        ScenarioVariant(name="A", scenario="x"),
        ScenarioVariant(name="B", scenario="y"),
    ]
    events = collect(
        swarm.stream_compare([_persona()] * 4, variants, 4, "low", adaptive=False)
    )
    assert sum(1 for e in events if e["type"] == "agent") == 32  # 4×4 per arm × 2 arms
    assert any(e["type"] == "posterior" for e in events)
    assert events[-1]["type"] == "done"


def test_stream_walkthrough_emits_per_step_events(monkeypatch):
    """Each gate decision must surface as its own event, tagged with the twin."""

    async def fake_walk(persona, steps, load, semaphore, on_step=None):
        for i, action in enumerate(["continue", "abandon"]):
            if on_step:
                on_step({"step_index": i, "action": action, "engagement": 0.4})
        return {
            "persona_label": persona.label,
            "outcome": "abandoned",
            "records": [
                {"step_index": 0, "step": steps[0], "action": "continue", "engagement": 0.4,
                 "friction_notes": "n/a", "inner_monologue": "…"},
                {"step_index": 1, "step": steps[1], "action": "abandon", "engagement": 0.4,
                 "friction_notes": "n/a", "inner_monologue": "…"},
            ],
        }

    monkeypatch.setattr(swarm, "_walk_twin", fake_walk)
    events = collect(
        swarm.stream_walkthrough([_persona(), _persona()], ["landing", "form"], 1, "high")
    )
    steps_events = [e for e in events if e["type"] == "step"]
    assert len(steps_events) == 4  # 2 twins × 2 steps
    assert {e["step_index"] for e in steps_events} == {0, 1}
    assert sum(1 for e in events if e["type"] == "twin_done") == 2
    done = events[-1]
    assert done["type"] == "done"
    assert done["result"]["survival"]["worst_step_index"] == 1


def test_stream_walkthrough_terminates_when_a_twin_raises(monkeypatch):
    """A twin that raises must not hang the stream.

    `stream_walkthrough` is the only streaming endpoint that counts its own
    completions instead of letting exceptions propagate through `as_completed`.
    If `pending` is decremented solely on the `twin_done` event enqueued after a
    successful `_walk_twin`, any exception skips the enqueue, `pending` never
    reaches zero, and `await queue.get()` blocks forever — the HTTP connection
    stays open until the client gives up, with orphaned tasks running behind it.

    The exception chosen here is the one that bites in practice:
    `openai.OpenAIError` is what `AsyncOpenAI()` raises when no API key
    resolves, and it is the ROOT of the openai exception hierarchy, not a
    subclass of `openai.APIError` — so per-twin `except (openai.APIError, …)`
    handlers do not catch it. Stubbing `_walk_twin` with a coroutine that always
    succeeds makes this failure path structurally unreachable, so the stub here
    always raises.
    """
    calls = {"n": 0}

    async def exploding_walk(persona, steps, load, semaphore, on_step=None):
        calls["n"] += 1
        raise openai.OpenAIError("no API key configured")

    monkeypatch.setattr(swarm, "_walk_twin", exploding_walk)

    events: list[dict] = []

    async def drain():
        gen = swarm.stream_walkthrough(
            [_persona(), _persona()], ["landing", "form"], 1, "high"
        )
        async for e in gen:
            events.append(e)

    # The generator must TERMINATE. When every twin fails there is nothing to
    # aggregate, so `aggregate_walks` raises — and that is the correct outcome:
    # `_sse` in main.py catches RuntimeError and sends the client an `error`
    # frame. The bug was never the exception, it was that control never got here
    # at all.
    with pytest.raises(RuntimeError, match="no valid twin traces"):
        asyncio.run(asyncio.wait_for(drain(), timeout=5.0))

    assert calls["n"] == 2, "both twins should have been attempted"
    # every twin still reported in before the stream ended, marked failed
    twin_done = [e for e in events if e["type"] == "twin_done"]
    assert len(twin_done) == 2
    assert {e["outcome"] for e in twin_done} == {"failed"}


def test_stream_price_emits_running_demand(monkeypatch):
    async def fake_price(persona, product, prices, load, semaphore):
        return [0.8, 0.4]

    monkeypatch.setattr(studies, "_price_twin", fake_price)
    events = collect(
        studies.stream_price_sensitivity([_persona()] * 3, "widget", [10.0, 20.0], 1, "low")
    )
    agents = [e for e in events if e["type"] == "agent"]
    assert len(agents) == 3
    assert agents[-1]["running_demand"] == [0.8, 0.4]  # constant stubs ⇒ mean = stub
    assert events[-1]["type"] == "done"
    assert events[-1]["result"]["recommended_price"] in (10.0, 20.0)


def test_stream_objection_emits_agents_then_clustering_phase(monkeypatch):
    async def fake_objection(persona, pitch, load, semaphore):
        return TwinObjection(
            sentiment="negative",
            biggest_objection="Too expensive",
            is_dealbreaker=True,
            would_recommend=0.2,
        )

    async def no_embeddings(texts):
        return None  # force exact-grouping fallback, no network

    monkeypatch.setattr(studies, "_objection_twin", fake_objection)
    monkeypatch.setattr(studies, "embed_texts", no_embeddings)
    events = collect(studies.stream_objection_scan([_persona()] * 2, "pitch", 2, "medium"))

    kinds = [e["type"] for e in events]
    assert kinds[0] == "start"
    assert kinds.count("agent") == 4
    # the clustering phase marker precedes the final result
    assert kinds.index("clustering") < kinds.index("done")
    assert events[-1]["result"]["top_objections"][0]["count"] == 4
