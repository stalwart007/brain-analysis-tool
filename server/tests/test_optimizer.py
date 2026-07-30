"""Copy optimizer: the pure aggregation, and the evolutionary loop's orchestration.

Everything here runs with zero network calls — `optimizer_result` is pure, and
the streaming test monkeypatches the twin scorer and the breeder, so what is
under test is the search itself: Thompson allocation, lineage bookkeeping, and
the refusal to claim an improvement the intervals don't support.
"""

import asyncio
import itertools
import json

import pytest
from fastapi.testclient import TestClient

from app import optimizer, storage
from app.main import app
from app.optimizer import optimizer_result, stream_copy_optimizer
from app.persona import seed_persona
from app.schemas import BehavioralSignal, CopyMutant, CopyOptimizerRequest, TwinCopyScore

client = TestClient(app)


def collect(gen):
    async def run():
        return [e async for e in gen]

    return asyncio.run(run())


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


def _score(intent: float, acts: bool | None = None) -> TwinCopyScore:
    return TwinCopyScore(
        intent=intent,
        would_act=(intent > 0.5) if acts is None else acts,
        strongest_phrase="the opening line",
        reaction="…",
    )


def _variant(vid, generation, text, parents=(), operation="mutation"):
    return {
        "id": vid,
        "generation": generation,
        "text": text,
        "parents": list(parents),
        "operation": operation,
        "rationale": "r",
    }


SEED = _variant("g0v0", 0, "the copy we have today", operation="seed")


def _result(lineage, scores, **kw):
    return optimizer_result(lineage, scores, "g0v0", "brief", "low", kw.pop("gens", 2), **kw)


# ─────────────────────────── pure aggregation ────────────────────────────


def test_ranks_population_and_records_lineage():
    lineage = [
        SEED,
        _variant("g0v1", 0, "mutant of the seed", parents=["g0v0"]),
        _variant("g1v0", 1, "child of both", parents=["g0v0", "g0v1"], operation="crossover"),
    ]
    scores = {
        "g0v0": [_score(0.2) for _ in range(10)],
        "g0v1": [_score(0.5, acts=False) for _ in range(10)],
        "g1v0": [_score(0.9) for _ in range(10)],
    }
    out = _result(lineage, scores)

    assert out["twin_count"] == 30
    assert out["best_variant_id"] == "g1v0"
    assert [v["id"] for v in out["population"]] == ["g1v0", "g0v1", "g0v0"]
    assert out["generations"][0]["best_fitness"] == 0.5  # best of gen 0
    assert out["generations"][1]["best_fitness"] == 0.9
    assert out["generations"][0]["evaluations"] == 20
    # every arm gets a posterior, and the lineage is a directed edge list
    assert all(v["posterior"] is not None for v in out["population"])
    assert {"parent": "g0v0", "child": "g0v1", "operation": "mutation"} in out["lineage_edges"]
    assert sum(1 for e in out["lineage_edges"] if e["child"] == "g1v0") == 2


def test_claims_a_win_only_when_both_intervals_clear_the_seed():
    lineage = [SEED, _variant("g0v1", 0, "much better copy", parents=["g0v0"])]
    scores = {
        "g0v0": [_score(0.05, acts=False) for _ in range(20)],
        "g0v1": [_score(0.95, acts=True) for _ in range(20)],
    }
    out = _result(lineage, scores)

    assert out["beat_seed"] is True
    assert out["convergence"] == "improved"
    assert out["posterior_intervals_separated"] is True
    assert out["intent_lift_vs_seed"] == pytest.approx(0.9, abs=1e-6)
    # the gate now runs on the honest quantities, and on a real 0.9 gap they
    # all agree: adjusted lift positive, its interval clear of zero, and the
    # act-rate posteriors essentially certain
    assert out["selection_adjusted_lift"] > 0
    assert out["selection_adjusted_lift_ci"][0] > 0
    assert out["p_best_beats_seed"] > 0.99


# ───────────────── ranking: shrinkage, not the raw mean ──────────────────


def test_a_lucky_two_trial_arm_does_not_outrank_a_solid_forty_trial_seed():
    """THE RANKING BUG. Raw-mean sorting put a 2-trial 0.90 above a 40-trial
    0.85 outright. Empirical-Bayes shrinkage pulls the thin arm most of the way
    to the grand mean while the seed barely moves.

    Measured on this exact population: grand mean 0.8068, within-arm variance
    0.0294, between-arm variance 0.0009 → τ̂ = 31.98 prior trials. The seed
    goes 0.8500 → 0.8308, the challenger 0.9000 → 0.8123, so the seed ranks
    first. Over 400 random mutant populations of this shape (realistic twin
    noise, sd 0.18–0.30) the shrunk ranking puts the seed above the lucky
    2-trial arm in 86.8% of them; the raw-mean ranking never did.
    """
    seed_rows = [1.0] * 11 + [0.9] * 8 + [0.8] * 11 + [0.7] * 10  # n=40, mean 0.85
    others = {
        "g0v2": [0.7, 0.9, 0.6, 0.8, 0.9],
        "g0v3": [0.6, 1.0, 0.6, 0.7, 1.0, 0.6, 0.7, 1.0],
        "g0v4": [0.4, 0.7, 1.0, 1.0, 0.6, 0.7, 0.9, 0.5, 1.0, 1.0, 0.4, 0.3, 0.9],
        "g0v5": [0.7, 0.7, 0.7, 0.5, 0.9, 0.4, 1.0, 0.8, 0.9, 0.6,
                 0.5, 1.0, 0.9, 1.0, 0.8, 1.0, 0.9, 0.5, 1.0, 0.9],
    }
    lineage = [SEED, _variant("g0v1", 0, "lucky pair", parents=["g0v0"])]
    scores = {
        "g0v0": [_score(x) for x in seed_rows],
        "g0v1": [_score(0.9), _score(0.9)],
    }
    for vid, rows in others.items():
        lineage.append(_variant(vid, 0, f"mutant {vid}", parents=["g0v0"]))
        scores[vid] = [_score(x) for x in rows]

    out = _result(lineage, scores)
    by_id = {v["id"]: v for v in out["population"]}

    # the raw means are exactly the ones that used to drive the sort…
    assert by_id["g0v1"]["mean_intent"] == pytest.approx(0.9, abs=1e-6)
    assert by_id["g0v0"]["mean_intent"] == pytest.approx(0.85, abs=1e-6)
    # …and the raw mean is still reported, unchanged
    assert by_id["g0v1"]["mean_intent"] > by_id["g0v0"]["mean_intent"]
    # …but the ranking key is the shrunk one, and it puts the seed first
    assert by_id["g0v0"]["shrunk_intent"] > by_id["g0v1"]["shrunk_intent"]
    assert out["best_variant_id"] == "g0v0"
    assert out["convergence"] == "seed_wins"
    assert out["shrinkage"]["prior_strength"] == pytest.approx(31.98, abs=0.1)


def test_full_pooling_when_arms_differ_by_no_more_than_their_own_noise():
    """τ̂² ≤ 0 — the arm means are no more spread than within-arm sampling
    noise alone predicts. Full pooling is the honest answer: every arm gets the
    grand mean and the ranking falls to the tie-breakers (p_best, then trials),
    which is what keeps a 2-trial arm off the top."""
    rows = {
        "g0v0": [0.9, 0.7, 0.5, 0.3, 0.9, 0.7, 0.5, 0.3],
        "g0v1": [0.3, 0.9, 0.5, 0.7, 0.7, 0.5, 0.9, 0.3],
        "g0v2": [0.5, 0.5, 0.7, 0.7, 0.3, 0.3, 0.9, 0.9],
    }
    shrunk, info = optimizer._shrunk_means(rows)
    assert info["pooled"] is True
    assert len(set(round(v, 9) for v in shrunk.values())) == 1
    assert shrunk["g0v0"] == pytest.approx(info["grand_mean"], abs=1e-6)


def test_shrinkage_edge_cases_do_not_invent_a_ranking():
    # a single arm has nothing to shrink toward — its own mean stands
    shrunk, info = optimizer._shrunk_means({"a": [0.2, 0.8]})
    assert shrunk == {"a": pytest.approx(0.5)}
    assert info["prior_strength"] == 0.0
    # every arm a single reading: within-arm noise is inestimable, and calling
    # it zero would rank pure luck. Pool.
    shrunk, info = optimizer._shrunk_means({"a": [0.9], "b": [0.1], "c": [0.5]})
    assert info["pooled"] is True
    assert len(set(shrunk.values())) == 1
    # no arm with any reading at all
    assert optimizer._shrunk_means({"a": []})[0] == {}


# ──────────── the honest lift: selection-adjusted, not in-sample ─────────


def test_naive_and_adjusted_lift_are_both_reported_and_differ():
    """Repo precedent (sequence.py): keep the naive number beside the honest
    one so the size of the correction is visible."""
    lineage = [SEED]
    scores = {"g0v0": [_score(x) for x in (0.5, 0.4, 0.6, 0.5, 0.5, 0.6, 0.4, 0.5)]}
    for i, rows in enumerate(
        [(0.6, 0.7, 0.5, 0.6), (0.5, 0.6, 0.4, 0.5), (0.7, 0.6, 0.8, 0.5)], start=1
    ):
        vid = f"g0v{i}"
        lineage.append(_variant(vid, 0, f"mutant {i}", parents=["g0v0"]))
        scores[vid] = [_score(x) for x in rows]
    out = _result(lineage, scores)

    # every key the honest reading needs is present
    for key in (
        "naive_intent_lift",
        "selection_adjusted_lift",
        "selection_adjusted_lift_ci",
        "selection_bias",
        "p_best_beats_seed",
    ):
        assert key in out
    # the legacy key still carries the naive number, byte for byte
    assert out["intent_lift_vs_seed"] == out["naive_intent_lift"]
    # and the correction is real: the in-sample margin exceeds the honest one
    assert out["naive_intent_lift"] > out["selection_adjusted_lift"]
    assert out["selection_bias"] == pytest.approx(
        out["naive_intent_lift"] - out["selection_adjusted_lift"], abs=1e-9
    )


def test_not_supported_does_not_showcase_the_naive_lift():
    """The audit finding: the naive lift was reported as a headline even when
    convergence said the search proved nothing. The verdict must now name the
    honest quantity, and may mention the naive one only as the bias it was."""
    lineage = [SEED, _variant("g0v1", 0, "marginally different copy", parents=["g0v0"])]
    scores = {
        "g0v0": [_score(i) for i in (0.4, 0.6, 0.5, 0.45, 0.55, 0.5, 0.5, 0.45)],
        "g0v1": [_score(i) for i in (0.5, 0.7, 0.6, 0.55, 0.65, 0.6, 0.6, 0.55)],
    }
    out = _result(lineage, scores)

    assert out["convergence"] == "not_supported"
    assert out["beat_seed"] is False
    assert "did NOT beat" in out["verdict"]
    # the naive lift is positive — and precisely the number that must not be
    # sold as the finding
    assert out["naive_intent_lift"] > 0
    naive_pct = f"{out['naive_intent_lift'] * 100:.1f}%"
    assert naive_pct not in out["verdict"]
    # the verdict must be about selection / overlap, not about a margin won
    assert any(
        phrase in out["verdict"]
        for phrase in ("selection", "overlap", "includes zero", "thinly sampled")
    )


def test_the_gate_needs_the_adjusted_interval_not_the_naive_one():
    """A win requires posterior separation AND an adjusted-lift interval clear
    of zero. An act rate that separates on its own is not enough.

    The two arms carry the SAME spread of intents — the challenger's twins
    merely crossed the would_act threshold — so the act-rate posteriors
    separate completely while the honest intent lift has nothing to find.
    """
    intents = (0.3, 0.7, 0.4, 0.6, 0.5, 0.5, 0.45, 0.55, 0.35, 0.65, 0.4, 0.6)
    lineage = [SEED, _variant("g0v1", 0, "acts more, likes it the same", parents=["g0v0"])]
    scores = {
        "g0v0": [_score(i, acts=False) for i in intents],
        "g0v1": [_score(i, acts=True) for i in intents],
    }
    out = _result(lineage, scores)

    assert out["posterior_intervals_separated"] is True
    assert out["p_best_beats_seed"] > 0.99
    # identical intent samples: the honest lift is exactly nothing
    assert out["selection_adjusted_lift"] == pytest.approx(0.0, abs=1e-9)
    assert out["beat_seed"] is False  # the intent side never cleared
    assert out["convergence"] == "not_supported"


def test_thin_challengers_cannot_produce_an_honest_lift_and_so_cannot_win():
    """Every challenger with a single reading: no split can both select and
    evaluate, so the adjusted lift is None — which must read as 'cannot
    support a claim', never as zero."""
    lineage = [SEED]
    scores = {"g0v0": [_score(0.3) for _ in range(10)]}
    for i in (1, 2, 3):
        vid = f"g0v{i}"
        lineage.append(_variant(vid, 0, f"one-shot {i}", parents=["g0v0"]))
        scores[vid] = [_score(1.0)]
    out = _result(lineage, scores)

    if out["best_variant_id"] != "g0v0":
        assert out["selection_adjusted_lift"] is None
        assert out["beat_seed"] is False
        assert out["convergence"] == "not_supported"
        assert "thinly sampled" in out["verdict"]


def test_p_beats_is_a_seeded_beta_comparison_and_is_deterministic():
    a = optimizer._p_beats((9, 10), (1, 10))
    b = optimizer._p_beats((9, 10), (1, 10))
    assert a == b  # seeded
    assert a > 0.99
    # symmetric-ish: reversing the arms reverses the verdict
    assert optimizer._p_beats((1, 10), (9, 10)) < 0.01
    # no data on either side is a coin flip, not a claim
    assert optimizer._p_beats((0, 0), (0, 0)) == pytest.approx(0.5, abs=0.02)


def test_result_is_deterministic_across_repeated_calls():
    """All randomness is seeded — the same twin outputs must produce the same
    lift, interval and verdict every time."""
    lineage = [SEED, _variant("g0v1", 0, "a challenger", parents=["g0v0"])]
    scores = {
        "g0v0": [_score(i) for i in (0.4, 0.6, 0.5, 0.45, 0.55, 0.5, 0.4, 0.6)],
        "g0v1": [_score(i) for i in (0.6, 0.7, 0.5, 0.65, 0.55, 0.6, 0.7, 0.5)],
    }
    a, b = _result(lineage, scores), _result(lineage, scores)
    for key in (
        "selection_adjusted_lift",
        "selection_adjusted_lift_ci",
        "p_best_beats_seed",
        "convergence",
        "verdict",
        "best_variant_id",
    ):
        assert a[key] == b[key]


def test_refuses_to_claim_a_win_when_intervals_overlap():
    """The winner's curse: the leader of a noisy search is upward-biased, so a
    +0.1 mean lift on overlapping intervals is not an improvement."""
    lineage = [SEED, _variant("g0v1", 0, "marginally different copy", parents=["g0v0"])]
    scores = {
        "g0v0": [_score(i) for i in (0.4, 0.6, 0.5, 0.45, 0.55, 0.5)],
        "g0v1": [_score(i) for i in (0.5, 0.7, 0.6, 0.55, 0.65, 0.6)],
    }
    out = _result(lineage, scores)

    assert out["best_variant_id"] == "g0v1"  # it does rank first
    assert out["beat_seed"] is False  # …but the win is not supported
    assert out["convergence"] == "not_supported"
    assert "did NOT beat" in out["verdict"]
    assert out["intent_lift_vs_seed"] > 0


def test_identical_variants_never_produce_a_claimed_improvement():
    """Two arms with byte-identical copy and identical scores: whichever wins
    the tiebreak, no improvement may be claimed."""
    lineage = [SEED, _variant("g0v1", 0, SEED["text"], parents=["g0v0"])]
    rows = [_score(0.5, acts=False) for _ in range(8)]
    out = _result(lineage, {"g0v0": list(rows), "g0v1": list(rows)})

    assert out["beat_seed"] is False
    assert out["convergence"] in {"seed_wins", "not_supported"}
    assert out["population"][0]["mean_intent"] == out["population"][1]["mean_intent"]


def test_single_evaluation_cannot_win_because_it_has_no_interval():
    """One reading is not a measurement: bootstrap_ci returns None below n=2,
    and a missing interval must block the claim rather than default to 'clear'."""
    lineage = [SEED, _variant("g0v1", 0, "lucky one-shot", parents=["g0v0"])]
    scores = {"g0v0": [_score(0.1) for _ in range(10)], "g0v1": [_score(1.0)]}
    out = _result(lineage, scores)

    assert out["best_variant_id"] == "g0v1"
    assert out["population"][0]["intent_ci"] is None
    assert out["intent_intervals_separated"] is False
    assert out["beat_seed"] is False


def test_unevaluated_variants_are_reported_not_ranked():
    """Thompson can starve an arm to zero trials; that must not divide by zero
    and must not rank an unmeasured variant above a measured one."""
    lineage = [SEED, _variant("g0v1", 0, "never scored", parents=["g0v0"])]
    out = _result(lineage, {"g0v0": [_score(0.3) for _ in range(4)], "g0v1": []})

    assert out["unevaluated"] == ["g0v1"]
    assert [v["id"] for v in out["population"]] == ["g0v0"]
    assert out["convergence"] == "seed_wins"
    assert out["twin_count"] == 4


def test_seed_only_population_is_honest_about_finding_nothing():
    out = _result([SEED], {"g0v0": [_score(0.4) for _ in range(5)]})
    assert out["best_variant_id"] == "g0v0"
    assert out["convergence"] == "seed_wins"
    assert out["beat_seed"] is False


def test_seed_that_never_scored_leaves_no_baseline():
    lineage = [SEED, _variant("g0v1", 0, "scored child", parents=["g0v0"])]
    out = _result(lineage, {"g0v0": [], "g0v1": [_score(0.8) for _ in range(6)]})
    assert out["convergence"] == "no_baseline"
    assert out["beat_seed"] is False
    assert "no baseline" in out["verdict"]


def test_all_twins_failing_raises():
    with pytest.raises(RuntimeError):
        _result([SEED], {"g0v0": []})


# ───────────────────────────── orchestration ─────────────────────────────


def _fake_breeder():
    counter = itertools.count()

    async def fake_breed(brief, parents, n_children, generation):
        # child 0 of every generation carries the marker the fake twin rewards,
        # so there is always a clear leader for Thompson to find
        return [
            CopyMutant(
                text=("BEST " if i == 0 else "meh ") + f"variant {next(counter)}",
                parent_ids=[parents[0]["id"]],
                operation="mutation",
                rationale="changed the hook",
            )
            for i in range(n_children)
        ]

    return fake_breed


def test_stream_runs_generations_and_concentrates_budget_on_the_leader(monkeypatch):
    async def fake_score(persona, text, load, semaphore):
        good = text.startswith("BEST")
        return _score(0.9 if good else 0.2, acts=good)

    monkeypatch.setattr(optimizer, "_score_twin", fake_score)
    monkeypatch.setattr(optimizer, "_breed", _fake_breeder())

    request = CopyOptimizerRequest(
        brief="sell the thing",
        seed_variant="the copy we have today",
        population_size=3,
        generations=2,
        twins_per_persona=3,
        min_evals_per_variant=2,
    )
    events = collect(stream_copy_optimizer([_persona() for _ in range(4)], request))

    kinds = [e["type"] for e in events]
    assert kinds[0] == "start"
    assert kinds[-1] == "done"
    assert kinds.count("generation") == 2
    assert kinds.count("mutation") == 5  # seed + 2 at gen 0, then 3 bred for gen 1
    # budget is len(personas) × twins_per_persona per generation
    assert kinds.count("agent") == 4 * 3 * 2
    assert kinds.count("variant_scored") == 6  # 3 variants × 2 generations
    assert any(e["type"] == "wave" and e["policy"] == "thompson" for e in events)

    gen0_agents = [e for e in events if e["type"] == "agent" and e["generation"] == 0]
    by_variant: dict[str, int] = {}
    for e in gen0_agents:
        by_variant[e["variant"]] = by_variant.get(e["variant"], 0) + 1
    leader = max(by_variant, key=lambda v: by_variant[v])
    # Thompson must send the surplus (12 budget − 6 floor) to the winning arm
    assert by_variant[leader] > 12 // 3

    result = events[-1]["result"]
    assert result["beat_seed"] is True
    assert result["best_text"].startswith("BEST")
    assert result["population"][0]["mean_intent"] == 0.9
    assert result["failed_evaluations"] == 0


def test_stream_survives_a_dead_breeder_and_failing_twins(monkeypatch):
    """No mutants and half the twins failing must still produce a run, scored
    on whatever survived — the codebase degrades to a smaller sample."""
    calls = itertools.count()

    async def flaky_score(persona, text, load, semaphore):
        return None if next(calls) % 2 else _score(0.4)

    async def dead_breed(brief, parents, n_children, generation):
        return []

    monkeypatch.setattr(optimizer, "_score_twin", flaky_score)
    monkeypatch.setattr(optimizer, "_breed", dead_breed)

    request = CopyOptimizerRequest(
        brief="b", seed_variant="s", population_size=2, generations=2, twins_per_persona=2
    )
    events = collect(stream_copy_optimizer([_persona(), _persona()], request))

    kinds = [e["type"] for e in events]
    assert "agent_failed" in kinds
    assert any(e["type"] == "mutation" and e.get("status") == "failed" for e in events)
    result = events[-1]["result"]
    assert result["failed_evaluations"] > 0
    assert result["twin_count"] > 0
    assert result["convergence"] == "seed_wins"  # nothing else was ever born


def test_stream_raises_when_every_twin_fails(monkeypatch):
    async def always_fail(persona, text, load, semaphore):
        return None

    async def dead_breed(brief, parents, n_children, generation):
        return []

    monkeypatch.setattr(optimizer, "_score_twin", always_fail)
    monkeypatch.setattr(optimizer, "_breed", dead_breed)
    request = CopyOptimizerRequest(brief="b", seed_variant="s", generations=1, twins_per_persona=1)
    with pytest.raises(RuntimeError):
        collect(stream_copy_optimizer([_persona()], request))


def test_adopt_downgrades_a_crossover_that_lost_a_parent():
    child = CopyMutant(
        text="x", parent_ids=["g0v0", "hallucinated"], operation="crossover", rationale="r"
    )
    record = optimizer._adopt(child, 1, 0, {"g0v0"}, "g0v0")
    assert record["parents"] == ["g0v0"]
    assert record["operation"] == "mutation"  # one parent is not a crossover
    assert record["id"] == "g1v0"


def test_endpoint_registered_and_key_guarded(monkeypatch):
    assert "/v1/studies/optimize/stream" in {r.path for r in app.routes}
    monkeypatch.setenv("COGNISWARM_API_KEYS", "key-alpha")
    res = client.post(
        "/v1/studies/optimize/stream",
        json={"brief": "b", "seed_variant": "s"},
    )
    assert res.status_code == 401


def _session_with_persona() -> str:
    """A profiled session, without going through the LLM profiler."""
    res = client.post(
        "/v1/ingest",
        json={
            "site_id": "opt",
            "consent": True,
            "sdk_version": "0.2.0",
            "page_path": "/",
            "features": {"segment_start": 0, "segment_end": 1, "event_count": 1},
        },
    )
    session_id = res.json()["session_id"]
    storage.set_persona(session_id, _persona().model_dump())
    return session_id


def test_endpoint_streams_sse_and_persists_the_run(monkeypatch):
    """End to end through _sse: every frame must survive json.dumps and the
    terminal frame must be persisted under the new `optimize` kind."""
    monkeypatch.delenv("COGNISWARM_API_KEYS", raising=False)

    async def fake_score(persona, text, load, semaphore):
        return _score(0.8 if text.startswith("BEST") else 0.2)

    monkeypatch.setattr(optimizer, "_score_twin", fake_score)
    monkeypatch.setattr(optimizer, "_breed", _fake_breeder())

    session_id = _session_with_persona()
    with client.stream(
        "POST",
        "/v1/studies/optimize/stream",
        json={
            "brief": "sell it",
            "seed_variant": "the copy we have today",
            "session_ids": [session_id],
            "population_size": 2,
            "generations": 1,
            "twins_per_persona": 4,
        },
    ) as resp:
        assert resp.status_code == 200
        frames = [json.loads(line[5:]) for line in resp.iter_lines() if line.startswith("data:")]

    done = frames[-1]
    assert done["type"] == "done"
    run_id = done["result"]["run_id"]
    assert run_id
    runs = storage.list_swarm_runs(kind="optimize")
    assert runs[0]["id"] == run_id
    assert runs[0]["result"]["best_text"].startswith("BEST")
    assert runs[0]["request"]["seed_variant"] == "the copy we have today"
