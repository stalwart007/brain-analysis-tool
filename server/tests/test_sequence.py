"""Message sequence optimiser: the estimator, the solver, and the stream.

The math is tested against hand-computed answers, not just properties — the
2×2 precedence example below is worked out fully in the test body so a future
change to the position adjustment cannot quietly pass.
"""

import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from app import sequence, storage
from app.main import app
from app.persona import seed_persona
from app.schemas import (
    BehavioralSignal,
    SequenceRequest,
    TwinSequenceStep,
    TwinSequenceWalk,
)
from app.sequence import (
    additive_effects,
    net_advantage,
    ordering_objective,
    precedence_matrix,
    sample_orderings,
    sequence_result,
    solve_linear_ordering,
    stream_sequence,
    walk_gains,
)

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


def _walk(ordering, intents, disengage_at=None, fatigue=0.2):
    return {
        "ordering": list(ordering),
        "steps": [
            {
                "intent": intent,
                "fatigue": fatigue,
                "disengaged": disengage_at is not None and i >= disengage_at,
            }
            for i, intent in enumerate(intents)
        ],
    }


# ──────────────────────────── design of the sample ───────────────────────


def test_design_starts_with_submitted_and_its_reverse():
    """Submitted + reverse is the identifiability condition: it puts every
    unordered pair on both sides of the precedence relation."""
    orderings = sample_orderings(4, 6)
    assert orderings[0] == [0, 1, 2, 3]
    assert orderings[1] == [3, 2, 1, 0]
    assert len(orderings) == 6
    assert len({tuple(o) for o in orderings}) == 6


def test_design_cannot_exceed_the_sample_space():
    assert len(sample_orderings(2, 6)) == 2  # 2! = 2
    assert len(sample_orderings(3, 24)) == 6  # 3! = 6


# ───────────────────── position vs content decomposition ─────────────────


def test_additive_fit_recovers_a_known_position_and_content_split():
    true_mu, true_content, true_position = 0.3, [0.2, 0.0, -0.2], [0.1, 0.0, -0.1]
    observations = []
    for ordering in ([0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]):
        for position, m in enumerate(ordering):
            observations.append(
                (m, position, true_mu + true_content[m] + true_position[position])
            )

    mu, content, position = additive_effects(observations, 3, 3)
    assert mu == pytest.approx(true_mu)
    assert content == pytest.approx(true_content)
    assert position == pytest.approx(true_position)


def test_additive_fit_reports_unobserved_cells_as_missing_not_zero():
    # message 2 never appears; position 2 never filled
    mu, content, position = additive_effects([(0, 0, 0.4), (1, 1, 0.2)], 3, 3)
    assert content[2] is None and position[2] is None
    assert content[0] is not None


def test_empty_observations_do_not_divide_by_zero():
    mu, content, position = additive_effects([], 3, 3)
    assert mu == 0.0 and content == [None] * 3 and position == [None] * 3


# ─────────────────────────── precedence matrix ───────────────────────────


def test_precedence_is_antisymmetric_and_position_adjusted():
    """Worked by hand. Gains: m0@p0=0.3, m1@p1=0.6 (ordering 0→1) and
    m1@p0=0.2, m0@p1=0.3 (ordering 1→0). The balanced 2×2 fit gives
    position=[-0.10, +0.10], so the adjusted gains for m1 are 0.50 with m0
    before it and 0.30 without → P[0][1] = +0.20, P[1][0] = −0.20."""
    walks = [_walk([0, 1], [0.3, 0.9]), _walk([1, 0], [0.2, 0.5])]
    infos = [walk_gains(w) for w in walks]
    observations = [obs for i in infos for obs in i["gains"]]
    _, _, position = additive_effects(observations, 2, 2)
    assert position == pytest.approx([-0.10, 0.10])

    matrix, support = precedence_matrix(infos, position, 2)
    assert matrix[0][1] == pytest.approx(0.20)
    assert matrix[1][0] == pytest.approx(-0.20)
    assert support[0][1] == 1 and support[1][0] == 1

    # the ordering decision is the net: putting 0 first gains 0.20 in message 1
    # AND avoids the 0.20 message 0 loses when it goes second
    net = net_advantage(matrix)
    assert net[0][1] == pytest.approx(0.40)
    assert net[1][0] == pytest.approx(-0.40)


def test_unmeasured_pairs_are_zero_with_zero_support():
    """One ordering only: nothing is ever observed both ways, so no cell is
    estimable and every support count is zero."""
    infos = [walk_gains(_walk([0, 1, 2], [0.2, 0.4, 0.6]))]
    matrix, support = precedence_matrix(infos, [0.0, 0.0, 0.0], 3)
    assert all(v == 0.0 for row in matrix for v in row)
    assert max(v for row in support for v in row) == 0


# ────────────────────────────── the solver ───────────────────────────────


def test_solver_recovers_a_planted_total_order():
    truth = [2, 0, 3, 1]
    rank = {m: i for i, m in enumerate(truth)}
    matrix = [
        [0.0 if i == j else (1.0 if rank[i] < rank[j] else -1.0) for j in range(4)]
        for i in range(4)
    ]
    order, objective, info = solve_linear_ordering(matrix)
    assert order == truth
    assert objective == pytest.approx(6.0)  # all 4C2 = 6 pairs satisfied
    assert info["passes"] >= 1


def test_objective_flips_sign_when_the_ordering_reverses():
    matrix = [[0.0, 0.5, 0.2], [-0.5, 0.0, 0.4], [-0.2, -0.4, 0.0]]
    assert ordering_objective([0, 1, 2], matrix) == pytest.approx(1.1)
    assert ordering_objective([2, 1, 0], matrix) == pytest.approx(-1.1)


def test_solver_is_trivial_for_one_message():
    order, objective, _ = solve_linear_ordering([[0.0]])
    assert order == [0] and objective == 0.0


# ──────────────────────── walk bookkeeping / edges ───────────────────────


def test_disengagement_truncates_the_walk():
    """Messages after a drop-off were never seen; scoring them as zero gain
    would credit the ordering for attention it never got."""
    info = walk_gains(_walk([2, 0, 1], [0.4, 0.5, 0.9], disengage_at=1))
    assert info["steps_seen"] == 2
    assert info["disengaged_at"] == 1
    assert info["terminal_intent"] == pytest.approx(0.5)
    assert [m for m, _, _ in info["gains"]] == [2, 0]
    assert info["gains"][0][2] == pytest.approx(0.4)  # gains start from zero intent
    assert info["gains"][1][2] == pytest.approx(0.1)


def test_gains_sum_to_the_end_state_intent():
    info = walk_gains(_walk([0, 1, 2], [0.2, 0.5, 0.45]))
    assert sum(g for _, _, g in info["gains"]) == pytest.approx(info["terminal_intent"])


# ─────────────────────────── full aggregation ────────────────────────────

MESSAGES = ["social proof", "the discount", "the ask"]


def _mixed_walks():
    """Message 2 ("the ask") lands far better once 0 and 1 have run; message 1
    is strong wherever it goes. Built so the order effect is not a position
    effect in disguise."""
    walks = []
    for _ in range(4):
        walks.append(_walk([0, 1, 2], [0.20, 0.55, 0.90]))  # ask last: +0.35
        walks.append(_walk([2, 1, 0], [0.10, 0.45, 0.55]))  # ask first: +0.10
        walks.append(_walk([1, 2, 0], [0.35, 0.55, 0.65]))  # ask after 1 only: +0.20
    return walks


def test_result_reports_ordering_lift_uncertainty_and_coverage():
    out = sequence_result(_mixed_walks(), MESSAGES, "low", orderings_designed=4)

    assert out["twin_count"] == 12
    assert sorted(out["recommended_ordering"]) == [0, 1, 2]
    assert out["objective"] >= out["submitted_objective"]  # best-of candidates
    assert out["objective_gain"] >= 0
    assert len(out["objective_gain_ci"]) == 2
    assert out["exploration"] == {
        "orderings_designed": 4,
        "orderings_walked": 3,
        "sample_space": 6,
        "explored_fraction": 0.5,
        "pairs_estimated": 6,
        "pairs_total": 6,
        "min_pair_support": 4,
    }
    # the DIRECTIONAL matrix is not antisymmetric (P[i][j] is about j, P[j][i]
    # about i) — the net-advantage matrix the objective uses is, exactly
    net = out["net_advantage_matrix"]
    assert all(net[i][j] == pytest.approx(-net[j][i]) for i in range(3) for j in range(3))
    assert all(net[i][i] == 0.0 for i in range(3))
    p = out["precedence_matrix"]
    assert net[0][2] == pytest.approx(p[0][2] - p[2][0])
    # per-ordering posteriors over what was actually walked
    assert out["bayesian"]["draws"] > 0
    assert {o["label"] for o in out["orderings"]} == {"1-2-3", "3-2-1", "2-3-1"}
    assert next(o for o in out["orderings"] if o["is_submitted"])["walks"] == 4
    assert out["terminal_intent_ci"] is not None
    assert out["search"]["optimal"] is False


def test_position_effects_are_reported_apart_from_content():
    """The headline separation: a message that only wins because it went last
    must show up in position_effects, not content_effects."""
    out = sequence_result(_mixed_walks(), MESSAGES, "low", orderings_designed=3)
    assert len(out["position_effects"]) == 3
    assert len(out["content_effects"]) == 3
    # centered effects sum to ~0 by construction — that's the identifiability
    assert sum(out["content_effects"]) == pytest.approx(0.0, abs=1e-3)
    assert sum(out["position_effects"]) == pytest.approx(0.0, abs=1e-3)
    assert out["primacy_effect"] is not None and out["recency_effect"] is not None
    assert [c["message_index"] for c in out["content_ranking"]] == sorted(
        range(3), key=lambda i: out["content_effects"][i], reverse=True
    )


def test_single_message_is_degenerate_but_does_not_crash():
    walks = [_walk([0], [0.6]), _walk([0], [0.4])]
    out = sequence_result(walks, ["only one"], "low", orderings_designed=1)
    assert out["recommended_ordering"] == [0]
    assert out["objective"] == 0.0 and out["objective_gain"] == 0.0
    assert out["gain_supported"] is False
    assert out["exploration"]["sample_space"] == 1
    assert out["exploration"]["explored_fraction"] == 1.0
    assert out["mean_terminal_intent"] == pytest.approx(0.5)


def test_disengagement_rates_surface_in_the_result():
    walks = [
        _walk([0, 1, 2], [0.3, 0.4, 0.5]),
        _walk([0, 1, 2], [0.3, 0.2, 0.1], disengage_at=1),
        _walk([2, 1, 0], [0.2, 0.3, 0.4]),
    ]
    out = sequence_result(walks, MESSAGES, "low", orderings_designed=2)
    assert out["disengage_rate"] == pytest.approx(1 / 3, abs=1e-4)  # rounded to 4dp
    assert out["mean_disengage_position"] == 1.0


def test_walks_with_the_wrong_step_count_are_dropped():
    good = _walk([0, 1, 2], [0.2, 0.4, 0.6])
    ragged = {"ordering": [0, 1, 2], "steps": [{"intent": 0.5, "fatigue": 0.1, "disengaged": False}]}
    out = sequence_result([good, ragged], MESSAGES, "low", orderings_designed=1)
    assert out["twin_count"] == 1


def test_no_valid_walks_raises():
    with pytest.raises(RuntimeError):
        sequence_result([], MESSAGES, "low", orderings_designed=2)
    with pytest.raises(RuntimeError):
        sequence_result(
            [{"ordering": [0, 1, 2], "steps": []}], MESSAGES, "low", orderings_designed=2
        )


# ───────────────────────────── the stream ────────────────────────────────


def test_stream_emits_design_agents_matrix_and_solution(monkeypatch):
    async def fake_twin(persona, messages, ordering, load, semaphore):
        # intent climbs, and climbs further when "the ask" (index 2) comes late
        steps = []
        intent = 0.0
        for position, m in enumerate(ordering):
            intent = min(1.0, intent + (0.4 if m == 2 and position == 2 else 0.2))
            steps.append(
                TwinSequenceStep(intent=intent, fatigue=0.1 * position, disengaged=False, reaction="…")
            )
        return TwinSequenceWalk(steps=steps, gist="…")

    monkeypatch.setattr(sequence, "_sequence_twin", fake_twin)
    request = SequenceRequest(messages=MESSAGES, twins_per_persona=3, orderings_sampled=4)
    events = collect(stream_sequence([_persona() for _ in range(3)], request))

    kinds = [e["type"] for e in events]
    assert kinds[0] == "start"
    assert events[0]["sample_space"] == 6
    assert kinds.count("ordering") == 4  # the design, emitted before any walk
    assert kinds.count("agent") == 9  # 3 personas × 3 twins
    assert kinds.index("matrix") < kinds.index("solved") < kinds.index("done")
    solved = next(e for e in events if e["type"] == "solved")
    assert sorted(solved["ordering"]) == [0, 1, 2]
    assert solved["search"]["optimal"] is False
    result = events[-1]["result"]
    assert result["twin_count"] == 9
    assert result["exploration"]["orderings_walked"] <= 4


def test_stream_reports_failures_and_raises_when_all_twins_fail(monkeypatch):
    async def always_fail(persona, messages, ordering, load, semaphore):
        return None

    monkeypatch.setattr(sequence, "_sequence_twin", always_fail)
    request = SequenceRequest(messages=MESSAGES, twins_per_persona=2)
    with pytest.raises(RuntimeError):
        collect(stream_sequence([_persona()], request))


def test_stream_caps_the_design_at_the_number_of_twins(monkeypatch):
    """Asking for 12 orderings with 2 twins would leave cells backed by a
    single walk, which is not an estimate."""

    async def fake_twin(persona, messages, ordering, load, semaphore):
        return TwinSequenceWalk(
            steps=[
                TwinSequenceStep(intent=0.3, fatigue=0.1, disengaged=False, reaction="…")
                for _ in ordering
            ],
            gist="…",
        )

    monkeypatch.setattr(sequence, "_sequence_twin", fake_twin)
    request = SequenceRequest(messages=MESSAGES, twins_per_persona=2, orderings_sampled=12)
    events = collect(stream_sequence([_persona()], request))
    assert events[0]["orderings"] == 2


def test_endpoint_registered_and_key_guarded(monkeypatch):
    assert "/v1/studies/sequence/stream" in {r.path for r in app.routes}
    monkeypatch.setenv("COGNISWARM_API_KEYS", "key-alpha")
    res = client.post("/v1/studies/sequence/stream", json={"messages": ["a", "b"]})
    assert res.status_code == 401


def _session_with_persona() -> str:
    res = client.post(
        "/v1/ingest",
        json={
            "site_id": "seq",
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
    """End to end through _sse: the matrix, the solved ordering and the result
    must all survive json.dumps, and the run lands under the `sequence` kind."""
    monkeypatch.delenv("COGNISWARM_API_KEYS", raising=False)

    async def fake_twin(persona, messages, ordering, load, semaphore):
        steps, intent = [], 0.0
        for position, m in enumerate(ordering):
            intent = min(1.0, intent + (0.4 if m == 2 and position == 2 else 0.15))
            steps.append(
                TwinSequenceStep(intent=intent, fatigue=0.1, disengaged=False, reaction="…")
            )
        return TwinSequenceWalk(steps=steps, gist="…")

    monkeypatch.setattr(sequence, "_sequence_twin", fake_twin)

    session_id = _session_with_persona()
    with client.stream(
        "POST",
        "/v1/studies/sequence/stream",
        json={
            "messages": MESSAGES,
            "session_ids": [session_id],
            "twins_per_persona": 6,
            "orderings_sampled": 4,
        },
    ) as resp:
        assert resp.status_code == 200
        frames = [json.loads(line[5:]) for line in resp.iter_lines() if line.startswith("data:")]

    kinds = [f["type"] for f in frames]
    assert kinds.index("matrix") < kinds.index("solved") < kinds.index("done")
    done = frames[-1]
    run_id = done["result"]["run_id"]
    assert run_id
    runs = storage.list_swarm_runs(kind="sequence")
    assert runs[0]["id"] == run_id
    assert len(runs[0]["result"]["recommended_ordering"]) == 3
    assert runs[0]["request"]["messages"] == MESSAGES


# ── design assignment: flat coverage AND no persona/ordering aliasing ─────


def _persona_ordering_sets(P: int, R: int, L: int) -> list[set[int]]:
    """Which orderings each persona reaches, under the real assignment."""
    from app.sequence import assign_orderings

    assignment = assign_orderings(P, R, L)
    sets: list[set[int]] = [set() for _ in range(P)]
    for slot, order_index in enumerate(assignment):
        sets[slot % P].add(order_index)
    return sets


def test_assignment_covers_every_ordering_flatly():
    """Including ordering 0 — the submitted baseline every gain is measured
    against. Under `(p + rep) % L` it was walked exactly once, so its CI was
    None and its Bayesian arm was a single observation."""
    from app.sequence import assign_orderings

    for P, R, L in ((6, 4, 6), (6, 3, 6), (5, 3, 6), (8, 2, 6), (3, 4, 6)):
        counts = [0] * L
        for order_index in assign_orderings(P, R, L):
            counts[order_index] += 1
        assert min(counts) >= 1, f"ordering never walked at P={P} R={R} L={L}"
        assert max(counts) - min(counts) <= 1, f"unflat counts at P={P} R={R} L={L}"


def test_personas_are_not_confined_to_disjoint_ordering_sets():
    """The regression this replaces.

    `(p_index + rep * len(personas)) % L` put persona p in the coset
    p + <P> of Z_L. Verified before the fix: at P=6, R=3, L=6 every persona
    walked exactly ONE ordering and no two personas shared any — so
    `mean_terminal_intent` per ordering was a per-persona mean, and the
    recommendation was a persona ranking wearing an ordering label.
    """
    for P, R, L in ((6, 3, 6), (6, 4, 6), (4, 3, 6), (8, 2, 6)):
        sets = _persona_ordering_sets(P, R, L)
        # no persona is pinned to a single ordering when it walks more than once
        if R > 1:
            assert max(len(s) for s in sets) > 1, f"persona pinned at P={P} L={L}"
        # and the sets genuinely overlap rather than partitioning
        shared = any(
            sets[a] & sets[b] for a in range(P) for b in range(a + 1, P)
        )
        assert shared, f"personas partition into disjoint ordering sets at P={P} L={L}"


def test_assignment_is_deterministic_under_the_design_seed():
    """Reproducibility is a product claim; the design must not drift per run."""
    from app.sequence import assign_orderings

    assert assign_orderings(6, 3, 6) == assign_orderings(6, 3, 6)
    assert assign_orderings(0, 3, 6) == [] and assign_orderings(6, 3, 0) == []
