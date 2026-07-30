"""The white room: known answers, refusal paths, and the two calibrations.

Every number asserted below was measured, not chosen. The three known-answer
tests come from DeGroot and Friedkin-Johnsen theory rather than from a previous
run of this code — a room of equals must land on the mean, a room with one
stubborn oracle must land on the oracle, and a room where nobody moves must not
move — so a rewrite of the solver has something outside itself to fail against.

The rest guard the failures these estimators are actually prone to: reporting an
influence matrix from a design that cannot identify one, inventing susceptibility
out of a rounding step, and calling a room's agreement conformity when it is
information (or the reverse).
"""

from __future__ import annotations

import json
import math
import random

import pytest

from app.room import (
    _POSITION_QUANTUM,
    _project_capped_simplex,
    arm_summary,
    cascade_scores,
    coalitions,
    conformity_vs_bayes,
    counterfactual_absence,
    equilibrium,
    fit_social_influence,
    influence_centrality,
    order_effect,
    placebo_contrast,
    preference_falsification,
    room_analysis,
)

# ── helpers: synthetic rooms with known answers ──────────────────────────


def _q(v: float) -> float:
    return min(1.0, max(0.0, round(v / _POSITION_QUANTUM) * _POSITION_QUANTUM))


def _fj_trajectories(u, w, lam, rounds, reps, sigma=0.03, seed=1, quantise=True):
    """Simulate Friedkin-Johnsen public trajectories from a known W and λ."""
    rng = random.Random(seed)
    n = len(u)
    out = []
    for _ in range(reps):
        x = [_q(u[i] + rng.gauss(0, sigma)) if quantise else u[i] for i in range(n)]
        traj = [list(x)]
        for _t in range(rounds - 1):
            nxt = []
            for i in range(n):
                v = lam[i] * sum(w[i][j] * x[j] for j in range(n)) + (1 - lam[i]) * u[i]
                v += rng.gauss(0, sigma)
                nxt.append(_q(v) if quantise else min(1.0, max(0.0, v)))
            x = nxt
            traj.append(list(x))
        out.append(traj)
    return out


def _turns(
    u,
    w,
    lam,
    rounds,
    reps,
    *,
    placebo=0,
    gap=0.05,
    seed=11,
    threshold=None,
):
    """Build the flat turn list `room_analysis` consumes, plus cast and meta."""
    n = len(u)
    names = [f"m{i}" for i in range(n)]
    cast = [
        {
            "name": nm,
            "role": "role",
            "archetype": "archetype",
            "openness": 0.5,
            "assertiveness": 0.5,
            "seniority": 0.5,
        }
        for nm in names
    ]
    turns: list[dict] = []
    meta: list[dict] = []
    rep_id = 0
    for arm, count in (("real", reps), ("placebo", placebo)):
        for _ in range(count):
            order = list(range(n))
            random.Random(seed + rep_id).shuffle(order)
            meta.append(
                {
                    "arm": arm,
                    "seed": rep_id,
                    "order": order,
                    "decision_threshold": threshold,
                }
            )
            traj = _fj_trajectories(
                u, w, lam, rounds, 1, seed=seed + 97 * rep_id + 1
            )[0]
            for i in range(n):
                turns.append(
                    {
                        "replicate": rep_id,
                        "round": 0,
                        "member": names[i],
                        "private_position": u[i],
                        "arm": arm,
                    }
                )
            for t in range(rounds):
                for slot, i in enumerate(order):
                    turns.append(
                        {
                            "replicate": rep_id,
                            "round": t + 1,
                            "member": names[i],
                            "order": slot,
                            "arm": arm,
                            "public_position": traj[t][i],
                            "private_position": max(
                                0.0, min(1.0, traj[t][i] - gap)
                            ),
                        }
                    )
            rep_id += 1
    return turns, cast, meta


# ══════════════════ known answers, from theory not from this code ══════════


def test_degroot_room_of_equals_converges_to_the_mean():
    """Everyone averages everyone, nobody is anchored ⇒ the plain mean.

    W = 1/n everywhere and λ = 1 is the DeGroot consensus with a doubly
    stochastic matrix, whose left Perron vector is uniform, so consensus =
    π'u = mean(u). Not a fixed point of this implementation — a theorem.
    """
    u = [0.1, 0.35, 0.5, 0.75, 0.9]
    n = len(u)
    w = [[1.0 / n] * n for _ in range(n)]
    x = equilibrium(w, [1.0] * n, u)
    assert x is not None
    for v in x:
        assert v == pytest.approx(sum(u) / n, abs=1e-4)

    c = influence_centrality(w, innate=u)
    assert c["converges"] is True
    assert c["centrality"] == pytest.approx([1.0 / n] * n, abs=1e-4)
    assert c["consensus_forecast"] == pytest.approx(sum(u) / n, abs=1e-4)


def test_room_with_one_stubborn_oracle_converges_to_the_oracle():
    """One member listens to nobody; everyone else averages ⇒ everyone lands on it.

    Member 0 has W[0] = e_0, so its position never moves. Every other member is
    fully open (λ = 1) and puts weight on 0, so the only closed communicating
    class is {0} and the unique consensus is u_0. The left Perron vector must put
    ALL its mass on the oracle: this is exactly the case where "who talks most"
    and "whose opinion the room lands on" come apart.
    """
    u = [0.9, 0.1, 0.2, 0.3]
    n = len(u)
    w = [[1.0, 0.0, 0.0, 0.0]]
    for i in range(1, n):
        w.append([0.5 if j == 0 else (0.5 if j == i else 0.0) for j in range(n)])
    x = equilibrium(w, [1.0] * n, u)
    assert x is not None
    for v in x:
        assert v == pytest.approx(0.9, abs=1e-4)

    c = influence_centrality(w, innate=u)
    # Reducible — member 0 reaches nobody — but there is exactly ONE closed
    # communicating class, {0}, so the stationary distribution is unique and
    # sits entirely on the oracle. Uniqueness needs one closed class, not strong
    # connectivity, and this is the configuration where the difference matters.
    assert c["irreducible"] is False
    assert c["closed_groups"] == [[0]]
    assert c["converges"] is True
    assert c["centrality"] == pytest.approx([1.0, 0.0, 0.0, 0.0], abs=1e-6)
    assert c["consensus_forecast"] == pytest.approx(0.9, abs=1e-6)


def test_immovable_room_does_not_move():
    """All λ = 0 ⇒ the FJ fixed point is the opening ballot, exactly."""
    u = [0.2, 0.45, 0.8]
    n = len(u)
    w = [[1.0 / n] * n for _ in range(n)]
    x = equilibrium(w, [0.0] * n, u)
    assert x == pytest.approx(u, abs=1e-9)


def test_fully_open_and_fully_stubborn_bracket_the_mixed_room():
    """A partly-anchored room lands between its ballot and the consensus."""
    u = [0.1, 0.5, 0.9]
    n = len(u)
    w = [[1.0 / n] * n for _ in range(n)]
    stubborn = equilibrium(w, [0.0] * n, u)
    open_room = equilibrium(w, [1.0] * n, u)
    mixed = equilibrium(w, [0.5] * n, u)
    assert stubborn == pytest.approx(u, abs=1e-9)
    assert open_room[0] == pytest.approx(0.5, abs=1e-4)
    # Member 0 starts below the consensus and must end between the two.
    assert stubborn[0] < mixed[0] < open_room[0]
    # And the spread must shrink, never invert.
    assert (max(mixed) - min(mixed)) < (max(stubborn) - min(stubborn))


def test_two_cliques_who_ignore_each_other_have_no_single_centrality():
    """Two closed groups ⇒ two stationary distributions ⇒ no centrality exists."""
    w = [
        [0.5, 0.5, 0.0, 0.0],
        [0.5, 0.5, 0.0, 0.0],
        [0.0, 0.0, 0.5, 0.5],
        [0.0, 0.0, 0.5, 0.5],
    ]
    c = influence_centrality(w, innate=[0.1, 0.2, 0.8, 0.9])
    assert c["centrality"] is None
    assert c["converges"] is False
    assert c["consensus_forecast"] is None
    assert c["closed_groups"] == [[0, 1], [2, 3]]
    assert "split into cliques" in c["note"]


def test_periodic_room_is_named_not_averaged():
    """Two members who only listen to each other cycle; π exists, consensus does not.

    W = [[0,1],[1,0]] is irreducible but has period 2, so the plain power
    iteration oscillates forever. The Cesàro limit is the true stationary
    distribution [0.5, 0.5], and it must be reported WITH converges=False rather
    than as a settled consensus.
    """
    c = influence_centrality([[0.0, 1.0], [1.0, 0.0]], innate=[0.2, 0.8])
    assert c["irreducible"] is True
    assert c["primitive"] is False
    assert c["converges"] is False
    assert c["consensus_forecast"] is None
    assert c["centrality"] == pytest.approx([0.5, 0.5], abs=1e-6)
    assert "cycle" in c["note"]


def test_spectral_gap_orders_fast_and_slow_rooms():
    """A uniform room settles in one step; a nearly-decoupled one crawls."""
    n = 4
    fast = [[1.0 / n] * n for _ in range(n)]
    slow = [
        [0.49 if j == i else (0.49 if j == (i + 1) % n else 0.01) for j in range(n)]
        for i in range(n)
    ]
    g_fast = influence_centrality(fast)["spectral_gap"]
    g_slow = influence_centrality(slow)["spectral_gap"]
    # A rank-one row-stochastic matrix has λ₂ = 0 exactly.
    assert g_fast == pytest.approx(1.0, abs=1e-6)
    assert 0.0 < g_slow < g_fast


# ══════════════════ the constrained solver ══════════════════


def test_capped_simplex_projection_is_the_projection():
    """Clip when the sum constraint is slack; Duchi-project when it binds."""
    assert _project_capped_simplex([0.2, -0.5, 0.3]) == pytest.approx([0.2, 0.0, 0.3])
    got = _project_capped_simplex([0.9, 0.9, 0.9])
    assert sum(got) == pytest.approx(1.0)
    assert got == pytest.approx([1 / 3, 1 / 3, 1 / 3])
    # Already feasible vectors are fixed points.
    v = [0.1, 0.2, 0.05]
    assert _project_capped_simplex(v) == pytest.approx(v)
    # Projection never leaves the feasible set, on any input.
    rng = random.Random(3)
    for _ in range(200):
        raw = [rng.uniform(-2, 2) for _ in range(6)]
        out = _project_capped_simplex(raw)
        assert min(out) >= -1e-12
        assert sum(out) <= 1.0 + 1e-9


def test_noiseless_fj_data_is_recovered_almost_exactly():
    """With no noise and enough rounds the fit is the inverse of the simulator."""
    u = [0.1, 0.4, 0.9]
    w = [[0.6, 0.3, 0.1], [0.2, 0.5, 0.3], [0.1, 0.2, 0.7]]
    lam = [0.5, 0.8, 0.3]
    traj = _fj_trajectories(u, w, lam, rounds=6, reps=4, sigma=0.0, quantise=False)
    fit = fit_social_influence(traj, u, names=["a", "b", "c"])
    assert fit["refused"] is None
    for i in range(3):
        assert fit["susceptibility"][i] == pytest.approx(lam[i], abs=0.02)
        for j in range(3):
            assert fit["W"][i][j] == pytest.approx(w[i][j], abs=0.05)


# ══════════════════ refusal paths ══════════════════


def test_one_replicate_of_a_five_seat_room_is_refused():
    """5 members × 4 rounds × 1 replicate is 0.6 obs/parameter: no fit exists.

    This is the case the schema's own docstring names. Measured at that design,
    both W and λ come out further from the truth than an uninformative guess
    (MAE ratios 1.66-1.78 and 1.40-1.60), so a number here would be worse than
    no number.
    """
    u = [0.1, 0.3, 0.5, 0.7, 0.9]
    n = len(u)
    w = [[1.0 / n] * n for _ in range(n)]
    traj = _fj_trajectories(u, w, [0.5] * n, rounds=4, reps=1)
    fit = fit_social_influence(traj, u)
    assert fit["W"] is None
    assert fit["susceptibility"] is None
    assert fit["refused"]
    assert "0.60 per parameter" in fit["refused"]
    assert fit["n_obs_per_member"] == 3
    assert fit["params_per_member"] == 5


def test_adding_replicates_lifts_the_same_room_over_the_gate():
    """Replicates are the cheap axis, and the refusal says so."""
    u = [0.1, 0.3, 0.5, 0.7, 0.9]
    n = len(u)
    w = [[1.0 / n] * n for _ in range(n)]
    lam = [0.5] * n
    refused = fit_social_influence(_fj_trajectories(u, w, lam, 4, 2), u)
    assert refused["refused"]  # 1.2 obs/param
    ok = fit_social_influence(_fj_trajectories(u, w, lam, 4, 3), u)
    assert ok["refused"] is None  # 1.8 obs/param
    assert ok["W"] is not None
    assert len(ok["W"]) == n and all(len(r) == n for r in ok["W"])
    for row in ok["W"]:
        assert sum(row) == pytest.approx(1.0, abs=1e-3)
        assert min(row) >= 0.0


def test_w_entries_are_flagged_unsupported_at_every_realistic_design():
    """The entry-level claim is refused far above the fit-level one.

    Measured: a fitted W entry only beats a uniform-row guess at ~9 observations
    per parameter, and 9 members × 8 rounds × 8 replicates reaches only 6.2. So
    `w_entries_supported` is False across the whole schema, and the caveat has to
    travel with the matrix.
    """
    n = 9
    u = [0.05 + 0.1 * i for i in range(n)]
    w = [[1.0 / n] * n for _ in range(n)]
    fit = fit_social_influence(_fj_trajectories(u, w, [0.5] * n, 8, 8), u)
    assert fit["refused"] is None
    assert fit["obs_per_param"] == pytest.approx(6.222, abs=0.01)
    assert fit["w_entries_supported"] is False
    assert "uniform-row guess" in fit["w_entry_caveat"]
    # A five-seat room at the maximum design does clear it.
    n5 = 5
    u5 = [0.1, 0.3, 0.5, 0.7, 0.9]
    w5 = [[1.0 / n5] * n5 for _ in range(n5)]
    big = fit_social_influence(_fj_trajectories(u5, w5, [0.5] * n5, 8, 8), u5)
    assert big["obs_per_param"] == pytest.approx(11.2, abs=0.01)
    assert big["w_entries_supported"] is True
    assert big["w_entry_caveat"] is None


def test_a_room_where_nobody_moves_manufactures_no_influence():
    """Zero noise, W = I, λ = 0: the fit must return exactly zero, not nearly.

    Measured across 300 member rows: λ = 0.0000 on every one, every off-diagonal
    weight 0.0000, and every row flagged unidentified — because a member that
    never moved carries no evidence about whom it would have listened to.
    """
    u = [0.2, 0.4, 0.6, 0.8, 0.35]
    n = len(u)
    identity = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    traj = _fj_trajectories(u, identity, [0.0] * n, rounds=6, reps=4, sigma=0.0)
    fit = fit_social_influence(traj, u)
    assert fit["refused"] is None
    assert fit["susceptibility"] == [0.0] * n
    for i in range(n):
        assert fit["W"][i][i] == 1.0
        assert sum(fit["W"][i][j] for j in range(n) if j != i) == 0.0
    assert len(fit["unidentified_rows"]) == n
    assert all("never left its opening ballot" in r["reason"] for r in fit["unidentified_rows"])


def test_quantisation_noise_alone_is_caught_by_the_design_matched_null():
    """An immovable room jittered onto the 0.05 grid fits a NONZERO λ — and is flagged.

    This is the defect the runtime null exists for. Measured on rooms where
    nobody moved, the fit reports a median λ of 0.10-0.39 and a p90 as high as
    1.000 depending on the design. So the estimator simulates immovable rooms of
    the same shape and refuses to call any λ inside their p90 a measurement.
    """
    n = 5
    u = [0.2, 0.4, 0.6, 0.8, 0.35]
    identity = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    traj = _fj_trajectories(u, identity, [0.0] * n, rounds=6, reps=4, sigma=0.03)
    fit = fit_social_influence(traj, u)
    assert fit["refused"] is None
    # The bias is real: something nonzero comes out of pure rounding.
    assert max(fit["susceptibility"]) > 0.0
    # And nothing survives the null.
    assert not any(fit["susceptibility_supported"])
    assert fit["susceptibility_null"]["p90"] is not None
    assert "did not measurably move" in fit["susceptibility_null"]["note"]


def test_a_real_room_survives_the_null_floor_the_immovable_one_fails():
    """The gate has to have power, not just a level."""
    n = 5
    u = [0.1, 0.3, 0.5, 0.7, 0.9]
    rng = random.Random(5)
    w = []
    for i in range(n):
        raw = [rng.expovariate(1.0) for _ in range(n)]
        raw[i] += 0.35 * sum(raw)
        s = sum(raw)
        w.append([v / s for v in raw])
    lam = [0.8] * n
    real = fit_social_influence(_fj_trajectories(u, w, lam, 8, 8, seed=4), u)
    identity = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    null = fit_social_influence(
        _fj_trajectories(u, identity, [0.0] * n, 8, 8, seed=4), u
    )
    assert sum(real["susceptibility_supported"]) > sum(null["susceptibility_supported"])
    assert sum(real["susceptibility_supported"]) >= 3


def test_short_or_ragged_input_refuses_rather_than_guessing():
    fit = fit_social_influence([], [0.5, 0.5])
    assert fit["W"] is None and fit["refused"]
    fit = fit_social_influence([[[0.1, 0.2]]], [0.1, 0.2])
    assert fit["W"] is None and "no usable transition" in fit["refused"]
    fit = fit_social_influence([[[0.1, 0.2], [0.2, 0.3]]] * 4, [0.1])
    assert fit["W"] is None and "one position per member" in fit["refused"]


def test_no_influence_matrix_means_no_centrality_and_no_equilibrium():
    """Downstream consumers of a refused fit must get None, never a default."""
    c = influence_centrality(None)
    assert c["centrality"] is None and c["converges"] is False
    assert equilibrium(None, [0.5], [0.5]) is None
    assert counterfactual_absence(None, [0.5], [0.5], ["a"]) == []


# ══════════════════ preference falsification ══════════════════


def test_five_member_gap_gets_no_interval_and_an_exact_test_with_a_floor():
    """n = 5 is below the bootstrap guard, and the sign test cannot reach 0.05.

    `analytics.bootstrap_ci` returns None below n = 8 and this must not be
    papered over. The exact sign-flip enumeration over 2⁵ = 32 assignments has a
    smallest attainable two-sided p of 2/32 = 0.0625, so even a unanimous gap
    cannot clear 0.05 — and the estimator says so instead of pretending.
    """
    n = 5
    public = [[0.6] * n for _ in range(4)]
    private = [[0.4] * n for _ in range(4)]
    res = preference_falsification(public, private, [f"m{i}" for i in range(n)])
    assert res["room_mean_gap"] == pytest.approx(0.2, abs=1e-9)
    assert res["gap_ci"] is None
    assert "below the n ≥ 8 guard" in res["gap_ci_note"]
    assert res["exact_test"] is True
    assert res["p_value"] == pytest.approx(2.0 / 32.0, abs=1e-9)
    assert res["p_floor"] == pytest.approx(2.0 / 32.0, abs=1e-9)
    assert res["supported"] is False  # the floor exceeds 0.05
    assert "more in favour" in res["direction"]


def test_eight_member_gap_gets_both_an_interval_and_a_usable_p():
    n = 8
    public = [[0.5 + 0.02 * i for i in range(n)] for _ in range(3)]
    private = [[0.3 + 0.02 * i for i in range(n)] for _ in range(3)]
    res = preference_falsification(public, private)
    assert res["gap_ci"] is not None
    assert res["p_floor"] == pytest.approx(2.0 / 256.0, abs=1e-4)
    assert res["p_value"] <= 0.05
    assert res["supported"] is True


def test_a_room_with_no_gap_reports_no_gap():
    n = 6
    same = [[0.1 * i for i in range(n)] for _ in range(3)]
    res = preference_falsification(same, same)
    assert res["room_mean_gap"] == 0.0
    assert res["p_value"] == pytest.approx(1.0)
    assert res["supported"] is False
    assert res["direction"] == "no net gap"


def test_the_gap_trajectory_shows_widening():
    """A gap that grows as the room converges is the false-consensus signature."""
    n = 6
    public = [[0.5] * n, [0.6] * n, [0.7] * n, [0.8] * n]
    private = [[0.5] * n, [0.5] * n, [0.5] * n, [0.5] * n]
    res = preference_falsification(public, private)
    assert res["gap_by_round"] == [0.0, 0.1, 0.2, 0.3]
    assert res["gap_widening"] == pytest.approx(0.3, abs=1e-9)


# ══════════════════ conformity calibration, both directions ══════════════


def _rational_room(n, rounds, reps, seed, sigma=0.18):
    """Agents that hold the running mean of 1 + t(n−1) independent signals."""
    rng = random.Random(seed)
    innates, trajs = [], []
    for _ in range(reps):
        own = [rng.gauss(0, sigma) for _ in range(n)]
        innates.append([_q(0.5 + v) for v in own])
        totals = list(own)
        traj = []
        for t in range(1, rounds + 1):
            k = 1 + t * (n - 1)
            for i in range(n):
                totals[i] += rng.gauss(0, sigma * math.sqrt(n - 1))
            traj.append([_q(0.5 + totals[i] / k) for i in range(n)])
        trajs.append(traj)
    return innates, trajs


def _degroot_room(n, rounds, reps, seed, lam):
    rng = random.Random(seed)
    u = [_q(rng.uniform(0.1, 0.9)) for _ in range(n)]
    w = [[1.0 / n] * n for _ in range(n)]
    return u, _fj_trajectories(u, w, [lam] * n, rounds, reps, sigma=0.02, seed=seed + 1)


def test_rational_poolers_read_about_one_and_do_not_fire():
    """Agents that literally obey the benchmark must read ≈ 1.0.

    Measured over 150 trials at 5 × 4 × 3 the ratio's median is 1.12 and the
    CONFORMITY verdict fires 3% of the time against a nominal 5%. This test walks
    40 of them and pins both the location and the level.
    """
    ratios, fires = [], 0
    for t in range(40):
        innate, traj = _rational_room(5, 4, 3, 77_000 + t)
        res = conformity_vs_bayes(traj, innate)
        assert res["refused"] is None
        ratios.append(res["conformity_ratio"])
        if res["supported"]:
            fires += 1
    ratios.sort()
    assert 0.85 <= ratios[len(ratios) // 2] <= 1.45
    assert fires <= 4  # 10% of 40; the measured rate is 3%


def test_conformists_read_above_one_and_do_fire():
    """A DeGroot room at λ = 0.95 discards 95% of its disagreement in one round.

    Measured over 150 trials at 5 × 4 × 3 the ratio's median is 2.04 and the
    verdict fires 85% of the time. Rooms this agreeable are the artefact the
    whole instrument exists to catch.
    """
    ratios, fires = [], 0
    for t in range(40):
        innate, traj = _degroot_room(5, 4, 3, 78_000 + t, 0.95)
        res = conformity_vs_bayes(traj, innate)
        assert res["refused"] is None
        ratios.append(res["conformity_ratio"])
        if res["supported"]:
            fires += 1
            assert res["verdict"].startswith("CONFORMITY")
    ratios.sort()
    assert ratios[len(ratios) // 2] > 1.5
    assert fires >= 26  # 65% of 40; the measured rate is 85%


def test_an_anchored_room_reads_under_converged_not_conformist():
    """λ = 0.05: almost nobody moves. The measure must not read that as agreement."""
    for t in range(10):
        innate, traj = _degroot_room(5, 4, 3, 79_000 + t, 0.05)
        res = conformity_vs_bayes(traj, innate)
        assert res["conformity_ratio"] < 0.5
        assert res["supported"] is False
        assert res["verdict"].startswith("UNDER-CONVERGED")


def test_conformity_refuses_when_the_room_started_in_agreement():
    """No opening spread means no collapse to attribute to anything."""
    n = 5
    traj = [[[0.5] * n for _ in range(4)]]
    res = conformity_vs_bayes(traj, [0.5] * n)
    assert res["conformity_ratio"] is None
    assert "started in agreement" in res["refused"]


def test_conformity_refuses_a_two_member_or_one_round_room():
    res = conformity_vs_bayes([[[0.1, 0.9], [0.2, 0.8]]], [0.1, 0.9])
    assert res["refused"] and "at least 3 members" in res["refused"]
    res = conformity_vs_bayes([[[0.1, 0.5, 0.9]]], [0.1, 0.5, 0.9])
    assert res["refused"]


def test_the_headline_ratio_is_the_worst_round_not_the_last():
    """A room that capitulates in round 1 and then sits still is still conformist.

    Measured: the endpoint version of this statistic read 1.34 for a λ = 0.85
    DeGroot room at 4 rounds but 1.09 at 8 — watching a conformist for longer
    made it look more rational, because the benchmark kept accruing while the
    room had already finished. The max over rounds is what makes it a timing
    claim rather than an endpoint one.
    """
    innate, traj = _degroot_room(5, 8, 3, 80_001, 0.95)
    res = conformity_vs_bayes(traj, innate)
    assert res["worst_round"] <= 2
    assert res["conformity_ratio"] == max(res["ratio_by_round"])
    assert res["conformity_ratio"] > res["ratio_by_round"][-1]


# ══════════════════ cascade, coalitions, order ══════════════════


def test_cascade_needs_the_speaking_order_and_says_so():
    n = 4
    pub = [[[0.5] * n for _ in range(4)]]
    res = cascade_scores(pub, pub, [], [f"m{i}" for i in range(n)])
    assert res["refused"] and "speaking order" in res["refused"]


def test_a_member_who_always_speaks_first_cannot_be_scored():
    """No predecessors anywhere means no score — reported, not zeroed."""
    n = 4
    rng = random.Random(9)
    pub = [
        [[_q(rng.uniform(0, 1)) for _ in range(n)] for _ in range(6)] for _ in range(3)
    ]
    orders = [[0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
    res = cascade_scores(pub, pub, orders, ["a", "b", "c", "d"], innate=[0.5] * n)
    first = next(m for m in res["per_member"] if m["name"] == "a")
    assert first["n_obs"] == 0
    assert first["cascade_r2_gain"] is None
    assert any(r["name"] == "a" for r in res["refusals"])
    assert "rotate the speaking order" in res["refusals"][0]["reason"]


def test_a_pure_herder_scores_high_and_an_independent_scores_zero():
    """Member 1 copies whoever spoke before it; member 2 ignores everyone."""
    n = 3
    rng = random.Random(21)
    reps, privs = [], []
    for _ in range(6):
        pub_rep, prv_rep = [], []
        for _t in range(6):
            leader = _q(rng.uniform(0, 1))
            own1 = _q(rng.uniform(0, 1))
            own2 = _q(rng.uniform(0, 1))
            # index 0 speaks first, 1 herds on it, 2 states its own private view
            pub_rep.append([leader, leader, own2])
            prv_rep.append([leader, own1, own2])
        reps.append(pub_rep)
        privs.append(prv_rep)
    orders = [[0, 1, 2]] * 6
    res = cascade_scores(reps, privs, orders, ["lead", "herder", "own"], innate=[0.5] * n)
    by_name = {m["name"]: m for m in res["per_member"]}
    assert by_name["herder"]["cascade_r2_gain"] > 0.8
    assert by_name["own"]["cascade_r2_gain"] < 0.2
    assert res["most_herding"] == "herder"


def test_camps_split_on_a_canyon_and_not_on_rounding():
    two = coalitions([0.1, 0.15, 0.85, 0.9], None, names=["a", "b", "c", "d"])
    assert two["n_public_camps"] == 2
    assert two["public_camps"][1]["between_gap"] == pytest.approx(0.75, abs=1e-4)
    one = coalitions([0.45, 0.5, 0.55, 0.6], None)
    assert one["n_public_camps"] == 1


def test_a_hidden_split_is_a_public_consensus_nobody_holds():
    res = coalitions([0.6] * 4, [0.1, 0.15, 0.9, 0.95])
    assert res["n_public_camps"] == 1
    assert res["n_private_camps"] == 2
    assert res["hidden_split"] is True


def test_power_indices_are_exact_and_sum_to_one():
    n = 5
    w = [[1.0 / n] * n for _ in range(n)]
    res = coalitions(
        [0.9, 0.8, 0.7, 0.2, 0.1], [0.9] * 5, w, threshold=0.5, names=list("abcde")
    )
    ss = [r["index"] for r in res["shapley_shubik"]]
    bz = [r["index"] for r in res["banzhaf"]]
    assert sum(ss) == pytest.approx(1.0, abs=1e-6)
    assert sum(bz) == pytest.approx(1.0, abs=1e-6)
    # Equal influence weights ⇒ equal power, which the note must own up to.
    assert ss == pytest.approx([1.0 / n] * n, abs=1e-6)
    assert res["vote"]["yes"] == 3
    assert res["vote"]["carries_by_headcount"] is True
    assert "2^(n−1) = 16" in res["power_note"]


def test_a_dominant_listener_takes_more_than_its_share_of_power():
    """When the room listens mostly to one member, its index must exceed 1/n."""
    n = 4
    w = [[0.7 if j == 0 else 0.1 for j in range(n)] for _ in range(n)]
    res = coalitions([0.9, 0.6, 0.4, 0.1], None, w, threshold=0.5, names=list("abcd"))
    ss = [r["index"] for r in res["shapley_shubik"]]
    assert ss[0] > 1.0 / n
    assert res["vote"]["weight_source"] == "influence centrality from the fitted W"


def test_order_effect_refuses_a_single_replicate():
    n = 4
    res = order_effect([[[0.5] * n for _ in range(4)]], [[0, 1, 2, 3]])
    assert res["refused"] and "never varied" in res["refused"]


def test_order_effect_reports_the_permutation_floor_it_cannot_beat():
    """Three replicates give 3! = 6 assignments, so the pull test floors at 0.333."""
    n = 4
    rng = random.Random(31)
    reps = [
        [[_q(rng.uniform(0, 1)) for _ in range(n)] for _ in range(4)] for _ in range(3)
    ]
    orders = [[0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 0, 1]]
    res = order_effect(reps, orders)
    assert res["refused"] is None
    assert res["pull_test_exact"] is True
    assert res["p_floor"] == pytest.approx(2.0 / 6.0, abs=1e-4)
    assert res["pull_p_value"] >= res["p_floor"] - 1e-9


def test_identical_replicates_show_no_order_effect():
    n = 4
    one = [[0.1, 0.4, 0.6, 0.9], [0.2, 0.4, 0.6, 0.8], [0.3, 0.45, 0.55, 0.7]]
    res = order_effect([one, one, one], [[0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 0, 1]])
    assert res["landing_sd"] == 0.0
    assert res["eta_squared"] == 0.0
    assert res["eta_p_value"] == pytest.approx(1.0)


# ══════════════════ the placebo control ══════════════════


def test_no_placebo_arm_is_named_as_an_absent_control():
    res = placebo_contrast({"convergence_ratio": 0.3, "mean_gap": 0.1}, None)
    assert res["ran"] is False
    assert res["refused"] == "placebo_replicates = 0"
    assert "NO CONTROL WAS RUN" in res["verdict"]


def test_a_placebo_that_converges_as_hard_voids_the_finding():
    real = {"convergence_ratio": 0.20, "mean_gap": 0.12, "n_replicates": 3}
    placebo = {"convergence_ratio": 0.18, "mean_gap": 0.12, "n_replicates": 1}
    res = placebo_contrast(real, placebo)
    assert res["separation"].startswith("none")
    assert "MEASURED NOTHING" in res["verdict"]
    assert "same in both arms" in res["gap_note"]


def test_a_clearly_separated_placebo_licenses_a_qualified_claim():
    real = {"convergence_ratio": 0.20, "mean_gap": 0.15, "n_replicates": 4}
    placebo = {"convergence_ratio": 0.80, "mean_gap": 0.02, "n_replicates": 2}
    res = placebo_contrast(real, placebo)
    assert res["separation"].startswith("clear")
    assert "not a tested one" in res["verdict"]


def test_a_marginal_separation_is_called_marginal():
    real = {"convergence_ratio": 0.30, "mean_gap": 0.10, "n_replicates": 3}
    placebo = {"convergence_ratio": 0.33, "mean_gap": 0.10, "n_replicates": 1}
    res = placebo_contrast(real, placebo)
    assert "marginal" in res["separation"]
    assert "unresolved" in res["verdict"]


def test_arm_summary_shape_is_the_contract_placebo_contrast_expects():
    n = 5
    u = [0.1, 0.3, 0.5, 0.7, 0.9]
    w = [[1.0 / n] * n for _ in range(n)]
    pub = _fj_trajectories(u, w, [0.8] * n, 4, 3)
    prv = [[[max(0.0, v - 0.05) for v in row] for row in rep] for rep in pub]
    summary = arm_summary(pub, prv, u)
    for key in (
        "n_members",
        "n_rounds",
        "n_replicates",
        "sd_innate",
        "sd_final",
        "convergence_ratio",
        "mean_gap",
        "gap_final_round",
        "conformity_ratio",
    ):
        assert key in summary
    assert summary["n_members"] == n
    assert summary["convergence_ratio"] < 1.0  # an open room converges
    assert summary["mean_gap"] == pytest.approx(0.05, abs=0.02)


# ══════════════════ counterfactual absence ══════════════════


def test_removing_the_oracle_moves_the_room_and_removing_a_bystander_does_not():
    """The member whose absence matters is not the one who talks most."""
    n = 4
    # 0 is an oracle everyone weights heavily; 3 is ignored by everyone but itself.
    w = [
        [1.0, 0.0, 0.0, 0.0],
        [0.7, 0.3, 0.0, 0.0],
        [0.7, 0.0, 0.3, 0.0],
        [0.3, 0.0, 0.0, 0.7],
    ]
    u = [0.95, 0.2, 0.25, 0.3]
    rows = counterfactual_absence(w, [0.9, 0.9, 0.9, 0.9], u, list("abcd"))
    assert [r["name"] for r in rows][0] == "a"
    assert rows[0]["abs_delta"] > 0.1
    assert rows[-1]["abs_delta"] < rows[0]["abs_delta"]
    assert rows[0]["per_member_delta"]


def test_a_survivor_who_trusted_only_the_departed_is_flagged_orphaned():
    n = 3
    w = [[0.5, 0.5, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
    rows = counterfactual_absence(w, [0.9, 0.9, 0.9], [0.2, 0.8, 0.4], list("abc"))
    removed_b = next(r for r in rows if r["name"] == "b")
    assert "c" in removed_b["orphaned"]


# ══════════════════ the assembled result ══════════════════


def _default_room(**kwargs):
    n = 5
    u = [0.1, 0.3, 0.5, 0.7, 0.9]
    rng = random.Random(2)
    w = []
    for i in range(n):
        raw = [rng.expovariate(1.0) for _ in range(n)]
        raw[i] += 0.35 * sum(raw)
        s = sum(raw)
        w.append([v / s for v in raw])
    return _turns(u, w, [0.6] * n, kwargs.pop("rounds", 4), kwargs.pop("reps", 3), **kwargs)


def test_room_analysis_returns_every_documented_key_and_is_json_serialisable():
    turns, cast, meta = _default_room(placebo=1, threshold=0.5)
    res = room_analysis(turns, cast, "Should we ship the thing?", replicates_meta=meta)
    for key in (
        "run_id",
        "motion",
        "names",
        "n_members",
        "n_rounds",
        "n_replicates",
        "n_placebo_replicates",
        "dropped_replicates",
        "roster",
        "trajectory",
        "influence",
        "centrality",
        "equilibrium",
        "absence",
        "falsification",
        "conformity",
        "cascade",
        "coalitions",
        "order_effect",
        "arms",
        "placebo",
        "consensus",
        "refusals",
        "timing_ms",
        "method",
        "limits",
    ):
        assert key in res, key
    assert res["n_members"] == 5
    assert res["n_rounds"] == 4
    assert res["n_replicates"] == 3
    assert res["n_placebo_replicates"] == 1
    assert len(res["roster"]) == 5
    assert res["roster"][0]["name"] == "m0"
    assert res["coalitions"]["shapley_shubik"] is not None
    json.dumps(res)  # must not raise


def test_room_analysis_is_deterministic():
    turns, cast, meta = _default_room(placebo=1, threshold=0.5)
    a = room_analysis(turns, cast, "Motion", replicates_meta=meta)
    b = room_analysis(turns, cast, "Motion", replicates_meta=meta)
    a.pop("timing_ms")
    b.pop("timing_ms")
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_room_analysis_stays_inside_its_runtime_budget():
    """9 members × 8 rounds × 8 replicates + 4 placebo, measured at 443 ms.

    The generous ceiling here is for slower CI hardware; the point of the test is
    that a change which makes this seconds-slow fails rather than ships.
    """
    n = 9
    u = [0.05 + 0.1 * i for i in range(n)]
    rng = random.Random(4)
    w = []
    for i in range(n):
        raw = [rng.expovariate(1.0) for _ in range(n)]
        raw[i] += 0.35 * sum(raw)
        s = sum(raw)
        w.append([v / s for v in raw])
    turns, cast, meta = _turns(u, w, [0.6] * n, 8, 8, placebo=4, threshold=0.5)
    res = room_analysis(turns, cast, "Motion", replicates_meta=meta)
    assert res["timing_ms"] < 3000
    assert res["n_replicates"] == 8
    assert res["n_placebo_replicates"] == 4


def test_an_incomplete_replicate_is_dropped_whole_and_named():
    turns, cast, meta = _default_room(reps=4)
    # Delete one member's turn from replicate 2; the whole replicate must go.
    turns = [
        t
        for t in turns
        if not (t["replicate"] == 2 and t["round"] == 2 and t["member"] == "m3")
    ]
    res = room_analysis(turns, cast, "Motion", replicates_meta=meta)
    assert res["n_replicates"] == 3
    assert any("replicate 2" == d["what"] for d in res["dropped_replicates"])


def test_no_usable_replicate_returns_a_refusal_shaped_result():
    turns, cast, meta = _default_room(reps=1)
    turns = [t for t in turns if t["round"] == 0]
    res = room_analysis(turns, cast, "Motion", replicates_meta=meta)
    assert res["n_replicates"] == 0
    assert res["roster"] == []
    assert any(d["what"] == "whole study" for d in res["refusals"])
    assert res["placebo"]["ran"] is False
    json.dumps(res)


def test_every_refusal_reaches_the_top_level_refusals_list():
    """A researcher must be able to see what the study could not answer."""
    turns, cast, meta = _default_room(rounds=4, reps=1)
    res = room_analysis(turns, cast, "Motion", replicates_meta=meta)
    reasons = {d["what"] for d in res["refusals"]}
    assert "influence matrix" in reasons  # 0.6 obs/param
    assert "order effect" in reasons  # a single replicate
    assert "placebo control" in reasons  # no placebo arm was requested
    assert res["influence"]["W"] is None
    assert res["equilibrium"]["positions"] is None


def test_the_roster_keeps_declared_and_fitted_dispositions_apart():
    """The schema promises both; overwriting one with the other loses the finding."""
    turns, cast, meta = _default_room(reps=3)
    res = room_analysis(turns, cast, "Motion", replicates_meta=meta)
    row = res["roster"][0]
    assert row["declared_openness"] == 0.5
    assert row["fitted_susceptibility"] is not None
    assert row["declared_vs_fitted"] == pytest.approx(
        row["fitted_susceptibility"] - 0.5, abs=1e-6
    )


def test_limits_and_method_are_present_and_say_the_hard_things():
    turns, cast, meta = _default_room(reps=3)
    res = room_analysis(turns, cast, "Motion", replicates_meta=meta)
    assert "language-model characters, not people" in res["limits"]
    assert "not recoverable" in res["limits"]
    assert "placebo" in res["limits"]
    assert "Friedkin-Johnsen" in res["method"]


def test_turn_objects_are_accepted_as_well_as_dicts():
    """The orchestration layer holds pydantic models, not dicts."""

    class Turn:
        def __init__(self, **kw):
            for k, v in kw.items():
                setattr(self, k, v)

    turns, cast, meta = _default_room(reps=3)
    objects = [Turn(**t) for t in turns]
    res = room_analysis(objects, cast, "Motion", replicates_meta=meta)
    assert res["n_replicates"] == 3
    assert res["influence"]["refused"] is None
