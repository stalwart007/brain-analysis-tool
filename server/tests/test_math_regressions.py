"""Regression guards for statistical bugs that produced wrong user-facing
conclusions.

Every test here was written against a *verified* counterexample — the value in
the comment is what the code actually returned before the fix, not a
hypothetical. They exist because each of these failures was silent: the output
was well-formed, plausible, and wrong.
"""

from __future__ import annotations

import math
import random

import pytest

from app.analytics import (
    consensus_label,
    price_elasticity,
    shannon_entropy_normalised,
)
from app.cognition import markov_transitions, powerlaw_vs_exponential, summarise_profile
from app.neuro import peak_end_memory
from app.virality import simulate_cascades, virality_result


# ── entropy normalisation ────────────────────────────────────────────────


def test_entropy_normalises_against_the_possible_outcome_set():
    """A two-way split must not score the same as a flat four-way spread.

    Both returned 1.0 before, because entropy was normalised by the number of
    *observed* outcomes. That made "half convert, half abandon" — a decisive
    split — indistinguishable from "no structure at all".
    """
    two_way = shannon_entropy_normalised([10, 0, 0, 10], k=4)
    four_way = shannon_entropy_normalised([5, 5, 5, 5], k=4)
    assert two_way == pytest.approx(math.log(2) / math.log(4))  # 0.5
    assert four_way == pytest.approx(1.0)
    assert two_way < four_way


def test_entropy_k_defaults_to_len_counts():
    """Callers that already pad with explicit zeros need no `k`."""
    assert shannon_entropy_normalised([5, 5]) == pytest.approx(1.0)
    assert shannon_entropy_normalised([10, 0, 0, 0]) == 0.0


def test_two_camp_split_still_reads_as_polarised():
    """The polarisation gate had to move when the entropy scale was fixed.

    A clean two-way split scores 0.5 under correct normalisation; the old
    `entropy > 0.5` threshold would have excluded the exact case it existed to
    catch.
    """
    assert consensus_label(0.5, 0.9) == "polarised"
    # …but a flat four-way spread is absence of signal, not polarisation
    assert consensus_label(1.0, None) == "split"


# ── degenerate regression ────────────────────────────────────────────────


def test_flat_demand_is_perfectly_inelastic_not_a_prestige_good():
    """R² is undefined for a constant response, and 1.0 is not a safe stand-in.

    Before: {'elasticity': 0.0, 'elasticity_r2': 1.0, 'demand_fit_r2': 1.0,
             'classification': 'anomalous — demand rises with price
              (prestige effect …)'}
    """
    r = price_elasticity([10, 20, 30, 40], [0.5, 0.5, 0.5, 0.5])
    assert r["ok"] is True
    assert r["elasticity_r2"] is None
    assert r["demand_fit_r2"] is None
    assert "perfectly inelastic" in r["classification"]
    assert "prestige" not in r["classification"]


def test_genuine_elasticity_is_still_recovered():
    """The degenerate guard must not blunt the normal path."""
    prices = [10.0, 20.0, 30.0, 40.0]
    demand = [p**-1.5 for p in prices]  # constant elasticity −1.5
    r = price_elasticity(prices, demand)
    assert r["elasticity"] == pytest.approx(-1.5, abs=0.01)
    assert r["elasticity_r2"] is not None and r["elasticity_r2"] > 0.99
    assert "elastic" in r["classification"]


# ── HMM occupancy ────────────────────────────────────────────────────────


def test_duplicate_state_labels_do_not_lose_occupancy_mass():
    """_label_states names by argmax emission, so two `scroll`-dominated states
    share the label "scanning". Keying a dict on the label overwrote the first.

    Before: {'scanning': 0.2, 'goal pursuit': 0.5} — sums to 0.7.
    """
    profile = {
        "hmm": {
            "state_path": [0, 0, 0, 1, 1, 2, 2, 2, 2, 2],
            "states": [
                {"label": "scanning"},
                {"label": "scanning"},
                {"label": "goal pursuit"},
            ],
        }
    }
    out = summarise_profile(profile)
    occ = out["state_occupancy"]
    assert sum(occ.values()) == pytest.approx(1.0, abs=1e-9)
    assert occ["scanning"] == pytest.approx(0.5)
    assert occ["goal pursuit"] == pytest.approx(0.5)


# ── Markov predictability ────────────────────────────────────────────────


def test_deterministic_sequence_reads_as_predictable():
    """A flat unit prior over 7×7 = 49 cells swamped a ~30-event session.

    Before: a perfectly deterministic alternating sequence scored
    predictability 0.242 at n=30 — and that number is handed to the profiler
    as "the strongest evidence available".
    """
    seq = [0, 1] * 15  # perfectly predictable, true entropy rate 0
    m = markov_transitions(seq, 7)
    assert m["n_symbols_observed"] == 2
    assert m["predictability"] > 0.75
    assert m["entropy_rate_bits"] < 0.25


def test_predictability_improves_with_evidence():
    short = markov_transitions([0, 1] * 15, 7)["predictability"]
    long = markov_transitions([0, 1] * 100, 7)["predictability"]
    assert long > short  # the prior washes out as data accumulates


def test_random_sequence_is_not_predictable():
    rng = random.Random(4)
    seq = [rng.randrange(7) for _ in range(400)]
    m = markov_transitions(seq, 7)
    assert m["predictability"] < 0.15


def test_transition_matrix_keeps_full_alphabet_shape():
    """The UI renders this against fixed event labels, so the shape is a
    contract even when only some symbols occur."""
    m = markov_transitions([0, 1] * 15, 7)
    assert len(m["transition"]) == 7
    assert all(len(row) == 7 for row in m["transition"])
    assert len(m["stationary"]) == 7
    assert sum(m["stationary"]) == pytest.approx(1.0, abs=1e-6)


# ── tail index ───────────────────────────────────────────────────────────


def test_xmin_is_estimated_not_assumed():
    """Clauset's method exists to avoid fitting the power law to the body.

    Before, with xmin = min(xs), cascade-shaped data with a large atom at 1
    returned alpha = 10.37 — far outside any Lévy range — because those points
    contribute log(1) = 0 to the sum and inflate 1 + n/Σlog.
    """
    finals = [1.0] * 1800 + [2.0] * 120 + [3.0] * 50 + [5.0] * 20 + [40.0] * 8 + [900.0] * 2
    r = powerlaw_vs_exponential(finals)
    assert r is not None
    assert r["xmin"] > 1.0  # the atom is excluded from the fit
    assert r["alpha"] < 5.0


def test_tail_index_recovers_a_known_pareto():
    """Ground truth: sampling (1−U)^(−1/1.5) gives a Pareto with density
    exponent α = 2.5."""
    rng = random.Random(5)
    xs = [(1.0 - rng.random()) ** (-1 / 1.5) for _ in range(2000)]
    r = powerlaw_vs_exponential(xs)
    assert r is not None
    assert r["alpha"] == pytest.approx(2.5, abs=0.1)
    assert r["alpha_se"] is not None and r["alpha_se"] < 0.1
    assert r["regime"] == "levy_like"


def test_ambiguous_sample_refuses_to_pick_a_regime():
    """A sign check on a per-sample mean coin-flips when the models are
    genuinely indistinguishable. Vuong gives a p-value, so it can abstain."""
    rng = random.Random(3)
    xs = [rng.expovariate(1.0) + 0.1 for _ in range(12)]
    r = powerlaw_vs_exponential(xs)
    assert r is not None
    assert r["regime"] == "indistinguishable"
    assert r["p_value"] > 0.10


def test_xmin_search_keeps_enough_tail_to_have_power():
    """Unconstrained KS minimisation walks xmin to the 96th percentile on
    non-power-law data, leaving too few points to reach any verdict."""
    rng = random.Random(13)
    xs = [rng.expovariate(1.0) + 0.05 for _ in range(2000)]
    r = powerlaw_vs_exponential(xs)
    assert r is not None
    assert r["n"] >= 0.10 * r["n_total"]
    assert r["regime"] == "brownian_like"


# ── branching process ────────────────────────────────────────────────────


def test_supercritical_cascade_is_not_reported_as_local():
    """The size cap `break`-ed out of the parent loop mid-generation, which
    strangled the process rather than truncating it, and piled every capped run
    onto a fake atom at SIZE_CAP.

    Before: a run at R0 = 2.03 reported
    criticality {'alpha': 1.11, 'regime': 'brownian_like'} →
    "exponentially bounded cascade sizes — spread stays local and predictable".
    """
    ps = [0.30, 0.34, 0.38, 0.28, 0.36, 0.32]  # R0 = 6 × ~0.33 ≈ 2.0
    res = virality_result(ps, k=6, load="low")
    assert res["supercritical"] is True
    assert res["criticality"]["regime"] == "explosive"
    assert "unbounded" in res["criticality"]["interpretation"]
    # and the size headline must exist in the supercritical regime
    assert res["simulated_size"]["median"] is not None


def test_subcritical_cascade_still_reports_a_finite_size():
    ps = [0.02, 0.03, 0.025, 0.018]  # R0 = 3 × ~0.02 ≈ 0.07
    res = virality_result(ps, k=3, load="low")
    assert res["supercritical"] is False
    assert res["expected_cascade_size"] is not None
    assert res["criticality"] is None or res["criticality"]["regime"] != "explosive"


def test_censored_runs_are_flagged_and_counted():
    ps = [0.5] * 5
    sims = simulate_cascades(ps, k=6, n_sims=60, seed=3)
    assert len(sims["censored"]) == len(sims["final_sizes"]) == 60
    assert sims["censored_frac"] > 0.0
    # a censored run is a lower bound pinned at the cap, never above it
    for size, was_censored in zip(sims["final_sizes"], sims["censored"]):
        if was_censored:
            assert size == 50_000


def test_extinction_ci_uses_general_percentiles():
    """Hard-coded indices [4] and [194] were only correct for exactly 200
    replicates and would have silently become the wrong quantiles."""
    ps = [0.1, 0.2, 0.15, 0.25, 0.12]
    res = virality_result(ps, k=4, load="low")
    ci = res["extinction_q_ci"]
    assert ci is not None and len(ci) == 2
    assert 0.0 <= ci[0] <= ci[1] <= 1.0


# ── peak-end memory ──────────────────────────────────────────────────────


def test_memorability_does_not_track_the_beat_count():
    """`max(recall_probability)` is a softmax over beats, so it falls as ~1/n:
    identical flat content scored 0.449 at 3 beats and 0.235 at 8, moving the
    "memory" bar on the brain map ~2× for a segmentation choice.
    """
    strengths = []
    for n_seg in (3, 5, 8):
        m = peak_end_memory([0.5] * n_seg, [0.2] * n_seg)
        assert m is not None
        strengths.append(m["encoding_strength"])
    assert max(strengths) - min(strengths) < 1e-9  # fully invariant
    # and it reflects the actual driver — arousal
    hot = peak_end_memory([0.2, 0.95, 0.3], [0.1, 0.1, 0.1])
    cold = peak_end_memory([0.2, 0.25, 0.3], [0.1, 0.1, 0.1])
    assert hot is not None and cold is not None
    assert hot["encoding_strength"] > cold["encoding_strength"]


def test_recall_concentration_is_scale_free():
    """0 = flat piece with no standout beat, 1 = one beat takes everything."""
    flat = peak_end_memory([0.5] * 6, [0.2] * 6)
    spiky = peak_end_memory([0.1, 0.1, 0.99, 0.1, 0.1, 0.1], [0.2] * 6)
    assert flat is not None and spiky is not None
    assert spiky["recall_concentration"] > flat["recall_concentration"]
    for m in (flat, spiky):
        assert 0.0 <= m["recall_concentration"] <= 1.0
