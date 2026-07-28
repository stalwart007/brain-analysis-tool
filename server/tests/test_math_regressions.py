"""Regression guards for statistical bugs that produced wrong user-facing
conclusions.

Every test here was written against a *verified* counterexample — the value in
the comment is what the code actually returned before the fix, not a
hypothetical. They exist because each of these failures was silent: the output
was well-formed, plausible, and wrong.
"""

from __future__ import annotations

import json
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


# ── EZ-diffusion: the accuracy must be observed, never manufactured ───────


def test_drift_rate_is_antisymmetric_about_chance():
    """Why a fabricated accuracy was not a safe fallback.

    `v` flips sign about p = 0.5 with identical magnitude. The old
    `_cognition_inputs` proxy took only the values 1.0 and 0.0 (the subtracted
    `rage_click_bursts * 3` term always exceeded the click count once any burst
    existed), so a single triple-click moved the reported drift rate from
    +0.2787 to -0.2787 on unchanged latencies — while the interpretation
    string still read "efficient evidence accumulation".
    """
    import statistics

    from app.cognition import ez_diffusion

    rng = random.Random(3)
    lat = [max(0.05, rng.gauss(0.7, 0.18)) for _ in range(24)]
    m, v, n = statistics.fmean(lat), statistics.pvariance(lat), len(lat)

    hi = ez_diffusion(m, v, 1.0, n, latencies_s=lat)
    lo = ez_diffusion(m, v, 0.0, n, latencies_s=lat)
    assert hi is not None and lo is not None
    assert hi["drift_rate"] == pytest.approx(-lo["drift_rate"])
    assert abs(hi["drift_rate"]) > 0.1  # the swing is large, not marginal


def test_cognition_inputs_never_fabricates_a_completion_ratio():
    """Browser telemetry carries no success/abandon signal, so there is none.

    Guards the seam rather than the estimator: `ez_diffusion` already refuses
    without an observed accuracy, and the bug was that its caller supplied one
    anyway. Any future proxy assembled from click counts must fail here.
    """
    from app.main import _cognition_inputs
    from app.schemas import FeaturePayload

    for clicks, bursts in ((0, 0), (5, 0), (5, 1), (40, 3)):
        feats = FeaturePayload(
            segment_start=0.0,
            segment_end=10_000.0,
            event_count=max(clicks, 1),
            click_count=clicks,
            rage_click_bursts=bursts,
        )
        assert _cognition_inputs(feats)["completion_ratio"] is None


def test_ddm_refuses_without_completion_signal_and_says_why():
    """The refusal must be visible in the payload, not a silent omission."""
    from app.cognition import iter_cognitive_profile

    rng = random.Random(11)
    stream = [[float(i * 40), i % 7] for i in range(260)]
    profile = None
    for frame in iter_cognitive_profile(
        event_stream=stream,
        velocity_series=[[float(i * 40), rng.random()] for i in range(120)],
        decision_latencies_ms=[rng.gauss(700, 180) for _ in range(24)],
        completion_ratio=None,
    ):
        if frame.get("stage") == "complete":
            profile = frame.get("profile") or profile
    assert profile is not None
    assert "ez_diffusion" not in profile["models_run"]
    assert "ez_diffusion" in profile["models_skipped"]
    assert "completion signal" in profile["ddm_skipped_reason"]


# ── change_points must be symmetric about the middle of the series ────────


def test_change_points_detects_a_collapse_at_the_ending():
    """The loop bound `range(min_size, m - min_size)` excluded k = m - min_size,
    which is admissible — it leaves a right segment of exactly min_size, the
    same size the left segment is allowed. Verified before the fix: a collapse
    in the ending returned [] while the mirror-image collapse in the opening
    returned [1]. For content analysis that is the worst possible asymmetry,
    since peak-end theory says the ending is what gets remembered.
    """
    from app.neuro import change_points

    ending = change_points([0.8] * 7 + [0.1], min_size=1, n_obs=20)
    opening = change_points([0.8] + [0.1] * 7, min_size=1, n_obs=20)
    assert ending, "a collapse in the final beat must be detectable"
    assert opening, "sanity: the mirror image was already detected"
    # mirror-image series must yield mirror-image split counts
    assert len(ending) == len(opening)


def test_change_points_is_mirror_symmetric_on_reversed_input():
    from app.neuro import change_points

    rng = random.Random(5)
    series = [0.2 + rng.random() * 0.05 for _ in range(6)] + [
        0.85 + rng.random() * 0.05 for _ in range(6)
    ]
    fwd = change_points(series, min_size=2, n_obs=20)
    rev = change_points(list(reversed(series)), min_size=2, n_obs=20)
    assert len(fwd) == len(rev) == 1
    assert fwd[0] + rev[0] == len(series)  # split reflects about the midpoint


# ── persona trait shifts: absence of evidence must not move a trait ───────


def _signal(**kw):
    from app.schemas import BehavioralSignal

    base = dict(
        segment_label="x",
        evidence="e",
        likely_mindset="m",
        confidence=1.0,
        deliberation="medium",
        frustration_signal="none",
        exploration_style="reader",
        price_sensitivity_signal="unknown",
    )
    base.update(kw)
    return BehavioralSignal(**base)


def test_absence_of_rage_clicks_is_not_evidence_of_calm():
    """`none` was -0.1, so an unremarkable session — the overwhelmingly common
    case, and the only thing this telemetry can observe — was pushed BELOW the
    documented baseline as though calm had been measured. Absence of evidence
    must leave the trait at its prior; only observed frustration may move it.
    """
    from app.persona import BASELINE, derive_traits

    n = {
        lv: derive_traits(_signal(frustration_signal=lv)).neuroticism.score
        for lv in ("none", "mild", "elevated")
    }
    assert n["none"] == pytest.approx(BASELINE, abs=1e-9)
    assert n["mild"] > BASELINE
    assert n["elevated"] > n["mild"]


def test_neutral_levels_sit_exactly_on_the_baseline():
    """Each vocabulary's neutral level is its semantic zero, so a session with
    nothing remarkable in it produces the population prior, not a persona."""
    from app.persona import BASELINE, derive_traits

    neutral = derive_traits(_signal())
    for trait in ("openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"):
        assert getattr(neutral, trait).score == pytest.approx(BASELINE, abs=1e-9), trait


def test_trait_shifts_stay_monotone_in_their_evidence():
    from app.persona import derive_traits

    o = {
        lv: derive_traits(_signal(exploration_style=lv)).openness.score
        for lv in ("goal_directed", "reader", "scanner", "wandering")
    }
    assert o["goal_directed"] < o["reader"] < o["scanner"] < o["wandering"]
    c = {
        lv: derive_traits(_signal(deliberation=lv)).conscientiousness.score
        for lv in ("low", "medium", "high")
    }
    assert c["low"] < c["medium"] < c["high"]


# ── virality: the two censoring bounds are different findings ─────────────


def test_capped_cascades_are_not_recorded_as_extinct():
    """`active_by_gen` stayed at its 0 initialiser for every generation after
    the size cap, so `alive_frac` read 0.000 for exactly the runs that
    exploded. Verified at R0 = 5.4: the survival curve showed the epidemic
    going extinct at generation 8, precisely because every run blew past
    SIZE_CAP there.
    """
    sims = simulate_cascades([0.9] * 6, 6, n_sims=200)
    assert sims["capped_frac"] > 0.5, "fixture must actually hit the cap"
    for row in sims["generations"]:
        assert row["alive_frac"] > 0.5, f"generation {row['g']} reads extinct"
        # active can never exceed cumulative — both are clamped to the bound
        assert row["active_q50"] <= row["cum_q50"]


def test_horizon_censoring_is_not_reported_as_hitting_the_size_cap():
    """At R0 = 1.2 roughly a third of runs are still alive at generation 20 and
    *none* reach the 50,000-node cap. Summing the two causes made the verdict
    print "36.7% of simulated cascades exceeded the 50,000-node simulation
    bound — growth is unbounded", which was false in both clauses.
    """
    sims = simulate_cascades([0.2] * 6, 6, n_sims=400)
    assert sims["horizon_frac"] > 0.05, "fixture must actually censor at horizon"
    assert sims["capped_frac"] == 0.0
    assert sims["censored_frac"] == pytest.approx(
        sims["capped_frac"] + sims["horizon_frac"], abs=1e-6
    )

    crit = virality_result([0.2] * 6, 6, "low")["criticality"]
    assert crit["regime"] == "unresolved"
    assert "exceeded" not in crit["interpretation"]
    assert "still spreading" in crit["interpretation"]


def test_genuinely_explosive_growth_still_reads_explosive():
    """The new gate must not blunt the case it exists to report."""
    crit = virality_result([0.9] * 6, 6, "low")["criticality"]
    assert crit["regime"] == "explosive"
    assert crit["capped_frac"] > 0.5


# ── batch results survive a malformed line ───────────────────────────────


def test_one_malformed_batch_line_does_not_discard_the_whole_batch():
    """`json.loads` sat OUTSIDE parse_batch_line's guard, so a single truncated
    line raised JSONDecodeError, propagated to jobs._process's blanket handler,
    and marked the job `error` — throwing away every completed twin in a batch
    that had already run up to 24 hours and been billed in full.
    """
    from app.batch_swarm import parse_batch_line

    good = json.dumps(
        {
            "response": {
                "status_code": 200,
                "body": {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "engagement": 0.6,
                                        "intent_score": 0.5,
                                        "action": "continue",
                                        "dropoff_point": None,
                                        "friction_notes": "the price block is buried",
                                        "inner_monologue": "curious but not sold",
                                    }
                                )
                            }
                        }
                    ]
                },
            }
        }
    )
    for bad in ('{"response": {"status_code": 200,', "", "not json at all", "[]"):
        assert parse_batch_line(bad) is None  # must not raise
    assert parse_batch_line(good) is not None


# ── schema violations are 502s, not 500s or truncated streams ────────────


def test_out_of_schema_model_output_maps_to_a_gateway_error():
    """pydantic's ValidationError subclasses ValueError, not RuntimeError, so
    the `except (openai.OpenAIError, RuntimeError)` handlers missed it: a bare
    500 on the JSON routes, and on SSE a body that truncated after the headers
    with no terminal frame — indistinguishable from a successful empty run.

    Reachable by construction: `oai._clean` strips minimum/maximum from the
    wire schema, so the model may legally return confidence: 1.4.
    """
    from pydantic import BaseModel, Field, ValidationError

    from app.main import _llm_errors

    class Bounded(BaseModel):
        confidence: float = Field(ge=0.0, le=1.0)

    try:
        Bounded(confidence=1.4)
        raise AssertionError("fixture must raise")
    except ValidationError as exc:
        assert isinstance(exc, ValueError)
        assert not isinstance(exc, RuntimeError)  # this is why it was missed
        http = _llm_errors(exc)
        assert http.status_code == 502
        assert "schema" in str(http.detail).lower()
