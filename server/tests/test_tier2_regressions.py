"""Regression guards for the Tier-2 statistical defects.

Same contract as test_math_regressions.py: every test here was written against
a *verified* counterexample, and the number quoted in each docstring is what
the code actually returned before the fix — measured, not hypothesised. They
exist because each of these failures was silent: the output was well-formed,
plausible, and wrong.
"""

from __future__ import annotations

import asyncio
import itertools
import math
import random
from collections import Counter

import pytest

from app import swarm
from app.cognition import (
    EVENT_SYMBOLS,
    _forward_scaled,
    _hmm_fit_single,
    habituation_fit,
    hmm_fit,
    hmm_select,
    sample_entropy,
    spectral_profile,
)
from app.neuro import change_points, intersubject_correlation
from app.persona import confidence_band, seed_persona
from app.schemas import BehavioralSignal, ScenarioVariant, TwinReaction
from app.validation import calibration_detail, metric_calibration_full


# ── fixtures ─────────────────────────────────────────────────────────────


def _planted_three_state(n: int, seed: int = 9) -> list[int]:
    """Three sticky latent states over a THREE-symbol alphabet: each state emits
    its own symbol 90% of the time. Ground truth K = 3."""
    rng = random.Random(seed)
    emit = {0: [0.90, 0.05, 0.05], 1: [0.05, 0.90, 0.05], 2: [0.05, 0.05, 0.90]}
    state, out = 0, []
    for _ in range(n):
        if rng.random() < 0.08:
            state = (state + rng.choice([1, 2])) % 3
        w = rng.random()
        p = emit[state]
        out.append(0 if w < p[0] else (1 if w < p[0] + p[1] else 2))
    return out


def _persona(mindset: str = "p"):
    return seed_persona(
        BehavioralSignal(
            deliberation="medium",
            frustration_signal="none",
            exploration_style="scanner",
            price_sensitivity_signal="unknown",
            friction_hotspot=None,
            evidence="t",
            likely_mindset=mindset,
            confidence=0.5,
        )
    )


def _reaction(action: str, intent: float) -> TwinReaction:
    return TwinReaction(
        engagement=0.5,
        intent_score=intent,
        action=action,
        dropoff_point=None,
        friction_notes="n/a",
        inner_monologue="…",
    )


def _collect(gen):
    async def run():
        return [e async for e in gen]

    return asyncio.run(run())


# ── HMM: parameter counting and identifiability ──────────────────────────


def test_hmm_charges_bic_for_the_observed_alphabet_only():
    """The emission block was billed K·(7−1) whatever the session emitted.

    Unobserved symbols receive only the _EPS pseudo-count in the M-step — they
    are pinned at ~0, not estimated — so charging BIC for them charges for
    parameters the fit never had. On a planted THREE-state, three-symbol
    sequence at T=170 the inflated count picked the wrong model:

    Before: n_params {K=2: 15, K=3: 26, K=4: 39}
            bic_by_k {'2': 331.4109, '3': 339.6313, '4': 401.7517} → K = 2
    After:  n_params {K=2:  7, K=3: 14, K=4: 23}
            bic_by_k {'2': 290.3246, '3': 278.0017, '4': 319.5082} → K = 3
    """
    seq = _planted_three_state(170)
    sel = hmm_select(seq, len(EVENT_SYMBOLS), k_candidates=(2, 3, 4))
    assert sel is not None
    assert sel["n_symbols_observed"] == 3
    assert sel["n_params_by_k"] == {"2": 7, "3": 14, "4": 23}
    assert sel["n_states"] == 3  # the planted truth, unreachable before
    assert sel["bic_by_k"]["3"] < sel["bic_by_k"]["2"]


def test_hmm_refuses_more_parameters_than_the_data_can_identify():
    """`T < 2*K` is a length check, not an identifiability check.

    Before: hmm_fit over an 8-event sequence with K=4 and a 7-symbol alphabet
    returned a full fit — 39 free parameters estimated from 8 observations
    (0.205 observations per parameter) — complete with a confident Viterbi
    state decoding. It now returns None.
    """
    eight_events = [0, 1, 2, 3, 4, 5, 6, 0]
    assert len(eight_events) == 8
    assert hmm_fit(eight_events, 4, len(EVENT_SYMBOLS)) is None
    # the guard is a ratio, so the same K becomes fittable with enough data
    assert hmm_fit(_planted_three_state(400), 4, len(EVENT_SYMBOLS)) is not None


def test_hmm_reports_the_likelihood_of_the_parameters_it_returns():
    """`ll_trace.append(ll)` ran BEFORE the M-step.

    On the max_iter-exhaustion path the returned A/B/pi have had one more
    M-step applied than ll_trace[-1] describes, so BIC was computed from a
    likelihood the returned model does not have.

    Before, on a 200-event three-state sequence at max_iter=20: reported
    log_likelihood −194.1735 while the returned parameters actually score
    −185.1258 — a 9.05 nat discrepancy, always in the anti-conservative
    direction because EM only ever improves.
    """
    seq = _planted_three_state(200)
    fit = _hmm_fit_single(seq, 3, len(EVENT_SYMBOLS), max_iter=20, tol=0.0)
    assert fit is not None and fit["converged"] is False

    recomputed, _, _ = _forward_scaled(
        seq, fit["transition"], fit["emission"], fit["initial"]
    )
    # the residual is 4-decimal rounding of the returned matrices, ~0.013
    assert abs(recomputed - fit["log_likelihood"]) < 0.05
    # the value the old code would have reported is the stale one, and it is far
    assert abs(recomputed - fit["ll_trace"][-2]) > 1.0
    # EM is monotone, so the refreshed entry must not break the trace
    assert all(b >= a - 1e-6 for a, b in zip(fit["ll_trace"], fit["ll_trace"][1:]))
    assert fit["bic"] == pytest.approx(
        -2.0 * fit["log_likelihood"] + fit["n_params"] * math.log(len(seq)), abs=1e-3
    )


# ── spectral: bandwidth and record length ────────────────────────────────


def test_spectral_bands_are_bandwidth_normalised():
    """The three bands got 5, 14 and 41 bins on the old fixed 0.05-Hz grid, so
    summing raw power inside each measured band *width* as much as band energy.

    Before, white noise — flat by definition, no rhythm at any scale — averaged
    drift 0.086 / rhythm 0.231 / burst 0.683 over 40 realisations and read out
    as "saccadic burst dominant". After: 0.357 / 0.308 / 0.336.
    """
    totals = [0.0, 0.0, 0.0]
    reps = 40
    for s in range(reps):
        rng = random.Random(500 + s)
        ts = [i * 100.0 for i in range(400)]  # 40 s at 10 Hz
        sp = spectral_profile(ts, [rng.gauss(0, 1) for _ in range(400)])
        assert sp is not None
        for i, name in enumerate(("drift", "rhythm", "burst")):
            totals[i] += sp[f"{name}_band"] / reps

    for share in totals:
        assert share == pytest.approx(1 / 3, abs=0.06)
    # and no band runs away with it the way `burst` used to
    assert max(totals) - min(totals) < 0.12


def test_spectral_bands_still_sum_to_one():
    """Normalising densities rather than raw power keeps the share contract."""
    rng = random.Random(2)
    ts = sorted(rng.uniform(0, 20000) for _ in range(80))
    vs = [math.sin(2 * math.pi * 0.4 * (t / 1000)) + 0.2 * rng.random() for t in ts]
    sp = spectral_profile(ts, vs)
    assert sp is not None
    total = sp["drift_band"] + sp["rhythm_band"] + sp["burst_band"]
    assert total == pytest.approx(1.0, abs=0.01)
    assert sp["peak_frequency_hz"] == pytest.approx(0.4, abs=0.1)


def test_spectral_grid_cannot_outrun_the_record():
    """A fixed 0.05-3.0 Hz grid was emitted regardless of the data.

    Before, a 1.15-second record reported drift_band 0.0937 — a fitted
    amplitude for the 0.05-0.25 Hz range, i.e. periods of 4 to 20 seconds,
    4× to 20× longer than the entire recording. Nothing there completed a
    single cycle. The band is now unresolvable and says so with None rather
    than reporting 0.0, which would read as "no drift energy".
    """
    rng = random.Random(7)
    ts = [i * 50.0 for i in range(24)]  # 1.15 s at 20 Hz
    sp = spectral_profile(ts, [rng.gauss(0, 1) for _ in range(24)])
    assert sp is not None
    assert sp["span_s"] == pytest.approx(1.15, abs=1e-6)
    f_lo, f_hi = sp["frequency_range_hz"]
    assert f_lo == pytest.approx(1.0 / 1.15, abs=1e-3)  # one cycle in the record
    assert f_hi <= 3.0 and f_hi <= 1.0 / (2 * sp["median_sample_interval_s"]) + 1e-9
    assert sp["band_bin_counts"]["drift"] == 0
    assert sp["drift_band"] is None  # not 0.0 — unmeasurable, not absent
    assert sp["rhythm_band"] is not None and sp["burst_band"] is not None


# ── sample entropy: matched template ranges ──────────────────────────────


def test_sample_entropy_of_a_perfectly_periodic_series_is_zero():
    """SampEn = −ln(A/B) with A and B counted over DIFFERENT template sets.

    `range(n - mm)` gave the length-m pass N−m origins and the length-(m+1)
    pass N−m−1, so the m+1 count was systematically one template short and A/B
    was not a conditional probability. A perfectly periodic series has
    SampEn ≡ 0 by construction — every m-match extends to an m+1-match — yet:

    Before: (i % 6)/5 over 30 points → 0.08
            period-8 triangle wave over 24 points → 0.1054
    Pure irregularity manufactured by an off-by-one, on the most regular
    inputs that exist.
    """
    period_six = [(i % 6) / 5.0 for i in range(30)]
    triangle = [abs((i % 8) - 4) / 4.0 for i in range(24)]
    assert sample_entropy(period_six) == 0.0
    assert sample_entropy(triangle) == 0.0
    # a 120-point sine also loses its inflation (0.2905 → 0.2576)
    assert sample_entropy([math.sin(i / 3.0) for i in range(120)]) < 0.28


def test_sample_entropy_still_orders_regular_below_irregular():
    """The bookkeeping fix must not blunt the measurement it corrects."""
    rng = random.Random(9)
    regular = sample_entropy([math.sin(i / 3.0) for i in range(120)])
    irregular = sample_entropy([rng.random() for _ in range(120)])
    assert regular is not None and irregular is not None
    assert irregular > regular


# ── habituation: censored tail and retransformation bias ─────────────────


def _quantised_decay(n: int, lam: float = 0.15, a: float = 3.0):
    ts = [float(i) for i in range(n)]
    return ts, [round(a * math.exp(-lam * t), 1) for t in ts]


# This fixture quantises by ROUNDING, so a reported 0 means the true value was
# below 0.05 — half the smallest positive value it can report. The production
# collector instead THRESHOLDS (velocity is |Δy|/Δt with Δy in whole pixels, so
# a 0 means nothing moved at all), where a 0 means the value was below the
# smallest positive reading, not below half of it. Those are different censoring
# points, and `habituation_fit` truncates at the smallest positive value by
# default because that is what the real collector produces. Passing the limit
# explicitly here keeps these tests measuring what they were written to measure
# — the censored-tail and smearing corrections — rather than silently testing a
# censoring model the fixture does not use.
_FIXTURE_LOD = 0.05


def test_habituation_does_not_discard_the_decayed_tail():
    """`if y > 0` deleted every sample where the person had stopped moving.

    That is precisely the decayed tail the model exists to measure, and the
    deletion is not random with respect to the fit — near the resolution limit
    only positive-noise realisations survive, so the retained points sit above
    the true line at the right end and the slope flattens.

    Before, on y = 3·exp(−0.15t) quantised to 0.1, the fit returned the
    IDENTICAL decay_rate_per_s 0.1406 and initial_engagement 2.7418 for records
    of 32, 41 and 50 samples — 18 extra seconds of recorded disengagement
    changed nothing, because everything past the quantum was thrown away.
    True λ = 0.15.
    """
    fits = [
        habituation_fit(*_quantised_decay(n), censoring_limit=_FIXTURE_LOD)
        for n in (32, 41, 50)
    ]
    assert all(f is not None for f in fits)

    # every sample is now in the fit, and the censored share is reported
    assert [f["n"] for f in fits] == [32, 41, 50]
    assert fits[0]["censored_fraction"] == pytest.approx(0.125, abs=1e-3)
    assert fits[2]["censored_fraction"] == pytest.approx(0.44, abs=1e-3)

    # and the estimate is closer to the truth than the truncated fit was
    for f in fits:
        assert f["decay_rate_per_s"] == pytest.approx(0.143, abs=0.005)
        assert abs(f["decay_rate_per_s"] - 0.15) < abs(0.1406 - 0.15)


def test_habituation_applies_the_smearing_correction():
    """exp(intercept) estimates the median of y, not its mean.

    With ln y = ln A − λt + ε, E[y] = A·exp(−λt)·exp(σ²/2), so retransforming
    the intercept alone is biased low by exactly the factor the scatter
    contributes. Before: initial_engagement 2.7418 against a true A = 3.0.
    """
    ts, ys = _quantised_decay(41)
    f = habituation_fit(ts, ys, censoring_limit=_FIXTURE_LOD)
    assert f is not None
    assert f["initial_engagement"] > f["initial_engagement_median"]
    assert f["initial_engagement"] == pytest.approx(2.84, abs=0.03)
    assert abs(f["initial_engagement"] - 3.0) < abs(2.7418 - 3.0)


def test_habituation_leaves_a_clean_exponential_untouched():
    """No censoring, no scatter ⇒ the correction must be exactly inert."""
    ts = [i * 2.0 for i in range(30)]
    f = habituation_fit(ts, [3.0 * math.exp(-0.05 * t) for t in ts])
    assert f is not None
    assert f["decay_rate_per_s"] == pytest.approx(0.05, abs=1e-4)
    assert f["initial_engagement"] == pytest.approx(3.0, abs=1e-3)
    assert f["censored_fraction"] == 0.0


# ── ISC: Fisher z, permutation null, bootstrap ───────────────────────────


def test_isc_averages_correlations_in_fisher_z_space():
    """r is not additive — its sampling distribution is skewed near ±1, which
    is exactly where a gripping piece of content puts it.

    Five twins locked together at r ≈ 0.95 with three dissenters at r ≈ 0.10
    average to 0.6312 raw but 0.8282 in Fisher-z. The arithmetic mean is
    dragged toward the dissenters by a metric that does not respect the
    geometry of a correlation; the Hasson ISC literature the module cites
    averages in z.
    """
    pair_rs = [0.95] * 5 + [0.10] * 3
    raw = sum(pair_rs) / len(pair_rs)
    z_mean = sum(0.5 * math.log((1 + r) / (1 - r)) for r in pair_rs) / len(pair_rs)
    assert raw == pytest.approx(0.6312, abs=1e-4)
    assert math.tanh(z_mean) == pytest.approx(0.8282, abs=1e-4)

    # and the module reports the z-mean, exposed for auditing
    base = [0.2, 0.8, 0.5, 0.9, 0.3]
    isc = intersubject_correlation([base] * 5)
    assert isc is not None
    assert isc["overall"] == pytest.approx(math.tanh(isc["mean_fisher_z"]), abs=1e-6)


def test_isc_reports_a_permutation_null_and_a_bootstrap_interval():
    """`overall` was reported bare, with nothing distinguishing real synchrony
    from the arithmetic of eight twins over four beats.

    Before: {'overall': 1.0, 'n_pairs': 10, 'grip': 'locked-in'} — no p, no
    interval, no null. The null is now beat-order shuffles applied
    independently within each curve, which preserves every twin's marginal
    appraisal distribution and destroys only the shared temporal structure ISC
    claims to measure.
    """
    base = [0.2, 0.8, 0.5, 0.9, 0.3]
    isc = intersubject_correlation([base] * 5)
    assert isc is not None
    assert isc["overall"] == pytest.approx(1.0, abs=1e-6)
    assert isc["p_value"] <= 0.01  # add-one estimator over 500 permutations
    assert isc["significant"] is True
    assert isc["grip"] == "locked-in"
    lo, hi = isc["ci"]
    assert lo <= isc["overall"] <= hi + 1e-9
    # the null itself must sit near zero: that is what makes it a null
    assert abs(isc["null_mean_r"]) < 0.3


def test_isc_grip_is_gated_on_significance_not_a_bare_threshold():
    """0.5 / 0.25 cuts on a point estimate convert sampling noise into a
    verdict. Eight uncorrelated twins are one lucky draw from "engaged".

    Before: grip came straight off `overall > 0.5 / > 0.25`. Now an audience
    whose synchrony does not beat its own permutation null is 'drifting'
    regardless of where the point estimate landed.
    """
    rng = random.Random(4)
    curves = [[rng.random() for _ in range(12)] for _ in range(8)]
    isc = intersubject_correlation(curves)
    assert isc is not None
    assert isc["p_value"] > 0.05
    assert isc["significant"] is False
    assert isc["grip"] == "drifting"

    # a genuinely synchronised audience still clears the bar
    shape = [0.2, 0.9, 0.3, 0.85, 0.25, 0.8]
    rng2 = random.Random(11)
    synced = [[v + rng2.gauss(0, 0.05) for v in shape] for _ in range(8)]
    hot = intersubject_correlation(synced)
    assert hot is not None
    assert hot["significant"] is True and hot["grip"] == "locked-in"


# ── change points: effective sample size and degenerate splits ───────────


def test_change_points_rejects_a_microscopic_shift():
    """`best_cost <= 1e-12` short-circuited the test and AUTO-ACCEPTED.

    A split that drives the residual to zero was treated as self-evidently
    real; it is the opposite — a perfect fit is what a degenerate split looks
    like. Verified before the fix:

        change_points([0.5, 0.5, 0.500001, 0.500001], min_size=1) == [2]

    i.e. a declared attention collapse on a shift of one part in 500,000.
    """
    assert change_points([0.5, 0.5, 0.500001, 0.500001], min_size=1) == []
    # The same shape at a magnitude anyone would act on is still detected — but
    # only on the twin-level null, which is the one the product uses. On four
    # beat MEANS alone the permutation p cannot go below 2/C(4,2) = 0.33, so a
    # 4-point curve can never clear 0.05 however large the step; the ratings the
    # means came from are what make the test able to fire at all.
    rng = random.Random(9)
    curve = [0.5, 0.5, 0.9, 0.9]
    ratings = [
        [min(1.0, max(0.0, rng.gauss(mu, 0.05))) for _ in range(20)] for mu in curve
    ]
    assert change_points(curve, min_size=1, observations=ratings) == [2]
    # and the microscopic version stays rejected even with the sharper null
    tiny = [0.5, 0.5, 0.500001, 0.500001]
    tiny_ratings = [
        [min(1.0, max(0.0, rng.gauss(mu, 0.05))) for _ in range(20)] for mu in tiny
    ]
    assert change_points(tiny, min_size=1, observations=tiny_ratings) == []


def test_change_points_holds_its_level_across_twin_counts():
    """The false-positive rate must be a NOMINAL LEVEL, not a function of how
    many twins ran.

    Two generations of this test. The original criterion ran a BIC on a mean
    curve while charging the penalty as if each beat were one observation: since
    m·ln(RSS0/RSS1) is scale-free, averaging shrank the residual noise by
    √n_twins while leaving the statistic untouched, so every wobble cleared a
    penalty that never moved — 36.5% false positives at 8 beats × 20 twins.
    Charging the penalty at the effective sample size m·n_twins pulled that to
    7.0% *at that one cell*, which is what this test used to pin, and the cell
    was mistaken for a level: the same criterion measured 19.8% / 9.5% / 5.4% /
    3.4% at 3 / 8 / 20 / 40 twins and 63.3% on a 3-beat curve. A fixed penalty
    cannot hold a level across those regimes, so there is no `n_obs` to tune.

    The test is now a permutation test calibrated against the study's own
    ratings, and `n_obs` sets no threshold at all. The property worth pinning is
    therefore the one the old design could not have: the level is FLAT in the
    twin count, in both directions. Measured at 600 trials/cell it sits in
    3.8-7.0% for every combination of 3/8/20 twins and 3/4/8/10 beats.
    """
    trials = 150
    rates = {}
    for twins in (3, 8, 20):
        fired = 0
        for s in range(trials):
            rng = random.Random(1000 + s)
            per_beat = [
                [min(1.0, max(0.0, rng.gauss(0.6, 0.15))) for _ in range(twins)]
                for _ in range(8)
            ]
            mean_curve = [sum(col) / len(col) for col in per_beat]
            if change_points(
                mean_curve, min_size=1, n_obs=twins, observations=per_beat
            ):
                fired += 1
        rates[twins] = fired / trials

    for twins, rate in rates.items():
        assert rate < 0.13, (twins, rate)  # a level, at every twin count
    # and it does not GROW with the twin count, which was the original defect
    assert rates[20] < 3 * rates[3] + 0.05

    # `n_obs` is inert now: it is accepted for signature compatibility, and
    # passing a wrong value must not change a verdict.
    rng = random.Random(4)
    curve = [0.9] * 5 + [0.4] * 5
    ratings = [
        [min(1.0, max(0.0, rng.gauss(mu, 0.05))) for _ in range(12)] for mu in curve
    ]
    assert change_points(curve, min_size=1, n_obs=1, observations=ratings) == (
        change_points(curve, min_size=1, n_obs=999, observations=ratings)
    )


def test_change_points_still_finds_a_planted_collapse():
    """The correction must not blind the detector to a real break."""
    assert change_points([0.9] * 10 + [0.2] * 10) == [10]
    assert change_points([0.9] * 10 + [0.2] * 10, n_obs=25) == [10]
    assert change_points([0.5] * 20) == []


# ── A/B compare: winner reconciliation and control substitution ──────────


def test_compare_names_both_winners_and_flags_the_disagreement():
    """`winner` is argmax mean_intent; `bayesian.winner` is argmax P(best) on
    convert COUNTS. Different estimands over different data.

    Before, both were emitted side by side with nothing saying they answer
    different questions and nothing flagging when they point at different arms
    — so whichever the reader's eye landed on became "the winner". Here arm A
    wins on warmth (mean intent 0.9 vs 0.5) while arm B wins on conversion
    (8 of 8 vs 0 of 8): the classic liked-it-more / bought-it-less split.
    """
    per_variant = {
        "A": [_reaction("continue", 0.9) for _ in range(8)],
        "B": [_reaction("convert", 0.5) for _ in range(8)],
    }
    res = swarm._compare_result(
        per_variant, {"A": "copy a", "B": "copy b"}, "low", 8
    )
    b = res.bayesian
    assert res.winner == "A"                      # unchanged, schema-pinned
    assert b["winner_by_mean_intent"] == "A"
    assert b["winner_by_conversion"] == "B"
    assert b["winners_disagree"] is True
    assert "different metrics" in b["winner_disagreement_note"]

    # when they agree, the flag is off and no note is manufactured
    agree = swarm._compare_result(
        {
            "A": [_reaction("abandon", 0.1) for _ in range(8)],
            "B": [_reaction("convert", 0.9) for _ in range(8)],
        },
        {"A": "copy a", "B": "copy b"},
        "low",
        8,
    )
    assert agree.bayesian["winners_disagree"] is False
    assert agree.bayesian["winner_disagreement_note"] is None


def test_compare_surfaces_a_re_designated_control():
    """`names = list(by_name)` is built from arms that produced usable twins.

    If every twin in the first variant failed, the second variant silently
    became the baseline and every lift_vs_control_pct in the payload was
    measured against a stimulus nobody nominated — with nothing in the response
    saying so. Before: control_arm/control_substituted did not exist and the
    lifts were unlabelled.
    """
    per_variant = {
        "A": [],  # every twin failed
        "B": [_reaction("continue", 0.4) for _ in range(6)],
        "C": [_reaction("convert", 0.8) for _ in range(6)],
    }
    res = swarm._compare_result(
        per_variant, {"A": "a", "B": "b", "C": "c"}, "low", 6
    )
    b = res.bayesian
    assert b["control_arm"] == "B"
    assert b["control_substituted"] is True
    assert "A" in b["control_substitution_note"]
    # the lift really is measured against B, which is why it must be labelled
    lifts = {r.name: r.lift_vs_control_pct for r in res.ranking}
    assert lifts["B"] is None and lifts["C"] == pytest.approx(100.0)

    # the ordinary case stays quiet
    ok = swarm._compare_result(
        {"A": [_reaction("continue", 0.4)] * 4, "B": [_reaction("continue", 0.5)] * 4},
        {"A": "a", "B": "b"},
        "low",
        4,
    )
    assert ok.bayesian["control_substituted"] is False
    assert ok.bayesian["control_substitution_note"] is None


# ── adaptive allocation: persona/variant confound ────────────────────────


def test_per_arm_persona_cycles_decouple_the_arms():
    """One shared `itertools.cycle(personas)` made an arm's persona mix depend
    on how many agents every OTHER arm happened to consume first.

    run_compare promises "every variant faces the same persona set … so the
    only difference between arms is the stimulus", and adaptive allocation is
    unequal by design, so the promise failed exactly where it mattered. With 3
    personas and a shared cycle, an arm given 4 agents after another arm took
    7 sees personas [p1, p2, p0, p1]; the same arm with the other arm idle sees
    [p0, p1, p2, p0]. Different composition, same stimulus — the measured lift
    is part stimulus and part persona mix, unrecoverably.
    """
    personas = [_persona(f"m{i}") for i in range(3)]
    names = ["A", "B"]

    # per-arm cycles: B's draw sequence is identical regardless of A's budget
    busy = swarm._per_arm_persona_cycles(names, personas)
    [next(busy["A"]) for _ in range(7)]
    b_after_busy = [next(busy["B"]).persona_id for _ in range(4)]

    idle = swarm._per_arm_persona_cycles(names, personas)
    b_after_idle = [next(idle["B"]).persona_id for _ in range(4)]
    assert b_after_busy == b_after_idle

    # the old shared cycle: the same two scenarios give B different personas
    shared_busy = itertools.cycle(personas)
    [next(shared_busy) for _ in range(7)]
    old_busy = [next(shared_busy).persona_id for _ in range(4)]
    shared_idle = itertools.cycle(personas)
    old_idle = [next(shared_idle).persona_id for _ in range(4)]
    assert old_busy != old_idle


def test_adaptive_arms_receive_balanced_persona_mixes(monkeypatch):
    """End-to-end through stream_compare with Thompson skewing the budget.

    Three personas, two arms, adaptive allocation landing 9 agents on the weak
    arm and 23 on the strong one. Each arm's persona composition must be as
    balanced as its own budget allows — counts differing by at most one.

    Before (single global cycle), the weak arm's persona counts came out
    **[2, 4, 3]**: one persona ran twice as often as another inside the same
    arm, purely as a function of how many agents the *other* arm consumed
    first. After: **[3, 3, 3]**, exactly balanced, with the strong arm at
    [7, 8, 8].
    """
    seen: list[tuple[str, str]] = []
    dispatched = Counter()

    async def fake_twin(persona, scenario, load, semaphore):
        # a modest, arm-dependent conversion rate keeps the posterior from
        # becoming conclusive on the seed wave, so Thompson actually skews
        dispatched[scenario] += 1
        rate = 6 if "winner" in scenario else 4
        seen.append((scenario, persona.persona_id))
        return _reaction(
            "convert" if dispatched[scenario] % 10 < rate else "abandon", 0.5
        )

    monkeypatch.setattr(swarm, "_run_twin", fake_twin)
    personas = [_persona(f"m{i}") for i in range(3)]
    variants = [
        ScenarioVariant(name="A", scenario="loser copy"),
        ScenarioVariant(name="B", scenario="winner copy"),
    ]
    _collect(
        swarm.stream_compare(
            personas,
            variants,
            twins_per_persona=6,
            cognitive_load="low",
            adaptive=True,
            min_per_arm=2,
        )
    )

    per_arm: dict[str, Counter] = {}
    for scenario, pid in seen:
        per_arm.setdefault(scenario, Counter())[pid] += 1
    assert len(per_arm) == 2
    # allocation IS unequal — that is the whole point of adaptive mode, and the
    # exact condition under which the shared cycle used to confound the arms
    assert len({sum(c.values()) for c in per_arm.values()}) == 2
    for counts in per_arm.values():
        assert len(counts) == len(personas)  # no arm misses a persona
        assert max(counts.values()) - min(counts.values()) <= 1


# ── calibration report ───────────────────────────────────────────────────


_ACTUALS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
_SIGNS = [1, -1, -1, 1, 1, -1, -1, 1]
# predictions shrunk 85% toward the mean: directionally perfect, systematically
# understated
_COMPRESSED = [(0.45 + 0.15 * (a - 0.45), a) for a in _ACTUALS]
# unbiased but noisy: right on average, wrong on every individual run
_NOISY = [(a + 0.17 * s, a) for a, s in zip(_ACTUALS, _SIGNS)]


def test_mae_and_bias_alone_cannot_grade_a_predictor():
    """MAE and bias are both first-moment summaries of the error distribution.

    These two predictors are indistinguishable on the pair the report used to
    emit — MAE 0.17 and bias 0.0 for BOTH — yet one tracks reality perfectly
    and merely understates it (slope 0.15, r = 1.00) while the other is
    correctly scaled and simply noisy (slope 1.00, r = 0.80). The first would
    have been shipped as "well calibrated, 0.17 average error".
    """
    comp = metric_calibration_full(_COMPRESSED)
    noisy = metric_calibration_full(_NOISY)
    assert comp is not None and noisy is not None

    # the old report: identical
    assert comp["mae"] == noisy["mae"] == pytest.approx(0.17, abs=1e-4)
    assert comp["bias"] == pytest.approx(0.0, abs=1e-9)
    assert noisy["bias"] == pytest.approx(0.0, abs=1e-9)

    # the new report: opposite diagnoses
    assert comp["calibration_slope"] == pytest.approx(0.15, abs=1e-3)
    assert comp["pearson_r"] == pytest.approx(1.0, abs=1e-3)
    assert noisy["pearson_r"] == pytest.approx(0.80, abs=0.02)
    assert "compressed" in comp["interpretation"]
    assert "well-calibrated" in noisy["interpretation"]

    # The NOISY fixture's actuals are exact — it adds error to the prediction
    # only — so the reported Deming slope (δ = 1, which assumes both axes are
    # equally noisy) over-corrects past 1 rather than landing on it: 1.3124, not
    # the 1.0 an OLS fit would give here. That is a property of the fixture, not
    # a defect: no fixed δ is right in both noise regimes, which is exactly why
    # the VERDICT is gated on the identification bracket instead. The bracket
    # must contain the structural slope of 1.0, and it is what makes the verdict
    # above come out "well-calibrated" despite the point estimate sitting high.
    assert noisy["calibration_slope"] == pytest.approx(1.3124, abs=1e-3)
    lo, hi = noisy["calibration_slope_bracket_ci"]
    assert lo <= 1.0 <= hi
    assert "exaggerated" not in noisy["interpretation"]
    # rmse separates them too: concentrated misses vs evenly spread error
    assert comp["rmse"] > comp["mae"]
    assert noisy["rmse"] == pytest.approx(noisy["mae"], abs=1e-4)


def test_calibration_reports_bootstrap_intervals_on_mae_and_bias():
    """A four-decimal MAE from a handful of shipped runs invites a precision
    nobody has. Before, neither number carried an interval at all."""
    comp = metric_calibration_full(_COMPRESSED)
    assert comp is not None
    lo, hi = comp["mae_ci"]
    assert lo <= comp["mae"] <= hi
    blo, bhi = comp["bias_ci"]
    assert blo <= comp["bias"] <= bhi
    assert blo < 0 < bhi  # zero bias is well inside the interval, as it should be


def test_calibration_detail_is_a_superset_and_keeps_the_schema_intact():
    """CalibrationReport is fixed in schemas.py and pydantic silently DROPS
    unknown kwargs, so the extra statistics are returned as plain dicts by a
    separate function rather than vanishing into the model."""
    from app.validation import calibration_report

    runs = [
        {"result": {"mean_intent": p, "mean_engagement": p}, "actuals": {"intent": a, "engagement": a}}
        for p, a in _COMPRESSED
    ]
    legacy = calibration_report(runs)
    detail = calibration_detail(runs)

    assert legacy.runs_with_actuals == detail["runs_with_actuals"] == 8
    assert detail["intent"]["n"] == legacy.intent.n
    assert detail["intent"]["mae"] == legacy.intent.mae
    assert detail["intent"]["bias"] == legacy.intent.bias
    for key in ("rmse", "pearson_r", "calibration_slope", "calibration_intercept",
                "mae_ci", "bias_ci"):
        assert key in detail["intent"]
    assert calibration_detail([])["intent"] is None


# ── persona: trait confidence reaches the prompt ─────────────────────────


def _signal(**overrides) -> BehavioralSignal:
    base = dict(
        deliberation="high",
        frustration_signal="elevated",
        exploration_style="goal_directed",
        price_sensitivity_signal="high",
        friction_hotspot="pricing-table",
        evidence="12 micro-hesitations and 3 rage-click bursts over pricing zone.",
        likely_mindset="I want the price, and this page is annoying me.",
        confidence=0.8,
    )
    base.update(overrides)
    return BehavioralSignal(**base)


def test_trait_confidence_reaches_the_twin_prompt():
    """Trait.confidence was computed, capped against the upstream signal,
    carried through the schema, unit-tested — and read by nothing.

    Before, _render_system_prompt emitted a bare
    `openness=0.7, conscientiousness=0.75, neuroticism=0.725`, so a trait
    inferred from a 0.1-confidence signal (trait confidence 0.06 — a coin flip
    dressed as a measurement) steered the twin exactly as hard as one from a
    0.9-confidence signal. The identical prompt line for both is the bug.
    """
    strong = seed_persona(_signal(confidence=0.9)).system_prompt
    weak = seed_persona(_signal(confidence=0.1)).system_prompt

    assert "well evidenced" in strong
    assert "barely evidenced" in weak
    assert "well evidenced" not in weak
    # the prompts must actually differ on the strength of the evidence
    assert strong != weak
    # every trait's confidence is stated, including the ones we have no read on
    for name in ("openness", "conscientiousness", "extraversion", "agreeableness",
                 "neuroticism"):
        assert name in strong
    assert "confidence 0.06" in weak  # neuroticism at signal confidence 0.1


def test_confidence_bands_track_the_discount_constant():
    """Bands are fractions of the maximum achievable trait confidence, so they
    cannot silently decouple from _CONFIDENCE_DISCOUNT."""
    assert confidence_band(0.54) == "strong"      # signal 0.9, evidence 1.0
    assert confidence_band(0.24) == "moderate"
    assert confidence_band(0.06) == "weak"        # extraversion, always
    seed = seed_persona(_signal(confidence=0.9))
    assert confidence_band(seed.traits.extraversion.confidence) == "weak"
    assert confidence_band(seed.traits.conscientiousness.confidence) == "strong"
