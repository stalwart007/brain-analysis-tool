import random

import pytest
from fastapi.testclient import TestClient

from app import storage
from app.analytics import _ols
from app.main import app
from app.validation import (
    _deming,
    _pairs_bootstrap_cis,
    _pav,
    calibration_report,
    metric_calibration_full,
)

client = TestClient(app)


def _noisy_pairs(n: int, structural_slope: float, sigma: float, seed: int):
    """(predicted, actual) pairs from truth ~ U(0.2, 0.8) with independent
    gaussian noise of the SAME sd on both axes — the regime where OLS dilutes
    and Deming (delta=1) is consistent."""
    rng = random.Random(seed)
    pairs = []
    for _ in range(n):
        t = rng.uniform(0.2, 0.8)
        p = 0.5 + structural_slope * (t - 0.5) + rng.gauss(0, sigma)
        a = t + rng.gauss(0, sigma)
        pairs.append((min(1.0, max(0.0, p)), min(1.0, max(0.0, a))))
    return pairs


def _run(mean_intent: float, mean_engagement: float, actuals=None) -> dict:
    return {
        "result": {"mean_intent": mean_intent, "mean_engagement": mean_engagement},
        "actuals": actuals,
    }


def test_calibration_math():
    runs = [
        _run(0.8, 0.7, {"intent": 0.6, "engagement": 0.7}),  # intent err +0.2, eng 0
        _run(0.5, 0.4, {"intent": 0.6}),                     # intent err -0.1
        _run(0.9, 0.9),                                       # no actuals -> excluded
    ]
    report = calibration_report(runs)
    assert report.runs_with_actuals == 2
    assert report.intent is not None
    assert report.intent.n == 2
    assert abs(report.intent.mae - 0.15) < 1e-9
    assert abs(report.intent.bias - 0.05) < 1e-9  # slight over-prediction
    assert report.engagement is not None and report.engagement.n == 1


def test_calibration_empty():
    report = calibration_report([])
    assert report.runs_with_actuals == 0
    assert report.intent is None and report.engagement is None


def test_actuals_endpoint_roundtrip():
    run_id = storage.insert_swarm_run(
        {"scenario": "x"}, {"mean_intent": 0.7, "mean_engagement": 0.6}, kind="swarm"
    )

    res = client.post(f"/v1/runs/{run_id}/actuals", json={"intent": 0.5})
    assert res.status_code == 200

    report = client.get("/v1/validation/report").json()
    assert report["runs_with_actuals"] >= 1
    assert report["intent"]["n"] >= 1


def test_actuals_validation():
    run_id = storage.insert_swarm_run({}, {"mean_intent": 0.5}, kind="swarm")
    assert client.post(f"/v1/runs/{run_id}/actuals", json={}).status_code == 400
    assert (
        client.post(f"/v1/runs/{run_id}/actuals", json={"intent": 1.4}).status_code == 422
    )
    assert client.post("/v1/runs/nope/actuals", json={"intent": 0.5}).status_code == 404


# ---------------------------------------------------------- regression dilution


def test_deming_resists_dilution_where_ols_attenuates():
    """Noise on the 'actual' axis drags an OLS slope below 1 (regression
    dilution); the Deming fit with delta=1 recovers the structural slope.
    This is the audited failure: an honest predictor read as 'compressed'."""
    pairs = _noisy_pairs(40, structural_slope=1.0, sigma=0.1, seed=11)
    xs = [a for _, a in pairs]
    ys = [p for p, _ in pairs]
    ols_slope = _ols(xs, ys)[0]
    dem_slope = _deming(xs, ys)[0]
    assert ols_slope < 0.9  # visibly attenuated
    assert 0.85 <= dem_slope <= 1.2  # near the structural truth
    assert dem_slope > ols_slope

    full = metric_calibration_full(pairs)
    assert full["calibration_slope"] == round(dem_slope, 4)
    assert "compressed" not in full["interpretation"]


def test_verdict_gated_on_ci_not_point_estimate():
    """'compressed' is asserted only when the entire slope CI sits below 1."""
    compressed = metric_calibration_full(
        _noisy_pairs(30, structural_slope=0.4, sigma=0.05, seed=23)
    )
    assert compressed["interpretation"].startswith("compressed")
    assert compressed["calibration_slope_ci"][1] < 1.0

    honest = metric_calibration_full(
        _noisy_pairs(30, structural_slope=1.0, sigma=0.05, seed=29)
    )
    lo, hi = honest["calibration_slope_ci"]
    assert lo <= 1.0 <= hi
    assert honest["interpretation"].startswith("well-calibrated")


def test_too_few_runs_verdict_below_ci_threshold():
    """n=5 used to earn a confident adjective; now the CIs are withheld and
    the verdict says so, while the point estimates stay reported."""
    full = metric_calibration_full(
        _noisy_pairs(5, structural_slope=1.0, sigma=0.05, seed=31)
    )
    assert full["calibration_slope"] is not None
    assert full["calibration_slope_ci"] is None
    assert full["pearson_r_ci"] is None
    assert full["reliability_curve"] is None
    assert full["brier_decomposition"] is None
    assert "too few runs" in full["interpretation"]


def test_slope_ci_present_and_deterministic_at_n8():
    pairs = _noisy_pairs(8, structural_slope=1.0, sigma=0.05, seed=37)
    s_ci, r_ci = _pairs_bootstrap_cis(pairs)
    assert s_ci is not None and r_ci is not None
    assert s_ci == _pairs_bootstrap_cis(pairs)[0]  # seeded => identical
    full_a = metric_calibration_full(pairs)
    full_b = metric_calibration_full(pairs)
    assert full_a == full_b


# ---------------------------------------------------------- degenerate axes


def test_deming_degenerate_axes():
    # both axes constant: no line
    assert _deming([0.5] * 5, [0.5] * 5) is None
    # actuals constant, predictions vary: vertical line, unidentifiable
    assert _deming([0.5] * 5, [0.1, 0.3, 0.5, 0.7, 0.9]) is None
    # predictions constant, actuals vary: the predictor genuinely does not move
    slope, intercept = _deming([0.1, 0.3, 0.5, 0.7, 0.9], [0.4] * 5)
    assert slope == 0.0 and abs(intercept - 0.4) < 1e-12
    # n = 2: below any fit
    assert _deming([0.1, 0.9], [0.2, 0.8]) is None


def test_degenerate_pairs_report_without_crashing():
    # constant predictions: slope 0, r undefined, verdict falls to the
    # unidentifiable branch rather than praising a flat predictor
    flat = metric_calibration_full([(0.5, a / 10) for a in range(1, 9)])
    assert flat["calibration_slope"] == 0.0
    assert flat["pearson_r"] is None
    assert "MAE and bias are all that is identifiable" in flat["interpretation"]
    # constant actuals: slope unidentifiable
    const_a = metric_calibration_full([(p / 10, 0.5) for p in range(1, 9)])
    assert const_a["calibration_slope"] is None


# ---------------------------------------------------------- reliability / Brier


def test_pav_reliability_curve_known_answer():
    """Hand-checkable PAV: violators pool, fitted values are block means of
    actual, knot x is the block mean of predicted, curve is non-decreasing."""
    pairs = [
        (0.1, 0.2), (0.2, 0.1),  # violation -> pooled at 0.15
        (0.3, 0.3), (0.4, 0.2),  # violation -> pooled at 0.25
        (0.5, 0.5), (0.6, 0.4),  # violation -> pooled at 0.45
        (0.7, 0.7), (0.8, 0.8),
    ]
    full = metric_calibration_full(pairs)
    assert full["reliability_curve"] == [
        [0.15, 0.15], [0.35, 0.25], [0.55, 0.45], [0.7, 0.7], [0.8, 0.8],
    ]


def test_pav_pools_tied_predictions():
    """Two identical predictions must share one fitted value: the isotonic
    fit is a function of the predicted value."""
    pairs = [(0.4, 0.9), (0.4, 0.1)] + [(0.1 * i, 0.1 * i) for i in range(1, 4)] + [
        (0.5, 0.5), (0.6, 0.6), (0.7, 0.7),
    ]
    knots, fitted = _pav(pairs)
    xs = [k[0] for k in knots]
    ys = [k[1] for k in knots]
    assert xs == sorted(xs) and ys == sorted(ys)
    assert xs.count(0.4) <= 1  # the tie collapsed into a single knot


def test_brier_decomposition_identity_and_signs():
    """CORP: brier = miscalibration - discrimination + uncertainty, exactly
    (up to the 4-decimal rounding of each reported term), with the first two
    non-negative."""
    pairs = _noisy_pairs(20, structural_slope=0.6, sigma=0.08, seed=41)
    full = metric_calibration_full(pairs)
    d = full["brier_decomposition"]
    assert d["miscalibration"] >= 0.0
    assert d["discrimination"] >= 0.0
    assert d["uncertainty"] >= 0.0
    lhs = full["brier_score"]
    rhs = d["miscalibration"] - d["discrimination"] + d["uncertainty"]
    assert abs(lhs - rhs) < 5e-4


# ---------------------------------------------------------- contract stability


def test_full_battery_keeps_every_legacy_key():
    full = metric_calibration_full(_noisy_pairs(10, 1.0, 0.05, seed=43))
    legacy = {
        "n", "mae", "bias", "rmse", "pearson_r", "calibration_slope",
        "calibration_intercept", "calibration_r2", "mae_ci", "bias_ci",
        "interpretation",
    }
    added = {
        "calibration_slope_ci", "pearson_r_ci", "reliability_curve",
        "brier_score", "brier_decomposition",
    }
    assert legacy | added <= set(full.keys())
    # pydantic path still constructs cleanly from the superset
    report = calibration_report(
        [
            {
                "result": {"mean_intent": p, "mean_engagement": p},
                "actuals": {"intent": a, "engagement": a},
            }
            for p, a in _noisy_pairs(10, 1.0, 0.05, seed=47)
        ]
    )
    assert report.intent is not None and report.intent.n == 10
    assert report.intent.slope is not None


def _exact_actual_pairs(n: int, structural_slope: float, sigma: float, seed: int):
    """(predicted, actual) pairs where the ACTUAL carries no noise at all.

    The mirror image of `_noisy_pairs`, and the regime where Deming with δ = 1
    is the wrong estimator rather than the right one: with an exact x-axis, OLS
    is consistent and orthogonal regression over-corrects past the truth.
    """
    rng = random.Random(seed)
    pairs = []
    for _ in range(n):
        t = rng.uniform(0.2, 0.8)
        p = 0.5 + structural_slope * (t - 0.5) + rng.gauss(0, sigma)
        pairs.append((min(1.0, max(0.0, p)), t))
    return pairs


def test_the_verdict_does_not_depend_on_an_unknowable_error_ratio():
    """An honest predictor must not be called compressed OR exaggerated, in
    either noise regime.

    Switching OLS → Deming(δ=1) fixed the false-"compressed" verdict and
    introduced a false-"exaggerated" one, because δ = 1 assumes the actual is as
    noisy as the prediction and pushes the slope above 1 when it is cleaner than
    that. Measured on true-slope-1.0 data with exact actuals, gating on the
    Deming interval returned "exaggerated" in 16.0% / 26.5% / 44.5% of trials at
    n = 10 / 20 / 40 — rising with n, which is a structural bias rather than
    sampling noise. Gating on the identification bracket instead: 5.0 / 6.0 /
    3.5%, and flat in n.

    Both fixtures below are honest predictors, and this is a RATE test rather
    than an every-seed one on purpose: the gate's own claim is a few percent, not
    zero, so demanding a clean sweep of hand-picked seeds would be asserting
    something the estimator never promised and would fail on the expected tail.
    30 seeds per regime with a ceiling of 4 catches a regression to 26-44%
    without flagging the 3-6% that is supposed to be there.
    """
    SEEDS = 30
    for label, build in (
        ("exact actuals", _exact_actual_pairs),
        ("equal noise", _noisy_pairs),
    ):
        directions = 0
        brackets_covering = 0
        for seed in range(SEEDS):
            m = metric_calibration_full(
                build(40, structural_slope=1.0, sigma=0.1, seed=seed)
            )
            assert m is not None
            verdict = m["interpretation"]
            if "exaggerated" in verdict or "compressed" in verdict:
                directions += 1
            lo, hi = m["calibration_slope_bracket_ci"]
            if lo <= 1.0 <= hi:
                brackets_covering += 1
        assert directions <= 4, (label, directions)
        # The bracket is the mechanism, so it is asserted directly too.
        assert brackets_covering >= SEEDS - 4, (label, brackets_covering)


def test_the_bracket_brackets_ols_and_deming_rather_than_replacing_them():
    """The identified range must contain both point estimates, since each is the
    answer under one particular assumption about δ. A bracket that excluded
    either would not be an identification region at all."""
    pairs = _noisy_pairs(40, structural_slope=1.0, sigma=0.1, seed=11)
    m = metric_calibration_full(pairs)
    assert m is not None
    lo, hi = m["calibration_slope_bracket"]
    ols_slope = _ols([a for _, a in pairs], [p for p, _ in pairs])[0]
    assert lo == pytest.approx(ols_slope, abs=1e-4)  # OLS is the lower bound
    assert lo <= m["calibration_slope"] <= hi  # Deming sits inside


def test_a_genuinely_compressed_predictor_is_still_caught():
    """The bracket gate is conservative, not inert — it costs power (87% vs the
    slope-CI gate's 97% at n = 40) and must not cost all of it."""
    caught = 0
    for seed in range(20):
        m = metric_calibration_full(
            _noisy_pairs(40, structural_slope=0.4, sigma=0.05, seed=seed)
        )
        assert m is not None
        if "compressed" in m["interpretation"]:
            caught += 1
    assert caught >= 15
