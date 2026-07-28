"""Validation harness: compare swarm predictions against recorded real-world
outcomes ("actuals"). This is the mechanism that keeps the product honest —
calibration numbers, not vibes.

Actuals are recorded per run via POST /v1/runs/{id}/actuals (e.g. the real
CTR-derived intent once the client actually shipped a variant, or the real
completion rate for a flow). The report aggregates prediction error.

**MAE and bias alone cannot grade a predictor.** They are both first-moment
summaries of the error distribution, and between them they cannot separate the
two failure modes that matter most:

  · a well-calibrated but noisy predictor — errors large, symmetric, and
    uncorrelated with the truth; and
  · a systematically compressed one — every prediction shrunk toward the
    population mean, which is the characteristic failure of an LLM swarm asked
    for a 0-1 score, and which can post a *better* MAE than the honest
    predictor precisely because it never commits.

Both can show MAE 0.12, bias 0.00. What separates them is the shape of the
predicted-vs-actual relationship: correlation (does the predictor order the
world correctly at all?) and the calibration line predicted = a + b·actual
(b ≈ 1 means the predictor moves one-for-one with reality; b < 1 is
compression, b > 1 exaggeration; a is calibration-in-the-large). RMSE is
reported alongside MAE because the gap between them is itself diagnostic —
RMSE ≫ MAE means the error is concentrated in a few bad misses rather than
spread evenly, which is a different remediation.

`calibration_report` is the endpoint-facing function and returns the pydantic
CalibrationReport unchanged, so nothing downstream breaks. The full battery is
`calibration_detail`, which returns plain dicts: CalibrationReport and
MetricCalibration are fixed in schemas.py, which this module does not own, and
pydantic silently DROPS unknown keyword arguments — extra fields passed to the
model would vanish without an error. Wiring the detail into the API needs the
schema extended first.
"""

import math
from typing import Any, Optional

from .analytics import _ols, bootstrap_ci
from .schemas import CalibrationReport, MetricCalibration

# maps the metric name in actuals -> the prediction key in a swarm result
_METRICS = {"engagement": "mean_engagement", "intent": "mean_intent"}


def _metric_calibration(pairs: list[tuple[float, float]]) -> Optional[MetricCalibration]:
    """The pydantic-facing calibration record.

    Delegates to `metric_calibration_full` rather than recomputing MAE and bias,
    so `/v1/validation/report` and `calibration_detail()` can never disagree.
    MetricCalibration now carries the regression terms, so the endpoint reports
    the full battery instead of two numbers that cannot distinguish a
    compressed predictor from a noisy one.
    """
    full = metric_calibration_full(pairs)
    if full is None:
        return None
    return MetricCalibration(
        n=full["n"],
        mae=full["mae"],
        bias=full["bias"],
        rmse=full["rmse"],
        r=full["pearson_r"],
        slope=full["calibration_slope"],
        intercept=full["calibration_intercept"],
        slope_r2=full["calibration_r2"],
        mae_ci=full["mae_ci"],
        bias_ci=full["bias_ci"],
        verdict=full["interpretation"],
    )


def metric_calibration_full(pairs: list[tuple[float, float]]) -> Optional[dict]:
    """The canonical calibration battery for one metric, as a plain dict.

    `pairs` are (predicted, actual). Everything the pydantic MetricCalibration
    carries is reproduced verbatim so this is a strict superset, plus:

      rmse                  √mean(e²). Compare against mae: rmse/mae ≈ 1.25 for
                            normal errors, and a much larger ratio means the
                            error is dominated by a handful of large misses.
      pearson_r             correlation of predicted with actual — whether the
                            swarm ranks stimuli correctly, independent of any
                            scale or offset error. A predictor can have
                            excellent MAE and r ≈ 0 (it predicts the mean every
                            time); that predictor is useless for choosing
                            between variants, which is what this product does.
      calibration_slope     b in predicted = a + b·actual. 1.0 = the prediction
                            moves one-for-one with reality. b < 1 is regression
                            to the mean — the compressed predictor above.
      calibration_intercept a in the same fit: calibration-in-the-large.
      calibration_r2        fit quality of that line.
      mae_ci / bias_ci      percentile bootstrap intervals. A calibration
                            report built on a handful of shipped runs is mostly
                            sampling noise, and a bare MAE to four decimals from
                            n = 5 invites a precision nobody has.

    r is derived from the OLS r² (r = sign(slope)·√r²) rather than recomputed:
    for a simple linear regression the two are identical by construction, and
    one code path cannot disagree with itself. The regression is
    predicted ~ actual (actual on the x-axis), which is the orientation that
    makes slope < 1 mean "compressed predictions".
    """
    if not pairs:
        return None
    n = len(pairs)
    predicted = [p for p, _ in pairs]
    actual = [a for _, a in pairs]
    errors = [p - a for p, a in pairs]
    abs_errors = [abs(e) for e in errors]

    slope = intercept = r2 = pearson_r = None
    fit = _ols(actual, predicted)  # x = actual, y = predicted
    if fit is not None:
        slope, intercept, r2 = fit
        if r2 is not None:
            pearson_r = math.copysign(math.sqrt(max(0.0, min(1.0, r2))), slope)

    mae_ci = bootstrap_ci(abs_errors, seed=101)
    bias_ci = bootstrap_ci(errors, seed=103)

    return {
        "n": n,
        "mae": round(sum(abs_errors) / n, 4),
        "bias": round(sum(errors) / n, 4),
        "rmse": round(math.sqrt(sum(e * e for e in errors) / n), 4),
        "pearson_r": round(pearson_r, 4) if pearson_r is not None else None,
        "calibration_slope": round(slope, 4) if slope is not None else None,
        "calibration_intercept": round(intercept, 4) if intercept is not None else None,
        "calibration_r2": round(r2, 4) if r2 is not None else None,
        "mae_ci": list(mae_ci) if mae_ci else None,
        "bias_ci": list(bias_ci) if bias_ci else None,
        "interpretation": _calibration_verdict(slope, pearson_r, n),
    }


def _calibration_verdict(
    slope: Optional[float], r: Optional[float], n: int
) -> str:
    if slope is None or r is None:
        return (
            f"n={n}: too few runs, or actuals with no spread, to fit a "
            "calibration line — MAE and bias are all that is identifiable here"
        )
    if abs(r) < 0.3:
        return (
            f"predictions barely track outcomes (r={r:.2f}) — the swarm is not "
            "ordering variants correctly, and a low MAE here only means it is "
            "predicting close to the population mean every time"
        )
    if slope < 0.7:
        return (
            f"compressed (slope={slope:.2f}): predictions move only {slope:.0%} as "
            "far as reality does — directionally right, systematically understated"
        )
    if slope > 1.3:
        return (
            f"exaggerated (slope={slope:.2f}): predictions swing further than "
            "outcomes do — extremes will over-promise"
        )
    return f"well-calibrated (slope={slope:.2f}, r={r:.2f}) over {n} runs"


def _collect_pairs(runs: list[dict[str, Any]]) -> tuple[dict[str, list[tuple[float, float]]], int]:
    pairs: dict[str, list[tuple[float, float]]] = {m: [] for m in _METRICS}
    runs_with_actuals = 0

    for run in runs:
        actuals = run.get("actuals")
        result = run.get("result") or {}
        if not actuals:
            continue
        counted = False
        for metric, prediction_key in _METRICS.items():
            actual = actuals.get(metric)
            predicted = result.get(prediction_key)
            if actual is None or predicted is None:
                continue
            pairs[metric].append((float(predicted), float(actual)))
            counted = True
        if counted:
            runs_with_actuals += 1
    return pairs, runs_with_actuals


def calibration_report(runs: list[dict[str, Any]]) -> CalibrationReport:
    """`runs` are hydrated swarm_runs rows (kind='swarm') with optional actuals."""
    pairs, runs_with_actuals = _collect_pairs(runs)
    return CalibrationReport(
        runs_with_actuals=runs_with_actuals,
        engagement=_metric_calibration(pairs["engagement"]),
        intent=_metric_calibration(pairs["intent"]),
    )


def calibration_detail(runs: list[dict[str, Any]]) -> dict:
    """The full calibration battery as a plain dict — same inputs, same pairing
    rules, superset of `calibration_report`'s numbers.

    Kept separate rather than folded into CalibrationReport because that model
    lives in schemas.py; see the module docstring.
    """
    pairs, runs_with_actuals = _collect_pairs(runs)
    return {
        "runs_with_actuals": runs_with_actuals,
        "engagement": metric_calibration_full(pairs["engagement"]),
        "intent": metric_calibration_full(pairs["intent"]),
        "note": (
            "MAE and bias are first-moment summaries and cannot separate a noisy "
            "but well-calibrated predictor from a systematically compressed one; "
            "read calibration_slope and pearson_r for that."
        ),
    }
