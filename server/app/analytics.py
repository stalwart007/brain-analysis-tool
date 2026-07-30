"""Statistical engine for CogniSwarm.

Everything here is a PURE function over plain numbers — no LLM calls, no I/O —
so each method is unit-testable against hand-computed answers. Randomised
procedures (bootstrap, Monte-Carlo posterior sampling) take an explicit seed and
are therefore deterministic and reproducible.

Implemented from first principles on the stdlib (no numpy/scipy dependency):

  · Beta-Binomial Bayesian A/B — posterior P(win), credible intervals,
    expected loss (the decision-theoretic "cost of choosing this arm").
  · BCa (bias-corrected and accelerated) bootstrap confidence intervals for
    the mean of a sample.
  · Shannon entropy (normalised) and Sarle's bimodality coefficient — these
    detect a *split* swarm that a mean would hide.
  · Price elasticity of demand via log-log OLS, plus a linear-demand model
    whose revenue optimum is continuous (not restricted to tested points).
  · Kaplan-Meier discrete survival, per-step hazard ranked by its Wilson
    lower bound, and log-log (Kalbfleisch-Prentice) confidence bands — the
    correct treatment of funnel drop-off.
  · Average-linkage agglomerative clustering over cosine distance, with medoid
    representatives — used to merge semantically duplicate objections.
"""

from __future__ import annotations

import math
import random
from typing import Optional, Sequence

# ────────────────────────────────────────────────────────── descriptive


def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _central_moment(xs: Sequence[float], k: int) -> float:
    m = mean(xs)
    return sum((x - m) ** k for x in xs) / len(xs)


def stdev(xs: Sequence[float], sample: bool = True) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    m = mean(xs)
    denom = n - 1 if sample else n
    return math.sqrt(sum((x - m) ** 2 for x in xs) / denom)


def skewness(xs: Sequence[float]) -> float:
    """Population (biased, g1) skewness — the form Sarle's coefficient expects."""
    m2 = _central_moment(xs, 2)
    if m2 <= 1e-12:
        return 0.0
    return _central_moment(xs, 3) / (m2**1.5)


def excess_kurtosis(xs: Sequence[float]) -> float:
    """Population excess kurtosis (g2): 0 for a normal distribution."""
    m2 = _central_moment(xs, 2)
    if m2 <= 1e-12:
        return 0.0
    return _central_moment(xs, 4) / (m2**2) - 3.0


def shannon_entropy_normalised(counts: Sequence[int], k: Optional[int] = None) -> float:
    """0.0 = perfect consensus (all mass on one outcome), 1.0 = maximally split.

    Normalised by log(k) where k is the size of the *possible* outcome set, so
    the value is comparable across runs.

    `k` matters, and getting it wrong was a real bug here. Normalising by the
    number of *observed* outcomes makes a swarm split 50/50 between two actions
    score exactly 1.0 — identical to one spread evenly across all four — because
    each is maximally uncertain *within its own observed support*. That is the
    opposite of comparable: the first swarm agrees that two options are dead,
    the second finds no structure at all.

    So callers pass the full vocabulary size. `k` defaults to `len(counts)`,
    which is correct only when the caller has already included explicit zeros
    for unobserved outcomes.

    Zero-count outcomes contribute nothing to the sum (0·log0 ≡ 0).

    MILLER-MADOW. The plug-in estimator — substituting observed frequencies for
    the unknown probabilities — is biased DOWNWARD, and the bias is O(K/2N):
    severe at exactly the sample sizes a swarm produces. Measured against the
    truth of 1.0 for a maximally-split audience, plug-in reported 0.786 at n=6.
    That understatement is not a rounding error here, because
    `consensus_label` cuts at 0.35 / 0.45 / 0.70 — a 0.21 bias is comparable to
    the entire gap between adjacent verdicts, so small swarms were being
    reported as agreeing when they were doing nothing of the kind. Under-
    reporting disagreement is the dangerous direction for this product: it
    manufactures consensus out of thin data.

    The correction is (K̂ − 1) / 2N, with K̂ the number of outcomes actually
    observed. It is the first-order term of the bias expansion and costs one
    line; NSB would be better in the very-small-N regime and is not worth its
    complexity here. Clamped at 1.0, because a bias correction can legitimately
    push the estimate past the maximum the normaliser assumes, and reporting
    "1.07 of a possible 1.0" would read as a bug rather than as an estimate.
    """
    positive = [c for c in counts if c > 0]
    total = sum(positive)
    support = k if k is not None else len(counts)
    if total <= 0 or support < 2:
        return 0.0
    if len(positive) < 2:
        return 0.0  # all mass on one outcome ⇒ perfect consensus
    h = -sum((c / total) * math.log(c / total) for c in positive)
    h += (len(positive) - 1) / (2.0 * total)  # Miller-Madow
    return min(1.0, h / math.log(support))


def bimodality_coefficient(xs: Sequence[float]) -> Optional[float]:
    """Sarle's bimodality coefficient, in SAS's finite-sample form.

        BC = (G1² + 1) / (G2 + 3(n−1)² / ((n−2)(n−3)))

    BC > 5/9 ≈ 0.555 is the conventional threshold suggesting a bimodal (i.e.
    polarised) distribution; a uniform distribution sits at exactly 5/9 and a
    normal one near 0.263.

    THE CONVENTIONS HAVE TO MATCH, and they did not. The correction term
    3(n−1)²/((n−2)(n−3)) is Sarle's, and it is derived for the *bias-corrected*
    sample moments G1 and G2. This function was feeding it the *population*
    moments g1 and g2 from `skewness`/`excess_kurtosis`. Mixing the two
    conventions deflates BC badly at small n — which is the only n this ever
    sees.

    The cost was not academic. On a perfect two-camp split (half the twins at
    0.1, half at 0.9) — the exact polarised audience `consensus_label` exists to
    catch — the mixed form scored 0.492 at n = 12 and did not fire, where the
    correct form scores 0.603 and does. The canonical case was undetectable at
    typical swarm sizes.

    Below n = 12 the coefficient has no useful power in either convention, so it
    returns None rather than a number that reads as a measurement.
    """
    n = len(xs)
    # Sarle's correction is undefined at n ≤ 3 and uninformative well past it;
    # 12 is where a perfect 50/50 split first becomes detectable.
    if n < 12:
        return None
    if _central_moment(xs, 2) <= 1e-12:  # zero variance ⇒ perfectly unimodal
        return 0.0

    # population g1, g2 → bias-corrected G1, G2 (the moments Sarle assumes)
    g1 = skewness(xs)
    g2 = excess_kurtosis(xs)
    big_g1 = g1 * math.sqrt(n * (n - 1.0)) / (n - 2.0)
    big_g2 = ((n + 1.0) * g2 + 6.0) * (n - 1.0) / ((n - 2.0) * (n - 3.0))

    correction = 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    denom = big_g2 + correction
    if abs(denom) < 1e-12:
        return None
    return (big_g1**2 + 1.0) / denom


def consensus_label(entropy: float, bc: Optional[float]) -> str:
    """Human-readable read of how united the swarm is.

    Two independent measurements, deliberately: `bc` (Sarle) describes the
    *continuous intent* distribution, `entropy` the *categorical action* mix.
    Two camps show up in both, and requiring both guards against calling a
    merely noisy audience polarised.

    On thresholds: entropy is now normalised against the full four-action
    vocabulary, so a perfect two-way split — the canonical polarised
    audience — scores exactly log2/log4 = 0.5, an even three-way split 0.79,
    and a flat four-way spread 1.0. The old `> 0.5` gate was calibrated
    against a normalisation that scored *any* maximal split as 1.0, and under
    the corrected scale it would have excluded the exact case it was written
    to catch. 0.45 admits the two-camp case with a little tolerance for
    uneven camps.

    Note that a flat four-way spread is NOT polarisation — it is absence of
    signal — and lands in "split" below, which is the honest label for it.
    """
    if bc is not None and bc > 0.555 and entropy >= 0.45:
        return "polarised"
    if entropy < 0.35:
        return "strong consensus"
    if entropy < 0.7:
        return "leaning"
    return "split"


# ────────────────────────────────────────────────────────── bootstrap


#: Below this, a resampled interval is not an interval.
#:
#: The guard is the fix, not the estimator choice: at n = 2 this function used
#: to return the sample RANGE and call it 95% — measured coverage of the range
#: of two Exponential(1) draws is 46.5%. Named rather than inlined because the
#: UI has to say WHY an interval is missing, and "fewer than 8" is the whole
#: reason. `None` from this function has exactly one cause, which is what lets
#: the frontend explain it without being told the sample size.
MIN_BOOTSTRAP_N = 8


def bootstrap_ci(
    values: Sequence[float],
    iterations: int = 2000,
    alpha: float = 0.05,
    seed: int = 7,
) -> Optional[tuple[float, float]]:
    """BCa bootstrap CI for the mean. Deterministic for a given seed.

    Returns None for samples too small to resample meaningfully (n < 8).

    THE GUARD MOVED FROM 2 TO 8. At n = 2 the old percentile version returned
    the sample range and labelled it a 95% CI — measured:
    `bootstrap_ci([0.1, 0.9])` gave exactly `(0.1, 0.9)`, and the range of two
    Exponential(1) draws covers the true mean 46.5% of the time (200k trials),
    less than half the advertised confidence. The problem shrinks but does not
    vanish through n = 3…7: the endpoints are order statistics of a handful of
    distinct resample atoms. Below 8 the honest answer is no interval, which
    is what None means everywhere in this module.

    BCa RATHER THAN PERCENTILE. The percentile interval is first-order
    accurate and undercovers on skewed data at exactly the sample sizes swarms
    produce. Measured (95% target, mean of Exponential(1), n = 10, 2000 trials
    of 2000 resamples each, both endpoints from the SAME bootstrap draws so
    the comparison is paired): percentile covered **86.3%**, BCa **88.4%**,
    mean width 1.08 vs 1.17. Both are still shy of nominal at n = 10, as any
    bootstrap is; BCa errs in the honest direction. The size of that gain
    should not be oversold — it is ~2 points, and in paired reruns at 200-400
    trials it ranged from +2.0 to −1.0 points, i.e. within Monte-Carlo error
    at those trial counts. It is a second-order-accuracy improvement, not a
    rescue; the guard above is what fixes the headline defect. The two
    corrections (Efron 1987):

      z0  — median bias: Φ⁻¹ of the fraction of bootstrap means below the
            sample mean. Ties count half — twin scores are 0.1-quantised, so
            bootstrap means exactly equal to the sample mean are routine here,
            and counting them wholly on either side would fabricate a shift.
      a   — acceleration: jackknife skewness of the mean,
            a = Σ(θ̄−θᵢ)³ / (6·[Σ(θ̄−θᵢ)²]^{3/2}) over leave-one-out means θᵢ.

    The nominal percentile levels α/2 and 1−α/2 are then remapped through
    Φ(z0 + (z0+z)/(1−a·(z0+z))) before indexing the sorted bootstrap means.

    A **constant** sample returns a zero-width interval, e.g. 20 twins all
    reporting 0.5 gives (0.5, 0.5). That is the literally correct bootstrap
    answer — every resample of a constant sample has the same mean — but it is
    not an honest uncertainty statement: the width reflects the 0.1 quantisation
    of LLM twin scores, not an absence of sampling error. Unanimity is common
    here, so this is a routine case rather than a corner one.

    It is deliberately NOT collapsed to None. Consumers use interval separation
    as a decision gate (`optimizer.beat_seed`), and withholding the interval
    makes two well-separated unanimous arms read as "no evidence of a
    difference" — trading a display problem for a wrong decision. Callers that
    render the interval should detect `hi - lo == 0` and label it as no
    observed variation rather than as a tight interval.

    The constant case is answered BEFORE the bias correction is attempted:
    with every bootstrap mean equal to the sample mean, the "fraction below"
    is 0 and z0 = Φ⁻¹(0) = −∞. BCa degenerates gracefully only because this
    edge is taken explicitly, and p0 is clamped to [1/B, 1−1/B] for the
    near-degenerate remainder.
    """
    n = len(values)
    if n < MIN_BOOTSTRAP_N:
        return None
    sample_mean = sum(values) / n
    if min(values) == max(values):
        # Constant sample: see the z0 = −∞ note above.
        return (round(sample_mean, 4), round(sample_mean, 4))

    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(iterations):
        total = 0.0
        for _ in range(n):
            total += values[rng.randrange(n)]
        means.append(total / n)
    means.sort()
    if means[0] == means[-1]:
        # All resamples agreed despite a non-constant sample (vanishingly rare
        # at B = 2000, but the z0 edge below must never see it).
        return (round(means[0], 4), round(means[0], 4))

    # z0 — median bias, ties split evenly.
    below = 0.0
    for m_b in means:
        if m_b < sample_mean:
            below += 1.0
        elif m_b == sample_mean:
            below += 0.5
    p0 = min(1.0 - 1.0 / iterations, max(1.0 / iterations, below / iterations))
    z0 = _norm_ppf(p0)

    # a — acceleration from jackknife skewness of the leave-one-out means.
    total = sum(values)
    jack = [(total - v) / (n - 1) for v in values]
    jbar = sum(jack) / n
    num = sum((jbar - j) ** 3 for j in jack)
    den = sum((jbar - j) ** 2 for j in jack) ** 1.5
    accel = num / (6.0 * den) if den > 1e-24 else 0.0

    def _adjusted(z_alpha: float) -> float:
        shifted = z0 + z_alpha
        denom = 1.0 - accel * shifted
        if denom <= 1e-12:
            # The remap has run off the end of its own validity; saturate to
            # the extreme order statistic rather than divide by ~zero.
            return 1.0 if shifted > 0 else 0.0
        return _norm_cdf(z0 + shifted / denom)

    def _order_stat(level: float) -> float:
        return means[min(iterations - 1, max(0, int(level * iterations)))]

    lo = _order_stat(_adjusted(_norm_ppf(alpha / 2)))
    hi = _order_stat(_adjusted(_norm_ppf(1 - alpha / 2)))
    if lo > hi:  # possible only under saturation; never silently inverted
        lo, hi = hi, lo
    return (round(lo, 4), round(hi, 4))


def simultaneous_band(
    per_index: Sequence[Sequence[float]],
    iterations: int = 2000,
    alpha: float = 0.05,
    seed: int = 7,
) -> Optional[dict]:
    """Sup-t simultaneous confidence band for a mean CURVE.

    WHY A CURVE NEEDS A DIFFERENT INTERVAL. A neuro study reports nine
    dimensions across up to ten beats — up to ninety intervals, each at nominal
    95%. Read one, 95% is right. Read the chart, which is the only way anyone
    reads it, and the probability that EVERY interval covers its truth is
    nothing like 95%: under independence 0.95⁹⁰ ≈ 1%. Every pointwise band
    drawn across a curve is an invitation to find the one beat that looks
    different, and at ninety looks, one of them always will.

    Two things are wrong with the pointwise version and this fixes both.

    **Resampling is now joint.** Calling `bootstrap_ci` once per beat resamples
    the twins independently at each beat, which silently asserts that a twin's
    beat-3 rating tells you nothing about its beat-4 rating. Twins have
    baselines — some rate everything high — so that is false, and it distorts
    the shape of the band. Here a bootstrap replicate draws a set of TWINS and
    uses that same set at every beat, so each replicate is a curve the study
    could actually have produced.

    **Coverage is simultaneous.** The critical value is the (1−α) quantile of
    the bootstrap distribution of max_i |mean_b(i) − mean(i)| / se(i) — the
    largest standardised deviation anywhere on the curve. A band built from it
    contains the entire true curve with probability 1−α, so "beat 4 is outside
    the band" is a claim that survives having looked at all ten beats.

    Sup-t rather than Bonferroni because the beats are strongly correlated and
    Bonferroni ignores that, paying full price for every comparison as though
    they were independent. Both critical values are returned so the width of
    the correction is visible: on real beat curves sup-t is typically 20-40%
    narrower.

    `per_index[i]` is every subject's value at index i, and the subject order
    must be the SAME in each — that correspondence is what makes joint
    resampling meaningful.
    """
    m = len(per_index)
    if m == 0:
        return None
    n = len(per_index[0])
    if n < 3 or any(len(col) != n for col in per_index):
        return None

    point = [sum(col) / n for col in per_index]

    def _se(col: Sequence[float], mu: float) -> float:
        """Standard error of the mean, sample convention (n−1)."""
        if n < 2:
            return 0.0
        return math.sqrt(sum((x - mu) ** 2 for x in col) / (n - 1) / n)

    se = [_se(per_index[i], point[i]) for i in range(m)]

    rng = random.Random(seed)
    curves: list[list[float]] = []
    # STUDENTIZED. Each replicate is standardised by ITS OWN standard error,
    # not by the whole sample's. The unstudentized version is the obvious one
    # and it undercovers badly at the sample sizes this runs on: measured
    # coverage of the full 8-beat curve was 84% against a 95% target at 12
    # twins, because the critical value is taken from a distribution that has
    # already conditioned on the observed spread. Studentizing gives the
    # bootstrap-t, whose error is second-order rather than first, and it is
    # what supplies the fatter-than-normal tails a 12-twin study deserves.
    t_stats: list[float] = []
    for _ in range(iterations):
        idx = [rng.randrange(n) for _ in range(n)]
        drawn = [[per_index[i][j] for j in idx] for i in range(m)]
        means = [sum(col) / n for col in drawn]
        curves.append(means)
        best = 0.0
        seen = False
        for i in range(m):
            se_b = _se(drawn[i], means[i])
            # A replicate that drew the same twin n times has zero spread at
            # every index; it carries no information about the maximum and
            # would divide by zero. Dropped from that replicate's max rather
            # than floored — a floor lets a degenerate draw produce an
            # arbitrarily large ratio and hijack the upper quantile.
            if se_b <= 1e-12:
                continue
            seen = True
            best = max(best, abs(means[i] - point[i]) / se_b)
        if seen:
            t_stats.append(best)

    # Indices with no variation at all carry no information about the maximum
    # and would divide by zero.
    live = [i for i in range(m) if se[i] > 1e-12]
    if not live:
        flat = [[round(point[i], 4), round(point[i], 4)] for i in range(m)]
        return {
            "point": [round(v, 4) for v in point],
            "band": flat,
            "pointwise": flat,
            "critical_value": 0.0,
            "bonferroni_critical_value": 0.0,
            "se": [0.0] * m,
            "n_subjects": n,
            "n_indices": m,
            "alpha": alpha,
            "degenerate": True,
            "method": "no bootstrap variation — every subject agrees at every index",
        }

    if not t_stats:
        return None
    t_stats.sort()
    crit = t_stats[min(len(t_stats) - 1, int((1 - alpha) * len(t_stats)))]

    # Pointwise percentile intervals, for the width comparison only.
    pointwise: list[list[float]] = []
    for i in range(m):
        col = sorted(c[i] for c in curves)
        lo = col[max(0, int((alpha / 2) * len(col)))]
        hi = col[min(len(col) - 1, int((1 - alpha / 2) * len(col)))]
        pointwise.append([round(lo, 4), round(hi, 4)])

    # Bonferroni's critical value on the same scale, for reference: the
    # two-sided normal quantile at α/m, via the inverse-erf identity
    # z = √2·erf⁻¹(1−α/m). Reported, never used — it is the number that shows
    # what accounting for correlation bought.
    q = 1.0 - alpha / m
    z_bonf = math.sqrt(2.0) * _erf_inv(2.0 * q - 1.0)

    return {
        "point": [round(v, 4) for v in point],
        "band": [
            [round(point[i] - crit * se[i], 4), round(point[i] + crit * se[i], 4)]
            for i in range(m)
        ],
        "pointwise": pointwise,
        "critical_value": round(crit, 4),
        "bonferroni_critical_value": round(z_bonf, 4),
        "se": [round(s, 5) for s in se],
        "n_subjects": n,
        "n_indices": m,
        "alpha": alpha,
        "degenerate": False,
        "method": (
            f"sup-t simultaneous band over {m} indices, subject-level bootstrap "
            f"({iterations} replicates); covers the whole curve at {1 - alpha:.0%}"
        ),
    }


def _erf_inv(x: float) -> float:
    """Inverse error function (Giles 2010 rational approximation).

    Stdlib has `math.erf` but no inverse, and the only use here is a reference
    number, so a compact approximation beats pulling in scipy for it. Accurate
    to ~1e-9 over the range that matters, which is far past what a displayed
    critical value needs.
    """
    x = max(-0.999999999999, min(0.999999999999, x))
    w = -math.log((1.0 - x) * (1.0 + x))
    if w < 5.0:
        w -= 2.5
        p = 2.81022636e-08
        for c in (
            3.43273939e-07, -3.5233877e-06, -4.39150654e-06,
            0.00021858087, -0.00125372503, -0.00417768164,
            0.246640727, 1.50140941,
        ):
            p = p * w + c
    else:
        w = math.sqrt(w) - 3.0
        p = -0.000200214257
        for c in (
            0.000100950558, 0.00134934322, -0.00367342844,
            0.00573950773, -0.0076224613, 0.00943887047,
            1.00167406, 2.83297682,
        ):
            p = p * w + c
    return p * x


def _norm_cdf(z: float) -> float:
    """Standard normal CDF via `math.erf` — exact to double precision.

    cognition.py carries its own copy; it cannot be imported from there
    without inverting the module dependency (cognition sits above analytics).
    """
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Standard normal quantile, z = √2·erf⁻¹(2p−1), via `_erf_inv` above.

    Accuracy follows `_erf_inv` (~1e-9), far past what a bootstrap percentile
    level needs — the level is about to be floored onto one of B = 2000
    order statistics anyway.
    """
    return math.sqrt(2.0) * _erf_inv(2.0 * p - 1.0)


def benjamini_hochberg(
    p_values: Sequence[Optional[float]], alpha: float = 0.05
) -> dict:
    """Benjamini-Hochberg step-up FDR control over a family of tests.

    The codebase runs many tests per study and, until now, controlled none of
    them: a neuro run emits dozens of intervals, the optimizer selects from
    ~40 arms, and every one is judged at its own nominal 5%. Reporting the
    significant subset of forty independent nulls means reporting two findings
    per study, forever, from data with nothing in it.

    FDR rather than family-wise error because of what this is for. Bonferroni
    controls the probability of ANY false claim, which is the right target when
    one wrong answer is catastrophic; here the researcher is triaging a ranked
    list of leads and would rather see ten findings of which one is spurious
    than one finding and nine missed. That is exactly the trade BH makes, and
    it is strictly more powerful.

    Returns q-values aligned to the input (None passes through as None, so a
    caller can hand in a ragged family without re-indexing), plus which
    hypotheses survive at `alpha` and the effective p-value cut.
    """
    indexed = [(i, p) for i, p in enumerate(p_values) if p is not None]
    q: list[Optional[float]] = [None] * len(p_values)
    if not indexed:
        return {
            "q_values": q, "rejected": [False] * len(p_values),
            "n_tests": 0, "n_rejected": 0, "alpha": alpha,
            "cutoff": None, "method": "Benjamini-Hochberg (no testable hypotheses)",
        }

    indexed.sort(key=lambda t: t[1])
    m = len(indexed)
    # Step-up: q₍ᵢ₎ = min over j ≥ i of (m·p₍ⱼ₎ / j), which enforces the
    # monotonicity BH requires — a larger p can never earn a smaller q.
    running = 1.0
    for rank in range(m, 0, -1):
        i, p = indexed[rank - 1]
        running = min(running, min(1.0, m * p / rank))
        q[i] = round(running, 6)

    rejected = [bool(v is not None and v <= alpha) for v in q]
    passing = [p for (i, p), r in zip(indexed, (rejected[i] for i, _ in indexed)) if r]
    return {
        "q_values": q,
        "rejected": rejected,
        "n_tests": m,
        "n_rejected": sum(rejected),
        "alpha": alpha,
        # The largest raw p that still survived — what the 5% actually became.
        "cutoff": round(max(passing), 6) if passing else None,
        "method": (
            f"Benjamini-Hochberg step-up FDR at q<{alpha} over {m} simultaneous "
            "tests"
        ),
    }


# ────────────────────────────────────────────────────────── Bayesian A/B


class ArmPosterior:
    """Beta(1+successes, 1+failures) posterior summary for one variant."""

    def __init__(
        self,
        name: str,
        successes: int,
        trials: int,
        p_best: float,
        ci: tuple[float, float],
        expected_loss: float,
    ):
        self.name = name
        self.successes = successes
        self.trials = trials
        self.mean = (1 + successes) / (2 + trials)  # posterior mean
        self.p_best = p_best
        self.ci = ci
        self.expected_loss = expected_loss

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "successes": self.successes,
            "trials": self.trials,
            "posterior_mean": round(self.mean, 4),
            "p_best": round(self.p_best, 4),
            "ci_low": round(self.ci[0], 4),
            "ci_high": round(self.ci[1], 4),
            "expected_loss": round(self.expected_loss, 5),
        }


def bayesian_ab(
    arms: dict[str, tuple[int, int]],
    draws: int = 20000,
    seed: int = 11,
    rope: float = 0.01,
) -> dict:
    """Bayesian A/B test over Beta-Binomial arms.

    `arms` maps variant name → (successes, trials) with a uniform Beta(1,1)
    prior. We Monte-Carlo the joint posterior to obtain, per arm:

      · p_best         P(this arm has the highest true rate)
      · ci             95% credible interval on its rate
      · expected_loss  E[max(0, θ_best − θ_this)] — the expected regret, in
                       rate points, of shipping this arm. The decision rule
                       used in practice is "ship when expected loss < ROPE".

    Unlike a p-value this answers the question a product owner actually asks:
    *how likely is B genuinely better, and what does being wrong cost me?*
    """
    names = [n for n in arms]
    if not names:
        return {"arms": [], "winner": None, "conclusive": False, "draws": 0}

    rng = random.Random(seed)
    samples: dict[str, list[float]] = {}
    for name in names:
        successes, trials = arms[name]
        successes = max(0, min(successes, trials))
        a = 1.0 + successes
        b = 1.0 + (trials - successes)
        samples[name] = [rng.betavariate(a, b) for _ in range(draws)]

    wins = {n: 0 for n in names}
    loss_total = {n: 0.0 for n in names}
    for i in range(draws):
        best_name = names[0]
        best_val = samples[best_name][i]
        for n in names[1:]:
            v = samples[n][i]
            if v > best_val:
                best_val, best_name = v, n
        wins[best_name] += 1
        for n in names:
            loss_total[n] += best_val - samples[n][i]

    lo_idx = int(0.025 * draws)
    hi_idx = min(draws - 1, int(0.975 * draws))
    posteriors: list[ArmPosterior] = []
    for n in names:
        ordered = sorted(samples[n])
        successes, trials = arms[n]
        posteriors.append(
            ArmPosterior(
                name=n,
                successes=successes,
                trials=trials,
                p_best=wins[n] / draws,
                ci=(ordered[lo_idx], ordered[hi_idx]),
                expected_loss=loss_total[n] / draws,
            )
        )

    posteriors.sort(key=lambda p: p.p_best, reverse=True)
    leader = posteriors[0]
    # "Conclusive" = the leader's expected regret is inside the region of
    # practical equivalence, i.e. picking it costs ~nothing even if we're wrong.
    conclusive = leader.expected_loss < rope
    return {
        "arms": [p.to_dict() for p in posteriors],
        "winner": leader.name,
        "p_win": round(leader.p_best, 4),
        "expected_loss": round(leader.expected_loss, 5),
        "conclusive": conclusive,
        "rope": rope,
        "draws": draws,
        "verdict": (
            f"{leader.name} leads with {leader.p_best:.0%} probability of being best"
            + (
                " — conclusive at the current sample size."
                if conclusive
                else " — NOT yet conclusive; run more twins before acting."
            )
        ),
    }


# ────────────────────────────────────────────────────────── regression / pricing


def _ols(
    xs: Sequence[float], ys: Sequence[float]
) -> Optional[tuple[float, float, Optional[float]]]:
    """Simple linear regression y = intercept + slope·x.

    Returns (slope, intercept, r2), where **r2 is None when it is undefined** —
    i.e. when the response is constant and the total sum of squares is zero.

    Reporting 1.0 there (the previous behaviour) is not a rounding choice, it
    inverts the meaning: R² = 1 − SS_res/SS_tot is 0/0 for a flat response, and
    claiming a perfect fit made a perfectly price-insensitive product read as a
    Veblen good with R² = 1.00 on both fits. None forces callers to say
    "undefined" instead of "certain".
    """
    n = len(xs)
    if n < 3 or n != len(ys):
        return None
    mx, my = mean(xs), mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return None
    sxy = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    slope = sxy / sxx
    intercept = my - slope * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((ys[i] - (intercept + slope * xs[i])) ** 2 for i in range(n))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else None
    return (slope, intercept, r2)


def price_elasticity(prices: Sequence[float], demand: Sequence[float]) -> dict:
    """Econometric read of a demand schedule.

    Two complementary models:

    1. **Log-log OLS**  ln Q = α + ε·ln P.  The slope ε *is* the price
       elasticity of demand — the % change in quantity per 1% change in price.
       |ε| > 1 ⇒ elastic (buyers flee on price); |ε| < 1 ⇒ inelastic (pricing
       power). This is the standard economic measure and is unit-free.

    2. **Linear demand**  Q = a + b·P  ⇒  revenue R(P) = aP + bP².
       Setting dR/dP = 0 gives a *continuous* revenue optimum P* = −a/(2b),
       which can fall between the tested price points — unlike an argmax over
       the discrete grid, which can only ever return a price you happened to
       test.

    Returns the elasticity, both fits' R², the optimum (clamped to the tested
    range, since extrapolating a fitted demand curve is not defensible), and a
    plain-language classification.
    """
    pairs = [(p, q) for p, q in zip(prices, demand) if p > 0]
    if len(pairs) < 3:
        return {"ok": False, "reason": "need at least 3 price points"}
    ps = [p for p, _ in pairs]
    qs = [q for _, q in pairs]

    # 1. elasticity from log-log (drop zero-demand points: ln 0 undefined)
    log_pairs = [(math.log(p), math.log(q)) for p, q in pairs if q > 1e-9]
    elasticity = None
    elasticity_r2 = None
    if len(log_pairs) >= 3:
        fit = _ols([x for x, _ in log_pairs], [y for _, y in log_pairs])
        if fit:
            elasticity, _, elasticity_r2 = fit

    # 2. linear demand model → continuous revenue optimum
    lin = _ols(ps, qs)
    optimal_price = None
    demand_r2 = None
    if lin:
        b, a, demand_r2 = lin  # slope, intercept
        if b < -1e-12:  # downward-sloping demand, as theory requires
            p_star = -a / (2 * b)
            optimal_price = min(max(p_star, min(ps)), max(ps))  # clamp to tested range

    # A flat demand schedule is a real and important result — buyers simply do
    # not respond to price over the tested range — but ε = 0 with an undefined
    # R² used to fall through to the "demand rises with price" branch and get
    # reported as a prestige good. Perfectly inelastic is its own verdict.
    flat_demand = elasticity is not None and abs(elasticity) < 1e-9 and elasticity_r2 is None

    if elasticity is None:
        classification = "undetermined"
    elif flat_demand:
        classification = (
            "perfectly inelastic over the tested range — demand did not move at all; "
            "widen the price spread before concluding you have pricing power"
        )
    elif elasticity <= -1.0:
        classification = "elastic — demand falls faster than price rises; discounting grows revenue"
    elif elasticity > 0:
        classification = "anomalous — demand rises with price (prestige effect or noisy sample)"
    else:
        classification = "inelastic — buyers tolerate price; you have room to raise it"

    return {
        "ok": True,
        "elasticity": round(elasticity, 4) if elasticity is not None else None,
        "elasticity_r2": round(elasticity_r2, 4) if elasticity_r2 is not None else None,
        "demand_fit_r2": round(demand_r2, 4) if demand_r2 is not None else None,
        "continuous_optimal_price": round(optimal_price, 2) if optimal_price is not None else None,
        "classification": classification,
    }


# ────────────────────────────────────────────────────────── survival analysis


#: Two-sided 95% normal quantile, to the precision the closed forms below use.
_Z95 = 1.959963984540054


def _wilson_interval(k: int, n: int, z: float = _Z95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion k/n.

    Chosen over Wald p̂ ± z·√(p̂(1−p̂)/n) for the same reason twice over: at
    k = 0 and k = n the Wald width is exactly zero — certainty from any n,
    including n = 1 — which is the pathology being removed from the survival
    bands. Wilson inverts the score test instead, stays inside (0, 1), and at
    k = n = 1 its lower bound is 0.2065: one observation's worth of evidence.
    """
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))


def kaplan_meier(entered: Sequence[int], dropped: Sequence[int]) -> dict:
    """Discrete-time Kaplan-Meier survival for a funnel.

    At step i, `entered[i]` agents reached it and `dropped[i]` abandoned there.

      hazard   h_i = d_i / n_i     conditional risk of dying *at* step i
      survival S_i = Π_{j≤i} (1 − h_j)

    Hazard is the key output: it normalises by how many agents actually
    *reached* a step, so a step that kills 5 of 10 is correctly flagged as
    worse than one that kills 8 of 100 — something raw drop-off counts get
    exactly backwards.

    BANDS ARE LOG-LOG (Kalbfleisch-Prentice), NOT WALD. The old band was
    Ŝ ± 1.96·se with Greenwood's variance Var(Ŝ) = Ŝ²·G,
    G = Σ d_j/(n_j(n_j−d_j)). Measured failure: a step with entered=1,
    dropped=1 reported `survival_ci = [0.0, 0.0]` — 95% certainty of
    extinction from ONE observation, because the factor Ŝ = 0 kills the width
    regardless of how little data produced it (and near Ŝ = 1 the same band
    escapes [0, 1] and has to be clipped). On θ = log(−log Ŝ) the variance is
    Greenwood's divided by (Ŝ·log Ŝ)², i.e. Var(θ̂) = G/(log Ŝ)², and the band

      exp(−exp(θ̂ ∓ z·se_θ))   (∓ because S is decreasing in θ)

    respects [0, 1] by construction. Same 1-of-1 step now: [0.0, 0.7935].

    The transform is undefined at both edges, handled explicitly:
      Ŝ = 1 (no drop observed yet): upper limit 1; the lower limit is the
        running product of (1 − Wilson upper bound of each step's hazard) —
        conservative (it compounds per-step bounds), but 0-of-1 honestly reads
        [0.2065, 1.0] where Wald read [1.0, 1.0].
      Ŝ = 0 (a step dropped everyone remaining): lower limit 0; the upper
        limit multiplies the previous step's upper limit by (1 − Wilson lower
        bound of the terminal hazard), so the width scales with how much
        evidence the terminal step actually carried: 1-of-1 → 0.7935,
        100-of-100 → 0.0370.

    WORST STEP BY WILSON LOWER BOUND. `worst_step_index` was an unguarded
    argmax on raw hazard, so a 1-of-1 step (ĥ = 1.0) outranked 100-of-200
    (ĥ = 0.5) — the certainty-from-nothing mistake again, one level up.
    Ranking by the Wilson lower bound asks how bad each step still is under
    its own sampling noise: measured, wilson_low(1,1) = 0.207 <
    wilson_low(5,10) = 0.237 < wilson_low(100,200) = 0.431, so well-attested
    leaks outrank single observations while a genuinely murderous step with
    real n still wins. The per-step statistic ships as `hazard_wilson_low`;
    `worst_step_hazard` remains the raw hazard at the chosen step.
    """
    steps = []
    survival = 1.0
    greenwood_sum = 0.0
    band_low, band_high = 1.0, 1.0  # carried un-rounded across steps
    worst_idx = None
    worst_hazard = -1.0
    worst_wilson = -1.0

    for i, (n_i, d_i) in enumerate(zip(entered, dropped)):
        if n_i <= 0:
            # Nobody reached this step: nothing observed, nothing updated —
            # the band carries forward instead of collapsing to a point.
            steps.append(
                {
                    "step_index": i,
                    "entered": n_i,
                    "dropped": d_i,
                    "hazard": 0.0,
                    "hazard_wilson_low": 0.0,
                    "survival": round(survival, 4),
                    "survival_ci_low": round(band_low, 4),
                    "survival_ci_high": round(band_high, 4),
                }
            )
            continue
        d_i = min(d_i, n_i)
        hazard = d_i / n_i
        w_low, w_high = _wilson_interval(d_i, n_i)
        survival *= 1.0 - hazard
        if n_i - d_i > 0:
            greenwood_sum += d_i / (n_i * (n_i - d_i))

        if survival <= 0.0:
            band_low = 0.0
            band_high = band_high * (1.0 - w_low)
        elif greenwood_sum <= 0.0:
            # Ŝ = 1: every step so far dropped nobody.
            band_high = 1.0
            band_low = band_low * (1.0 - w_high)
        else:
            log_s = math.log(survival)
            theta = math.log(-log_s)
            se_theta = math.sqrt(greenwood_sum) / -log_s
            band_low = math.exp(-math.exp(theta + _Z95 * se_theta))
            band_high = math.exp(-math.exp(theta - _Z95 * se_theta))

        steps.append(
            {
                "step_index": i,
                "entered": n_i,
                "dropped": d_i,
                "hazard": round(hazard, 4),
                "hazard_wilson_low": round(w_low, 4),
                "survival": round(survival, 4),
                "survival_ci_low": round(band_low, 4),
                "survival_ci_high": round(band_high, 4),
            }
        )
        if w_low > worst_wilson:
            worst_wilson, worst_hazard, worst_idx = w_low, hazard, i

    return {
        "steps": steps,
        "overall_survival": round(survival, 4),
        "worst_step_index": worst_idx,
        "worst_step_hazard": round(worst_hazard, 4) if worst_idx is not None else None,
        "worst_step_hazard_wilson_low": (
            round(worst_wilson, 4) if worst_idx is not None else None
        ),
    }


# ────────────────────────────────────────────────────────── clustering


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    return dot / (na * nb)


def agglomerative_cluster(
    vectors: list[Sequence[float]], threshold: float = 0.82
) -> list[list[int]]:
    """Average-linkage agglomerative clustering over cosine similarity.

    Repeatedly merges the two clusters whose *average* pairwise similarity is
    highest, stopping once no pair exceeds `threshold`. Average linkage (rather
    than single linkage) is deliberate: it resists the "chaining" failure where
    a string of marginally-similar items collapses into one giant blob.

    O(n²·log n)-ish on small inputs, which is all we need (≤ a few hundred
    objections). Returns clusters as lists of original indices.
    """
    n = len(vectors)
    if n == 0:
        return []
    sim = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            s = cosine_similarity(vectors[i], vectors[j])
            sim[i][j] = sim[j][i] = s

    clusters: list[list[int]] = [[i] for i in range(n)]
    while len(clusters) > 1:
        best, bi, bj = threshold, -1, -1
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                pairs = [(a, b) for a in clusters[i] for b in clusters[j]]
                avg = sum(sim[a][b] for a, b in pairs) / len(pairs)
                if avg > best:
                    best, bi, bj = avg, i, j
        if bi < 0:
            break
        clusters[bi] = clusters[bi] + clusters[bj]
        clusters.pop(bj)

    clusters.sort(key=len, reverse=True)
    return clusters


def medoid(indices: Sequence[int], vectors: list[Sequence[float]]) -> int:
    """The member with the highest mean similarity to its cluster-mates — i.e.
    the most representative phrasing, used as the cluster's display label."""
    if len(indices) == 1:
        return indices[0]
    best_idx, best_score = indices[0], -2.0
    for i in indices:
        score = mean([cosine_similarity(vectors[i], vectors[j]) for j in indices if j != i])
        if score > best_score:
            best_score, best_idx = score, i
    return best_idx


def silhouette_score(
    vectors: list[Sequence[float]], clusters: list[list[int]]
) -> Optional[float]:
    """Mean silhouette coefficient of a clustering, under cosine distance.

    For each point i: a(i) = mean distance to its own cluster-mates,
    b(i) = the smallest mean distance to any other cluster, and

        s(i) = (b(i) − a(i)) / max(a(i), b(i))

    s ∈ [−1, 1]; near 1 means points sit deep inside well-separated clusters,
    near 0 means clusters overlap, negative means points are mis-assigned.
    Points in singleton clusters take s(i) = 0 by convention. Undefined
    (returns None) when there are fewer than 2 clusters — a single cluster has
    no "nearest other cluster" to compare against.
    """
    if len(clusters) < 2:
        return None

    def dist(i: int, j: int) -> float:
        return 1.0 - cosine_similarity(vectors[i], vectors[j])

    member_of = {i: ci for ci, cluster in enumerate(clusters) for i in cluster}
    scores: list[float] = []
    for cluster in clusters:
        for i in cluster:
            if len(cluster) == 1:
                scores.append(0.0)
                continue
            a = mean([dist(i, j) for j in cluster if j != i])
            b = min(
                mean([dist(i, j) for j in other])
                for ci, other in enumerate(clusters)
                if ci != member_of[i]
            )
            denom = max(a, b)
            scores.append(0.0 if denom <= 1e-12 else (b - a) / denom)
    return mean(scores)


def auto_cluster(
    vectors: list[Sequence[float]],
    thresholds: Sequence[float] = (0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90),
    fallback_threshold: float = 0.82,
) -> tuple[list[list[int]], float, Optional[float]]:
    """Agglomerative clustering with the merge cutoff CHOSEN BY THE DATA.

    Runs the clustering at each candidate threshold and keeps the partition
    with the highest mean silhouette — i.e. model selection by internal
    validity rather than a hand-tuned constant. Degenerate partitions (one
    giant cluster, or all singletons) score no silhouette and are only used
    as a fallback. Returns (clusters, chosen_threshold, silhouette).
    """
    n = len(vectors)
    if n < 3:
        return agglomerative_cluster(vectors, fallback_threshold), fallback_threshold, None

    best: Optional[tuple[list[list[int]], float, float]] = None
    for t in thresholds:
        clusters = agglomerative_cluster(vectors, t)
        if not (1 < len(clusters) < n):
            continue  # degenerate: nothing to validate
        score = silhouette_score(vectors, clusters)
        if score is None:
            continue
        if best is None or score > best[2]:
            best = (clusters, t, score)

    if best is not None:
        return best
    clusters = agglomerative_cluster(vectors, fallback_threshold)
    return clusters, fallback_threshold, silhouette_score(vectors, clusters)


# ────────────────────────────────────────────────────────── adaptive allocation


# ────────────────────────────────────────────────────────── k-means segmentation


def _euclid(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _kmeans_once(
    points: list[Sequence[float]], k: int, seed: int, iters: int = 40
) -> tuple[list[int], float]:
    """One k-means run with k-means++ seeding. Returns (labels, inertia).

    k-means++ picks each next centre with probability proportional to D², the
    squared distance to the nearest already-chosen centre — this is the
    Arthur–Vassilvitskii initialisation with an O(log k) expected optimality
    guarantee, versus arbitrarily bad plain random seeding.
    """
    rng = random.Random(seed)
    n = len(points)
    centres: list[list[float]] = [list(points[rng.randrange(n)])]
    while len(centres) < k:
        d2 = [min(_euclid(p, c) ** 2 for c in centres) for p in points]
        total = sum(d2)
        if total <= 1e-12:  # all points identical — duplicate a centre
            centres.append(list(points[rng.randrange(n)]))
            continue
        r = rng.random() * total
        acc = 0.0
        for i, w in enumerate(d2):
            acc += w
            if acc >= r:
                centres.append(list(points[i]))
                break

    labels = [0] * n
    for _ in range(iters):
        new_labels = [
            min(range(k), key=lambda c: _euclid(p, centres[c])) for p in points
        ]
        if new_labels == labels and _ > 0:
            break
        labels = new_labels
        for c in range(k):
            members = [points[i] for i in range(n) if labels[i] == c]
            if members:
                centres[c] = [mean([m[d] for m in members]) for d in range(len(points[0]))]
    inertia = sum(_euclid(points[i], centres[labels[i]]) ** 2 for i in range(n))
    return labels, inertia


def _euclid_silhouette(points: list[Sequence[float]], labels: list[int]) -> float:
    """Mean silhouette under euclidean distance (labels form)."""
    n = len(points)
    ks = sorted(set(labels))
    if len(ks) < 2:
        return 0.0
    scores = []
    for i in range(n):
        own = [j for j in range(n) if labels[j] == labels[i] and j != i]
        if not own:
            scores.append(0.0)
            continue
        a = mean([_euclid(points[i], points[j]) for j in own])
        b = min(
            mean([_euclid(points[i], points[j]) for j in range(n) if labels[j] == c])
            for c in ks
            if c != labels[i]
        )
        denom = max(a, b)
        scores.append(0.0 if denom <= 1e-12 else (b - a) / denom)
    return mean(scores)


def _best_inertia(
    points: list[Sequence[float]], k: int, seed: int, restarts: int
) -> tuple[list[int], float]:
    """Lowest-inertia partition over `restarts` k-means++ runs."""
    best: Optional[tuple[list[int], float]] = None
    for r in range(restarts):
        labels, inertia = _kmeans_once(points, k, seed=seed + 97 * k + r)
        if best is None or inertia < best[1]:
            best = (labels, inertia)
    assert best is not None
    return best


def kmeans_segments(
    points: list[Sequence[float]],
    k_max: int = 4,
    n_reference: int = 100,
    min_spread: float = 0.15,
    seed: int = 23,
    restarts: int = 3,
) -> Optional[dict]:
    """Segment discovery by the GAP STATISTIC (Tibshirani, Walther & Hastie 2001).

    WHAT WAS WRONG. The previous selector maximised silhouette over k ∈ {2,3,4}
    and gated on `silhouette ≥ 0.4` plus a centroid-spread ROPE. Both gates are
    inert on the domain this is actually called on — `(intent_score,
    engagement)` ∈ [0,1]² — because silhouette is scale-invariant and k-means
    partitions *any* cloud, including a structureless one, into compact-looking
    pieces. Measured false-discovery rate on pure uniform noise, 300 trials per
    cell:

        n twins        8      12      20      40
        uniform     85.7%   89.7%   89.7%   89.7%
        quantised   85.3%   89.7%   88.0%   89.0%

    Nine runs in ten invented an audience split that was not there, and
    downstream those fabricated groups are labelled "enthusiasts" and
    "skeptics" and shown to a researcher as a finding. A selector that never
    says no is not a selector.

    THE FIX. The gap statistic compares the observed within-cluster dispersion
    against the dispersion of a NULL WITH NO CLUSTER STRUCTURE — B uniform draws
    over the data's own bounding box:

        Gap(k) = E*[log W*ₖ] − log Wₖ

    Structure exists when the observed dispersion falls *below* what
    featureless data of the same shape and scale produces. Two properties are
    what make this the right tool and silhouette the wrong one:

    · **k = 1 is a candidate.** Silhouette is undefined at k = 1, so the old
      code could not express "there are no segments" — it was structurally
      obliged to return at least two. The gap statistic ranks k = 1 against the
      rest on the same axis, so "no segments" is an answer it can actually give.
    · **The null is calibrated to the data's own scale.** The reference box is
      drawn from the observed min/max per dimension, so a tight cloud is
      compared against tight noise. No threshold to tune, and nothing for the
      quantisation of LLM scores to fool.

    Selection is Tibshirani's 1-SE rule: the smallest k whose gap is within one
    standard error of the next k's, where sₖ = sd(log W*ₖ)·√(1 + 1/B) carries
    the Monte-Carlo error of the reference draws.

    The centroid-spread ROPE is kept on top, unchanged — significance is not
    effect size, and two genuinely distinct segments 0.02 apart in intent do
    not change a decision.

    Returns None when there are no segments (k = 1, or the spread gate fires),
    which is itself a finding: a homogeneous audience is a real and reportable
    result. Otherwise a dict carrying the partition AND the evidence for it, so
    a reader can see how far from noise the split actually was.
    """
    n = len(points)
    if n < 6:
        return None
    dims = len(points[0])
    k_max = max(1, min(k_max, n - 1))

    lo = [min(p[d] for p in points) for d in range(dims)]
    hi = [max(p[d] for p in points) for d in range(dims)]
    if all(hi[d] - lo[d] <= 1e-12 for d in range(dims)):
        return None  # every point identical — no box to draw a null over

    rng = random.Random(seed)
    # A shared reference ensemble across all k: the same B synthetic datasets
    # are clustered at every k, so Gap(k) and Gap(k+1) differ by the data and
    # not by which noise happened to be drawn for each.
    references = [
        [
            tuple(lo[d] + rng.random() * (hi[d] - lo[d]) for d in range(dims))
            for _ in range(n)
        ]
        for _ in range(max(2, n_reference))
    ]

    # log W is undefined at W = 0, which happens whenever k partitions the data
    # into singletons or exact duplicates. Floored rather than special-cased:
    # the floor is far below any dispersion a real cloud produces, so it only
    # ever binds in the degenerate case it exists to keep finite.
    _log = lambda w: math.log(max(w, 1e-12))  # noqa: E731

    gaps: dict[int, float] = {}
    errs: dict[int, float] = {}
    parts: dict[int, list[int]] = {}
    for k in range(1, k_max + 1):
        labels, inertia = _best_inertia(points, k, seed, restarts)
        parts[k] = labels
        # One clustering per reference, not `restarts` of them. The reference
        # ensemble's job is to estimate E*[log W*ₖ] and its spread, and B = 100
        # independent draws already average away the restart-to-restart
        # variation that `restarts` exists to suppress on the single observed
        # dataset. Restarting here would triple the cost of the whole statistic
        # to sharpen a quantity that is about to be averaged anyway.
        ref_logs = [
            _log(_best_inertia(ref, k, seed + 1013, 1)[1]) for ref in references
        ]
        m = mean(ref_logs)
        gaps[k] = m - _log(inertia)
        sd = math.sqrt(sum((x - m) ** 2 for x in ref_logs) / len(ref_logs))
        errs[k] = sd * math.sqrt(1.0 + 1.0 / len(ref_logs))

    chosen = k_max
    for k in range(1, k_max):
        if gaps[k] >= gaps[k + 1] - errs[k + 1]:
            chosen = k
            break

    if chosen < 2:
        return None  # the honest answer the old selector could not produce

    labels = parts[chosen]
    if len(set(labels)) < 2:
        return None

    centroids = [
        [
            mean([points[i][d] for i in range(n) if labels[i] == c])
            for d in range(dims)
        ]
        for c in sorted(set(labels))
    ]
    spread = max(
        _euclid(centroids[i], centroids[j])
        for i in range(len(centroids))
        for j in range(i + 1, len(centroids))
    )
    if spread < min_spread:
        return None

    return {
        "labels": labels,
        "k": chosen,
        "silhouette": round(_euclid_silhouette(points, labels), 4),
        "gap": round(gaps[chosen], 4),
        "gap_se": round(errs[chosen], 4),
        # How far the chosen k sits above "no structure at all", in standard
        # errors. This is the number that says whether to believe the split.
        "gap_vs_k1": round((gaps[chosen] - gaps[1]) / max(errs[chosen], 1e-9), 2),
        "centroid_spread": round(spread, 4),
        "n_reference": len(references),
        "method": (
            "gap statistic vs uniform reference over the data's bounding box "
            "(Tibshirani et al. 2001), 1-SE rule, k=1 admissible"
        ),
    }


def kmeans_auto(
    points: list[Sequence[float]],
    k_range: Sequence[int] = (2, 3, 4),
    min_silhouette: float = 0.4,
    min_spread: float = 0.15,
    seed: int = 23,
    restarts: int = 3,
) -> Optional[tuple[list[int], int, float]]:
    """Back-compatible `(labels, k, silhouette)` wrapper over `kmeans_segments`.

    `min_silhouette` is accepted and ignored: it was the gate that let 90% of
    pure noise through, and honouring it now would only re-introduce a second,
    weaker opinion about the same question. Selection is the gap statistic.
    """
    result = kmeans_segments(
        points,
        k_max=max(k_range) if k_range else 4,
        min_spread=min_spread,
        seed=seed,
        restarts=restarts,
    )
    if result is None:
        return None
    return result["labels"], result["k"], result["silhouette"]


# ────────────────────────────────────────────────────────── PCA (semantic map)


def pca_2d(vectors: list[Sequence[float]], seed: int = 31) -> Optional[list[tuple[float, float]]]:
    """Project high-dimensional vectors to their top-2 principal components.

    Uses the Gram-matrix (kernel) trick: for n points in d dimensions with
    n ≪ d (e.g. 30 objections × 1536 embedding dims), the eigenvectors of the
    n×n Gram matrix K = X_c·X_cᵀ give the principal-component *scores* directly
    — score_j = u_j·√λ_j — without ever forming the d×d covariance matrix.
    Top eigenpairs are found by power iteration with deflation
    (K ← K − λ·uuᵀ), which converges geometrically at rate λ₂/λ₁.

    Output is scaled to [−1, 1] per axis for display. Returns None for n < 3.
    """
    n = len(vectors)
    if n < 3:
        return None
    d = len(vectors[0])
    # centre the data
    centre = [mean([v[j] for v in vectors]) for j in range(d)]
    xc = [[v[j] - centre[j] for j in range(d)] for v in vectors]
    # Gram matrix K = Xc Xcᵀ  (n×n)
    K = [[sum(xc[i][t] * xc[j][t] for t in range(d)) for j in range(n)] for i in range(n)]

    def power_iter(M: list[list[float]]) -> tuple[list[float], float]:
        rng = random.Random(seed)
        u = [rng.gauss(0, 1) for _ in range(n)]
        for _ in range(200):
            v = [sum(M[i][j] * u[j] for j in range(n)) for i in range(n)]
            norm = math.sqrt(sum(x * x for x in v))
            if norm <= 1e-12:
                return u, 0.0
            u = [x / norm for x in v]
        lam = sum(u[i] * sum(M[i][j] * u[j] for j in range(n)) for i in range(n))
        return u, max(0.0, lam)

    u1, l1 = power_iter(K)
    K2 = [[K[i][j] - l1 * u1[i] * u1[j] for j in range(n)] for i in range(n)]
    u2, l2 = power_iter(K2)

    xs = [u1[i] * math.sqrt(l1) for i in range(n)]
    ys = [u2[i] * math.sqrt(l2) for i in range(n)]
    mx = max(1e-9, max(abs(x) for x in xs))
    my = max(1e-9, max(abs(y) for y in ys))
    return [(round(xs[i] / mx, 4), round(ys[i] / my, 4)) for i in range(n)]


# ────────────────────────────────────────────────────────── adaptive allocation


def thompson_allocate(
    arms: dict[str, tuple[int, int]], wave: int, seed: int = 17
) -> dict[str, int]:
    """Batched Thompson sampling: allocate the next `wave` of agents.

    For each slot we draw θ ~ Beta(1+successes, 1+failures) from every arm's
    posterior and send the agent to the argmax. This is the classic Bayesian
    bandit policy: allocation is proportional to the current probability that
    each arm is best, so budget flows toward promising variants while weak
    ones still get occasional probes (posterior mass never hits zero).
    Deterministic for a given seed. Combined with an expected-loss stopping
    rule this implements a sequential experiment design — the same budget
    yields tighter posteriors on the arms that actually matter.
    """
    rng = random.Random(seed)
    names = list(arms)
    counts = {n: 0 for n in names}
    for _ in range(max(0, wave)):
        best_name, best_draw = names[0], -1.0
        for n in names:
            s, t = arms[n]
            s = max(0, min(s, t))
            draw = rng.betavariate(1.0 + s, 1.0 + (t - s))
            if draw > best_draw:
                best_draw, best_name = draw, n
        counts[best_name] += 1
    return counts
