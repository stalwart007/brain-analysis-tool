"""Statistical engine for CogniSwarm.

Everything here is a PURE function over plain numbers — no LLM calls, no I/O —
so each method is unit-testable against hand-computed answers. Randomised
procedures (bootstrap, Monte-Carlo posterior sampling) take an explicit seed and
are therefore deterministic and reproducible.

Implemented from first principles on the stdlib (no numpy/scipy dependency):

  · Beta-Binomial Bayesian A/B — posterior P(win), credible intervals,
    expected loss (the decision-theoretic "cost of choosing this arm").
  · Bootstrap percentile confidence intervals for any statistic of a sample.
  · Shannon entropy (normalised) and Sarle's bimodality coefficient — these
    detect a *split* swarm that a mean would hide.
  · Price elasticity of demand via log-log OLS, plus a linear-demand model
    whose revenue optimum is continuous (not restricted to tested points).
  · Kaplan-Meier discrete survival, per-step hazard, and Greenwood-variance
    confidence bands — the correct treatment of funnel drop-off.
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
    """
    positive = [c for c in counts if c > 0]
    total = sum(positive)
    support = k if k is not None else len(counts)
    if total <= 0 or support < 2:
        return 0.0
    if len(positive) < 2:
        return 0.0  # all mass on one outcome ⇒ perfect consensus
    h = -sum((c / total) * math.log(c / total) for c in positive)
    return h / math.log(support)


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


def bootstrap_ci(
    values: Sequence[float],
    iterations: int = 2000,
    alpha: float = 0.05,
    seed: int = 7,
) -> Optional[tuple[float, float]]:
    """Percentile bootstrap CI for the mean. Deterministic for a given seed.

    Returns None for samples too small to resample meaningfully (n < 2).

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
    """
    n = len(values)
    if n < 2:
        return None
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(iterations):
        total = 0.0
        for _ in range(n):
            total += values[rng.randrange(n)]
        means.append(total / n)
    means.sort()
    lo_idx = max(0, int((alpha / 2) * iterations))
    hi_idx = min(iterations - 1, int((1 - alpha / 2) * iterations) - 1)
    return (round(means[lo_idx], 4), round(means[hi_idx], 4))


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


def kaplan_meier(entered: Sequence[int], dropped: Sequence[int]) -> dict:
    """Discrete-time Kaplan-Meier survival for a funnel.

    At step i, `entered[i]` agents reached it and `dropped[i]` abandoned there.

      hazard   h_i = d_i / n_i     conditional risk of dying *at* step i
      survival S_i = Π_{j≤i} (1 − h_j)

    Greenwood's formula gives the variance of Ŝ:

      Var(Ŝ_i) = Ŝ_i² · Σ_{j≤i} d_j / (n_j (n_j − d_j))

    from which we derive a 95% band. Hazard is the key output: it normalises by
    how many agents actually *reached* a step, so a step that kills 5 of 10 is
    correctly flagged as worse than one that kills 8 of 100 — something raw
    drop-off counts get exactly backwards.
    """
    steps = []
    survival = 1.0
    greenwood_sum = 0.0
    worst_idx, worst_hazard = None, -1.0

    for i, (n_i, d_i) in enumerate(zip(entered, dropped)):
        if n_i <= 0:
            steps.append(
                {
                    "step_index": i,
                    "entered": n_i,
                    "dropped": d_i,
                    "hazard": 0.0,
                    "survival": round(survival, 4),
                    "survival_ci_low": round(survival, 4),
                    "survival_ci_high": round(survival, 4),
                }
            )
            continue
        d_i = min(d_i, n_i)
        hazard = d_i / n_i
        survival *= 1.0 - hazard
        if n_i - d_i > 0:
            greenwood_sum += d_i / (n_i * (n_i - d_i))
        se = survival * math.sqrt(greenwood_sum)
        steps.append(
            {
                "step_index": i,
                "entered": n_i,
                "dropped": d_i,
                "hazard": round(hazard, 4),
                "survival": round(survival, 4),
                "survival_ci_low": round(max(0.0, survival - 1.96 * se), 4),
                "survival_ci_high": round(min(1.0, survival + 1.96 * se), 4),
            }
        )
        if hazard > worst_hazard:
            worst_hazard, worst_idx = hazard, i

    return {
        "steps": steps,
        "overall_survival": round(survival, 4),
        "worst_step_index": worst_idx,
        "worst_step_hazard": round(worst_hazard, 4) if worst_idx is not None else None,
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


def kmeans_auto(
    points: list[Sequence[float]],
    k_range: Sequence[int] = (2, 3, 4),
    min_silhouette: float = 0.4,
    min_spread: float = 0.15,
    seed: int = 23,
    restarts: int = 3,
) -> Optional[tuple[list[int], int, float]]:
    """k-means with k CHOSEN BY SILHOUETTE — segment discovery, not imposition.

    For each candidate k we run several k-means++ restarts (keeping the lowest
    inertia), then score the best run's partition by mean silhouette. The k with
    the highest silhouette wins. Two honesty gates then apply:

    · silhouette ≥ `min_silhouette` — statistical separation. Below it we
      return None: "no real segments" beats inventing structure.
    · max pairwise centroid distance ≥ `min_spread` — PRACTICAL separation.
      Silhouette is scale-invariant, so coincident duplicate points (LLM
      scores are often quantised to 0.1 steps) can cluster "perfectly" at a
      scale nobody cares about. This is the clustering analogue of a ROPE:
      segments must differ by an amount that would change a decision.
    """
    n = len(points)
    if n < 6:
        return None
    best: Optional[tuple[list[int], int, float]] = None
    for k in k_range:
        if k >= n:
            continue
        run_best: Optional[tuple[list[int], float]] = None
        for r in range(restarts):
            labels, inertia = _kmeans_once(points, k, seed=seed + 97 * k + r)
            if run_best is None or inertia < run_best[1]:
                run_best = (labels, inertia)
        if run_best is None or len(set(run_best[0])) < 2:
            continue
        score = _euclid_silhouette(points, run_best[0])
        if best is None or score > best[2]:
            best = (run_best[0], len(set(run_best[0])), score)
    if best is None or best[2] < min_silhouette:
        return None

    # practical-significance gate on centroid spread
    labels = best[0]
    dims = len(points[0])
    centroids = []
    for c in sorted(set(labels)):
        members = [points[i] for i in range(n) if labels[i] == c]
        centroids.append([mean([m[d] for m in members]) for d in range(dims)])
    spread = max(
        _euclid(centroids[i], centroids[j])
        for i in range(len(centroids))
        for j in range(i + 1, len(centroids))
    )
    if spread < min_spread:
        return None
    return best


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
