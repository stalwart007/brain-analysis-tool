"""Neuro-impact content studies: how a piece of content moves an audience,
beat by beat, mapped onto functional cognitive systems.

The science this stands on (and the honest limits of each):

- Appraisal theory (Scherer's Component Process Model): each twin rates every
  content beat on appraisal dimensions — attention, valence, arousal, novelty,
  goal relevance, social resonance, threat, reward anticipation, effort.
- Inter-subject correlation (ISC, Hasson et al., "neurocinematics"): when a
  film grips an audience, their brain responses synchronise. We compute the
  behavioral analogue — correlation of attention trajectories across twins.
  High ISC ⇒ the content controls attention; low ISC ⇒ minds wander apart.
- Change-point detection (binary segmentation, permutation-calibrated against
  a null built by reshuffling each twin's own beats): finds the exact beats
  where audience attention structurally shifts — the drop-off moments.
- Peak-end rule (Kahneman): what people *remember* of an experience is
  dominated by its emotional peak and its ending, plus serial-position
  (primacy/recency) effects on recall.
- The "functional systems" map projects these measurements onto seven named
  cognitive systems (salience, reward, threat, memory, social, semantic,
  executive). These are MODEL-DERIVED ACTIVATION SCORES for visualization —
  explicitly not neuroimaging and labeled so everywhere.

All math is stdlib-only, deterministic, and unit-tested with known answers.
"""

from __future__ import annotations

import asyncio
import math
import random
from typing import AsyncIterator, Optional, Sequence

import openai

from .analytics import bootstrap_ci, simultaneous_band
from .modality import UnsupportedAsset, extract_beats
from .config import LOAD_TO_TEMPERATURE, SWARM_CONCURRENCY, TWIN_MODEL
from .oai import Refusal, async_client, parse_completion, response_format_for
from .schemas import (
    ContentAsset,
    ContentSegments,
    ContentStudyRequest,
    PersonaSeed,
    TwinContentResponse,
)
from .studies import _clamp01, _persona_system

# ─────────────────────────────── pure math ───────────────────────────────


def pearson(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    n = min(len(a), len(b))
    if n < 3:
        return None
    ma, mb = sum(a[:n]) / n, sum(b[:n]) / n
    va = sum((x - ma) ** 2 for x in a[:n])
    vb = sum((x - mb) ** 2 for x in b[:n])
    if va <= 1e-12 or vb <= 1e-12:
        return None
    cov = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    return cov / math.sqrt(va * vb)


def _zscore(xs: Sequence[float]) -> Optional[list[float]]:
    """Unit-variance centring. None when the curve is flat (correlation with a
    constant is undefined, not zero)."""
    n = len(xs)
    m = sum(xs) / n
    ss = sum((x - m) ** 2 for x in xs)
    if ss <= 1e-12:
        return None
    sd = math.sqrt(ss)
    return [(x - m) / sd for x in xs]


# artanh saturates at ±1; real curves hit r = 1.0 exactly whenever two twins
# rate identically, which is common with quantised LLM scores.
_R_CLAMP = 1.0 - 1e-9


def _fisher_z(r: float) -> float:
    r = max(-_R_CLAMP, min(_R_CLAMP, r))
    return 0.5 * math.log((1.0 + r) / (1.0 - r))


def _mean_pair_z(zcurves: list[list[float]]) -> tuple[Optional[float], int]:
    """Mean Fisher-z over all twin pairs. Inputs are pre-z-scored, so each
    pairwise Pearson r is just the dot product."""
    total, n_pairs = 0.0, 0
    for i in range(len(zcurves)):
        for j in range(i + 1, len(zcurves)):
            total += _fisher_z(sum(a * b for a, b in zip(zcurves[i], zcurves[j])))
            n_pairs += 1
    return (total / n_pairs, n_pairs) if n_pairs else (None, 0)


def intersubject_correlation(
    curves: list[Sequence[float]],
    n_permutations: int = 500,
    n_bootstrap: int = 500,
    alpha: float = 0.05,
    seed: int = 17,
) -> Optional[dict]:
    """Behavioral ISC: pairwise synchrony of attention trajectories, with a
    permutation null and a bootstrap interval, plus leave-one-out per-twin
    synchrony (who moves with the crowd).

    Three things the previous version got wrong, all of which inflated or
    over-claimed the headline:

    **Correlations were averaged raw.** r is not additive — its sampling
    distribution is skewed and variance-unstable near ±1, which is exactly
    where a gripping piece of content puts it. The Hasson ISC literature this
    docstring cites averages in Fisher-z (artanh) space and back-transforms,
    and the difference is not cosmetic: a pair set of five twins at r = 0.95
    and three at r = 0.10 averages to 0.631 raw but 0.828 in z — the arithmetic
    mean is dragged toward the sparse dissenters by a metric that does not
    respect the geometry of a correlation.

    **There was no null.** `overall` was reported bare, so nothing distinguished
    "this audience genuinely moves together" from "eight twins, four beats, and
    the arithmetic of small samples". Any measure of synchrony needs a
    distribution under desynchrony to be read against. The null here is the
    standard one for ISC: shuffle the beat order **independently within each
    curve**, which preserves every twin's marginal appraisal distribution and
    destroys only the shared temporal structure that ISC is claiming to
    measure. p is the add-one estimator (1 + #{null ≥ observed}) / (1 + P), so
    it is never reported as exactly zero at finite P.

    **`grip` was three hard-coded cuts on a point estimate.** 0.5 / 0.25 on a
    number with no interval attached converts sampling noise into a verdict —
    with eight uncorrelated twins the old thresholds are one lucky draw from
    calling a random audience "engaged". `grip` is now gated on the permutation
    test first; the 0.5 cut only chooses *between* "locked-in" and "engaged"
    once synchrony is established. A practical floor is kept as well, because
    significance is not effect size: at large twin counts a genuinely
    negligible r ≈ 0.05 becomes detectable, and calling that "engaged" would be
    the same category error in the other direction.

    The bootstrap resamples **twins**, not pairs: pairs share twins and are not
    independent, so a percentile interval over the pair list would be
    anticonservative by construction.
    """
    curves = [c for c in curves if len(c) >= 3]
    if len(curves) < 3:
        return None
    n_seg = min(len(c) for c in curves)
    trimmed = [list(c[:n_seg]) for c in curves]
    zc_all = [_zscore(c) for c in trimmed]
    zcurves = [z for z in zc_all if z is not None]
    if len(zcurves) < 3:
        return None

    mean_z, n_pairs = _mean_pair_z(zcurves)
    if mean_z is None:
        return None
    overall = math.tanh(mean_z)

    # leave-one-out synchrony against the mean of the other twins
    loo = []
    for i, c in enumerate(trimmed):
        others = [trimmed[j] for j in range(len(trimmed)) if j != i]
        mean_other = [sum(o[k] for o in others) / len(others) for k in range(n_seg)]
        r = pearson(c, mean_other)
        loo.append(round(r, 4) if r is not None else None)

    rng = random.Random(seed)

    # ── permutation null: independent beat shuffles, shared structure gone ──
    n_ge = 0
    null_z: list[float] = []
    for _ in range(max(1, n_permutations)):
        shuffled = []
        for z in zcurves:
            cp = list(z)
            rng.shuffle(cp)
            shuffled.append(cp)
        nz, _ = _mean_pair_z(shuffled)
        null_z.append(nz)
        if nz is not None and nz >= mean_z:
            n_ge += 1
    p_value = (1.0 + n_ge) / (1.0 + max(1, n_permutations))

    # ── subject-level bootstrap on the Fisher-z mean ────────────────────────
    boot: list[float] = []
    k = len(zcurves)
    for _ in range(max(1, n_bootstrap)):
        idx = [rng.randrange(k) for _ in range(k)]
        total, cnt = 0.0, 0
        for a in range(k):
            for b in range(a + 1, k):
                # a duplicated twin paired with itself is r = 1 by construction
                # and carries no information; drop those pairs.
                if idx[a] == idx[b]:
                    continue
                total += _fisher_z(
                    sum(x * y for x, y in zip(zcurves[idx[a]], zcurves[idx[b]]))
                )
                cnt += 1
        if cnt:
            boot.append(total / cnt)
    ci: Optional[list[float]] = None
    if len(boot) >= 20:
        boot.sort()
        lo_i = max(0, int((alpha / 2) * len(boot)))
        hi_i = min(len(boot) - 1, int((1 - alpha / 2) * len(boot)))
        ci = [round(math.tanh(boot[lo_i]), 4), round(math.tanh(boot[hi_i]), 4)]

    significant = p_value < alpha
    # ROPE on the correlation itself, matching the practical-significance gates
    # used elsewhere in the codebase (bayesian_ab's rope, kmeans_auto's spread).
    meaningful = overall > 0.10
    if not (significant and meaningful):
        grip = "drifting"
    elif overall > 0.5:
        grip = "locked-in"
    else:
        grip = "engaged"

    return {
        "overall": round(overall, 4),
        "mean_fisher_z": round(mean_z, 4),
        "ci": ci,
        "p_value": round(p_value, 4),
        "significant": significant,
        "n_pairs": n_pairs,
        "n_twins": k,
        "n_permutations": max(1, n_permutations),
        "null_mean_r": round(
            math.tanh(sum(z for z in null_z if z is not None) / max(1, len(null_z))), 4
        ),
        # The null itself, not just its mean.
        #
        # Five hundred permutations are computed to produce `p_value` and then
        # collapsed to one number, which is the one summary that cannot be
        # argued with and also the one nobody believes. Shown as a distribution
        # with the observed statistic standing next to it, "these twins agree
        # more than chance" stops being a claim and becomes a picture: here is
        # what agreement looks like when the beats are shuffled, and here is
        # where this content landed.
        #
        # Back-transformed to r rather than shipped as Fisher z, because r is
        # the axis the observed value is drawn on and asking the client to
        # tanh() a null is asking it to re-implement the estimator.
        "null_r_distribution": [
            round(math.tanh(z), 4) for z in null_z if z is not None
        ],
        "leave_one_out": loo,
        "grip": grip,
        "method": (
            "inter-subject correlation of attention trajectories (Hasson et al.), "
            "Fisher-z averaged, beat-shuffle permutation null, twin-level bootstrap CI"
        ),
    }


def _rss(xs: Sequence[float]) -> float:
    if not xs:
        return 0.0
    m = sum(xs) / len(xs)
    return sum((x - m) ** 2 for x in xs)


# Work cap for the twin-level permutation null, in (draws × beats × twins).
# Sized from measurement: at 120k a 10-beat curve that splits four ways ran
# 21 ms at 40 twins and 33 ms at 200 — inside the 50 ms streaming budget but
# not comfortably. At 60k the same worst cases are 12 ms / 25 ms while every
# curve at 8-20 twins still gets the full 300 draws. See change_points.
_NULL_WORK_CAP = 60_000


def _best_split(
    seg: Sequence[float], min_size: int
) -> tuple[Optional[int], float, float]:
    """Best single split of `seg`: (k, RSS_left + RSS_right, total RSS), the
    cost minimised over admissible k, via prefix sums so scoring one
    permutation draw is O(m). The total RSS comes back with it because the
    caller's statistic is the between-segment SS, total − cost, and a
    twin-level surrogate is a different curve from the observed one — it has
    its own total, and must be scored against it.

    `m - min_size` IS an admissible split — it leaves a right segment of
    exactly min_size, the same size the left segment is allowed. An exclusive
    bound here once silently required the right segment to hold at least
    min_size + 1, so the last beat could never end a segment: a collapse in
    the *ending* returned [] while the identical collapse in the opening
    returned [1]. For content analysis that is the worst possible asymmetry —
    peak-end says the ending is the part that gets remembered, and
    peak_end_memory reads the same curve.
    """
    m = len(seg)
    ps, pss = [0.0], [0.0]
    for x in seg:
        ps.append(ps[-1] + x)
        pss.append(pss[-1] + x * x)
    total = max(pss[m] - ps[m] * ps[m] / m, 0.0)
    best_k: Optional[int] = None
    best_cost = math.inf
    for k in range(min_size, m - min_size + 1):
        sl, sl2 = ps[k], pss[k]
        sr, sr2 = ps[m] - ps[k], pss[m] - pss[k]
        cost = (sl2 - sl * sl / k) + (sr2 - sr * sr / (m - k))
        if cost < best_cost:
            best_k, best_cost = k, cost
    return best_k, max(best_cost, 0.0), total


def change_points(
    xs: Sequence[float],
    min_size: int = 2,
    n_obs: int = 1,
    min_effect: float = 0.01,
    alpha: float = 0.05,
    n_permutations: int = 300,
    observations: Optional[Sequence[Sequence[float]]] = None,
) -> list[int]:
    """Permutation-calibrated binary segmentation: split where the best split
    buys more than reshuffling the same data ever does. Returns sorted indices
    of the first sample of each new regime.

    **Why the BIC criterion this replaces was uncalibrated.** Its statistic,
    max_k [m·ln(RSS0/RSS_k)], is a MAXIMUM over every admissible split, so it
    is not χ²(2)-ish; its null distribution depends hard on the segment
    length; and recursive segmentation added uncorrected multiplicity on top.
    A fixed penalty — even one charged at the effective sample size m·n_obs —
    cannot hold a nominal level across those regimes. Measured on pure-noise
    beat-mean curves as the product makes them (per twin per beat N(0.5,
    0.15) clipped to [0,1], averaged across twins; min_size=1; 2000
    trials/cell; nominal 5%), the BIC version's false-positive rate was a
    function of both counts: at m=8 beats, **19.8% / 9.5% / 5.4% / 3.4%** for
    3 / 8 / 20 / 40 twins — and on the short curves the product actually
    produces, far worse: m=4: **43.6% / 28.8% / 18.6% / 12.3%**; m=3:
    **63.3% / 43.6% / 33.7% / 24.9%**. (The 7.0% this docstring used to claim
    was one cell of that grid — 8 beats at 20 twins — not a level.)

    **The fix.** The statistic is now the between-segment sum of squares,
    S = RSS0 − min_k RSS_k: how much of the curve's variation the best split
    explains, in the units of the series. It is compared against the
    distribution of the SAME max-over-k statistic on `n_permutations` seeded
    reshuffles of the segment, and a split is accepted only if
    p = (1 + #{S_perm ≥ S}) / (draws + 1) ≤ `alpha` AND it clears the
    `min_effect` ROPE. Two nulls, chosen by what the caller can supply:

    - **`observations` given** — beats × observations-per-beat; for the
      product path, each beat's per-twin ratings. Each twin's ratings are
      reshuffled across the segment's beats INDEPENDENTLY PER TWIN and the
      beat-mean curve recomputed. That is a stratified test of "no beat
      structure shared across twins", and it is what the old `n_obs`
      argument was reaching for: the fact that a 4-point curve rests on 4·20
      ratings enters through how tight the recomputed means are, not through
      a ln(m·n_obs) penalty. Calibrated at every beat count, including m=3-4.
    - **curve only** — the beat means themselves are reshuffled. Valid but
      structurally conservative, and not by a little: the max statistic is
      invariant to permutation WITHIN each candidate segment, so at least
      2·k!(m−k)!/m! of all arrangements tie with the observed one and p can
      never fall below 2/C(m,k). Measured false positives 0.0% at m ≤ 5,
      0.1% at m=6, 2.1-2.5% at m=8 and 3.0-4.2% at m=10 — a test that mostly
      cannot fire on a short curve. Pass the twin ratings if you have them.

    `n_obs` is accepted for signature compatibility and no longer sets any
    threshold: the null is conditional on the observed data, so it
    self-calibrates to whatever noise the twin count left on the curve.

    **Measured after the fix**, twin-level null, min_size=1, 2000
    trials/cell, at 3 / 8 / 20 / 40 twins:

        m=3   0.3%  4.9%  4.5%  5.4%      m=8    5.1% 4.9% 4.9% 4.9%
        m=4   3.5%  5.4%  5.2%  4.5%      m=8q   4.8% 5.2% 3.7% 4.0%
        m=5   4.1%  5.2%  6.5%  5.1%      m=10   5.2% 6.0% 5.1% 5.4%
        m=6   5.4%  5.2%  4.6%  4.2%      m=10q  4.3% 4.5% 4.3% 4.3%

    (q = the 0.1-quantised variant, since twin scores are quantised; the
    default min_size=2 measures the same, 4.6-6.1% at m=8-10.) The level is
    flat in both counts instead of ranging 3% to 63%. The one conservative
    corner is 3 beats × 3 twins, where the whole randomization support is
    (3!)³ = 216 arrangements.

    **Power**, true step 0.5 → 0.62 at beat 5 of 8: **70.2% at 8 twins,
    98.8% at 20, 100% at 40**, against 63.0% / 80.6% for the uncalibrated
    BIC. The calibration is not paid for in power here — it is *gained*,
    because the twin-level null replaces a scale-free ratio with a statistic
    that uses the observation scale. (On the curve-only null the same step
    scores 32.1% / 65.5%, the price of having no twin data.)

    **Runtime** on 10-beat curves, the streaming path's budget being ~50 ms:
    0.13 ms curve-level; twin-level 0.10 / 0.22 / 0.22 / 0.69 ms at 8 / 20 /
    40 / 200 twins on a null curve (the null loop stops as soon as p > alpha
    is certain), 3.8-13.2 ms when a split IS accepted and every draw must
    run, and 9.7 ms (40 twins) / 34.0 ms (200 twins) on a pathological
    4-regime curve that splits at every level.

    Deterministic: the permutation RNG is seeded from the segment's own
    values, so repeated calls on the same curve agree exactly.

    Two retained corrections from the previous version:

    **A zero-residual split must not short-circuit the test.** A split that
    drives the residual to zero is what a *degenerate* split looks like, not
    proof of structure — `change_points([0.5, 0.5, 0.500001, 0.500001],
    min_size=1)` once returned `[2]`, announcing an attention collapse on a
    shift of one part in five hundred thousand. The residual is floored so a
    perfect split still has to beat the null like everything else (and the
    ROPE below rejects it anyway).

    **A practical-significance floor.** A calibrated test still cannot, on
    its own, distinguish a real structural break from a real but negligible
    one. `min_effect` is a ROPE on the difference between the two segment
    means, in the units of the series (attention curves are 0-1, so the 0.01
    default is one point on a 0-100 scale). This is the same gate
    `kmeans_auto` applies to centroid spread and for the same reason.
    """
    n = len(xs)
    if n < 2 * min_size + 1:
        return []
    obs_ok = (
        observations is not None
        and len(observations) == n
        and all(len(col) == len(observations[0]) for col in observations)
        and len(observations[0]) >= 2
    )
    n_perm = max(1, int(n_permutations))
    if obs_ok:
        # The twin-level null costs O(draws · beats · twins), and a study may
        # carry 200+ twins. Cap the work per segment so the streaming path
        # keeps its budget; 99 draws is the floor because it is the coarsest
        # grid on which p = (1 + exceed)/(draws + 1) can still certify 0.05.
        # A permutation test is exact at any draw count — the cap costs
        # p-resolution, never level.
        per_draw = n * len(observations[0])
        n_perm = max(99, min(n_perm, _NULL_WORK_CAP // max(1, per_draw)))
    # Rejection needs (1 + exceed) / (n_perm + 1) <= alpha. Once `exceed`
    # passes this bound the split can no longer be accepted, so the null loop
    # stops early — the decision is identical, and a clearly-null segment
    # settles in ~exceed_max/p_true draws instead of n_perm.
    exceed_max = int(alpha * (n_perm + 1) - 1 + 1e-9)
    if exceed_max < 0:
        return []  # alpha below the resolution n_permutations can certify
    out: list[int] = []

    def recurse(lo: int, hi: int) -> None:
        seg = xs[lo:hi]
        m = len(seg)
        if m < 2 * min_size + 1:
            return
        base = _rss(seg)
        if base <= 1e-12:  # a perfectly flat parent has nothing to split
            return
        best_k, best_cost, _ = _best_split(seg, min_size)
        if best_k is None:
            return
        left, right = seg[:best_k], seg[best_k:]
        if abs(sum(left) / len(left) - sum(right) / len(right)) < min_effect:
            return
        # Between-segment sum of squares — the size of the step the best split
        # buys, in the units of the series. NOT the scale-free ratio
        # m·ln(RSS0/RSS_k) the BIC criterion used: under the twin-level null
        # every surrogate has its own RSS0, and dividing by it discards
        # exactly the evidence the twin data carries — that the observed mean
        # curve is far tighter within its segments than reshuffling can make
        # it. Measured cost of the ratio form on the twin-level null: power on
        # the 8-beat step fell to 38.8% at 8 twins (70.2% here) and 79.1% at
        # 20 (98.8% here), and the 4-beat 0.95 → 0.25 collapse in
        # test_aggregate_content_study_end_to_end went UNDETECTED with nine
        # twins that all saw it. Under the curve-level null the two forms are
        # the same test — reshuffling the means leaves RSS0 fixed, so ranking
        # by RSS0 − cost and by RSS0/cost coincide.
        observed = base - best_cost
        # Data-derived seed: identical curves give identical splits on every
        # call, which the streaming path and the tests both rely on.
        rng = random.Random(
            "change_points|" + ",".join(f"{v:.9g}" for v in seg)
        )
        if obs_ok:
            # One list per twin, restricted to this segment's beats.
            n_twins = len(observations[0])
            twin_rows = [
                [observations[lo + j][t] for j in range(m)]
                for t in range(n_twins)
            ]
        else:
            vals = list(seg)
        draw = rng.random
        exceed = 0
        for _ in range(n_perm):
            if obs_ok:
                # Sorting on i.i.d. uniform keys is a uniform permutation, and
                # measured 2x faster than random.shuffle here because the keys
                # come from one C-level call each and the sort itself is C.
                for row in twin_rows:
                    row.sort(key=lambda _v: draw())
                surro = [sum(col) / n_twins for col in zip(*twin_rows)]
            else:
                rng.shuffle(vals)
                surro = vals
            _, cost, total = _best_split(surro, min_size)
            # Scored against the surrogate's own RSS0 — see _best_split.
            if total - cost >= observed - 1e-12:
                exceed += 1
                if exceed > exceed_max:
                    return  # p > alpha is already certain
        # Here (1 + exceed) / (n_perm + 1) <= alpha: accept the split.
        split = lo + best_k
        recurse(lo, split)
        out.append(split)
        recurse(split, hi)

    recurse(0, n)
    return sorted(out)


def peak_end_memory(
    arousal: Sequence[float], valence: Sequence[float]
) -> Optional[dict]:
    """Kahneman peak-end rule + serial-position recall weights.

    remembered_affect = mean(valence at arousal peak, final valence);
    recall_probability per beat = softmax of arousal z-score + primacy/recency
    bonuses — the beats people will actually retain."""
    n = min(len(arousal), len(valence))
    if n < 3:
        return None
    peak_i = max(range(n), key=lambda i: arousal[i])
    remembered = (valence[peak_i] + valence[n - 1]) / 2.0
    mu = sum(arousal[:n]) / n
    sd = math.sqrt(sum((a - mu) ** 2 for a in arousal[:n]) / n) or 1e-9
    scores = []
    for i in range(n):
        s = (arousal[i] - mu) / sd
        if i == 0:
            s += 0.7  # primacy
        if i == n - 1:
            s += 0.9  # recency
        scores.append(s)
    exps = [math.exp(s) for s in scores]
    tot = sum(exps)
    probs = [e / tot for e in exps]

    # `recall_probability` is a softmax and therefore sums to 1 by
    # construction — it is a *relative allocation of attention across beats*,
    # not a probability that anything is recalled. Anything derived from it
    # must be invariant to the number of beats, or it measures the segmenter
    # rather than the content. Two such quantities:
    #
    #   encoding_strength    peak arousal. High-arousal moments are
    #                        consolidated more strongly (amygdala modulation of
    #                        hippocampal encoding) — the direct, citable
    #                        mechanism, already in [0,1], independent of n.
    #   recall_concentration how far the allocation departs from uniform: 0 = a
    #                        flat piece with no standout beat, 1 = one beat
    #                        takes everything. Normalised by (1 − 1/n) so it,
    #                        too, does not move with n.
    peak_arousal = max(arousal[:n])
    uniform = 1.0 / n
    concentration = (max(probs) - uniform) / (1.0 - uniform) if n > 1 else 0.0

    return {
        "remembered_affect": round(remembered, 4),
        "peak_index": peak_i,
        "end_valence": round(valence[n - 1], 4),
        "recall_probability": [round(p, 4) for p in probs],
        "encoding_strength": round(max(0.0, min(1.0, peak_arousal)), 4),
        "recall_concentration": round(max(0.0, min(1.0, concentration)), 4),
        "model": "peak-end rule + serial-position effects (Kahneman)",
    }


def peak_end_per_twin(
    responses: "list", dims: "tuple[str, str]" = ("arousal", "valence")
) -> Optional[dict]:
    """Peak-end computed WITHIN each twin, then aggregated over the results.

    The peak-end rule is a within-subject phenomenon: an individual remembers
    the moment THEY peaked and the moment it ended for THEM. Running it on the
    cross-twin mean curve is an ecological fallacy, and not a harmless one —
    with two audience camps peaking at different beats the mean peaks at a beat
    NO twin peaked at, and `remembered_affect` is then the valence of a beat
    nobody had a peak experience of. Jensen also bites: max(mean arousal) is
    <= mean(max arousal), so encoding_strength was systematically understated.

    The peak-index DISTRIBUTION that falls out of doing this correctly is worth
    as much as the correction — it is a direct polarisation signal, showing
    whether the audience has one peak or several.
    """
    arousal_dim, valence_dim = dims
    per: list[dict] = []
    for r in responses:
        a = [getattr(x, arousal_dim) for x in r.appraisals]
        v = [getattr(x, valence_dim) for x in r.appraisals]
        m = peak_end_memory(a, v)
        if m:
            per.append(m)
    if not per:
        return None

    n_beats = len(responses[0].appraisals)
    counts = [0] * n_beats
    for m in per:
        counts[m["peak_index"]] += 1
    modal = max(range(n_beats), key=lambda i: counts[i])
    remembered = [m["remembered_affect"] for m in per]
    strength = [m["encoding_strength"] for m in per]

    # Agreement on WHERE the peak was. 1.0 = every twin peaked on the same
    # beat; near 1/n_beats = no shared peak at all, which is the case the mean
    # curve silently hid.
    peak_agreement = counts[modal] / len(per)

    return {
        "peak_index": modal,
        "peak_index_distribution": counts,
        "peak_agreement": round(peak_agreement, 4),
        "remembered_affect": round(sum(remembered) / len(remembered), 4),
        "remembered_affect_ci": bootstrap_ci(remembered),
        "encoding_strength": round(sum(strength) / len(strength), 4),
        "encoding_strength_ci": bootstrap_ci(strength),
        "end_valence": round(
            sum(m["end_valence"] for m in per) / len(per), 4
        ),
        "recall_concentration": round(
            sum(m["recall_concentration"] for m in per) / len(per), 4
        ),
        "n_twins": len(per),
        "model": (
            "peak-end rule computed per twin and aggregated (Kahneman); the "
            "peak-index distribution is the polarisation signal"
        ),
    }


def retention_hazard(responses: "list", threshold: float = 0.25) -> Optional[dict]:
    """Discrete-time abandonment hazard across beats.

    The commercially load-bearing question for anything with a running order:
    where do people stop. A twin is treated as having abandoned at the FIRST
    beat whose attention falls below `threshold`, and as retained thereafter
    only if it never does — abandonment is absorbing, because someone who has
    stopped watching does not come back for beat 5 and rate it.

    Reported as a hazard per beat plus the surviving fraction, with the risk
    set shrinking as twins drop. That is the shape Kaplan-Meier expects, and it
    is why this is not just `mean(attention) < threshold`: a beat that loses
    half of a small remaining audience is a different finding from one that
    loses the same count out of everyone.
    """
    if not responses:
        return None
    n_beats = len(responses[0].appraisals)
    if n_beats < 2:
        return None

    first_drop: list[Optional[int]] = []
    for r in responses:
        idx = next(
            (i for i, a in enumerate(r.appraisals) if a.attention < threshold), None
        )
        first_drop.append(idx)

    at_risk = len(responses)
    survival = 1.0
    steps = []
    for i in range(n_beats):
        dropped = sum(1 for d in first_drop if d == i)
        if at_risk <= 0:
            # The curve keeps its full length so a chart spans the whole
            # content, but the hazard is None rather than 0.0: with an empty
            # risk set the conditional probability of dropping is UNDEFINED,
            # not zero, and a run of zeros at the tail would read as "nobody
            # left here" when it means "nobody was left to leave".
            steps.append(
                {
                    "beat": i,
                    "entered": 0,
                    "dropped": 0,
                    "hazard": None,
                    "survival": round(survival, 4),
                }
            )
            continue
        hazard = dropped / at_risk
        survival *= 1.0 - hazard
        steps.append(
            {
                "beat": i,
                "entered": at_risk,
                "dropped": dropped,
                "hazard": round(hazard, 4),
                "survival": round(survival, 4),
            }
        )
        at_risk -= dropped

    completed = sum(1 for d in first_drop if d is None)
    scored = [s for s in steps if s["hazard"] is not None]
    worst = max(scored, key=lambda s: s["hazard"]) if scored else None
    return {
        "threshold": threshold,
        "steps": steps,
        "completion_rate": round(completed / len(responses), 4),
        # Only meaningful where a beat actually lost someone; an argmax over
        # all-zero hazards would name beat 0 and read as a finding.
        "worst_beat": worst["beat"] if worst and worst["dropped"] > 0 else None,
        "model": "discrete-time hazard; abandonment is absorbing",
    }


def emotional_trajectory(
    valence: Sequence[float], arousal: Sequence[float]
) -> Optional[dict]:
    """The content's path through the valence×arousal circumplex (Russell):
    how far it moves the audience, where it ends up, which quadrants it
    visits. Dynamism = total path length; a flat ad scores ~0."""
    n = min(len(valence), len(arousal))
    if n < 2:
        return None
    path = sum(
        math.hypot(valence[i + 1] - valence[i], arousal[i + 1] - arousal[i])
        for i in range(n - 1)
    )
    quadrant = lambda v, a: (  # noqa: E731
        "energised" if v >= 0 and a >= 0.5
        else "content" if v >= 0
        else "distressed" if a >= 0.5
        else "disengaged"
    )
    occupancy: dict[str, int] = {}
    for i in range(n):
        q = quadrant(valence[i], arousal[i])
        occupancy[q] = occupancy.get(q, 0) + 1
    return {
        "dynamism": round(path, 4),
        "net_valence_shift": round(valence[n - 1] - valence[0], 4),
        "net_arousal_shift": round(arousal[n - 1] - arousal[0], 4),
        "quadrant_occupancy": {k: round(v / n, 3) for k, v in occupancy.items()},
        "model": "circumplex affect trajectory (Russell)",
    }


def functional_systems(dims: dict[str, float], memorability: float) -> dict:
    """Project appraisal means onto seven named cognitive systems (0..1).
    Formulas are documented and fixed — model-derived scores for the brain
    map, NOT neuroimaging."""
    g = lambda k: _clamp01(dims.get(k, 0.0))  # noqa: E731
    vpos = _clamp01((dims.get("valence", 0.0) + 1) / 2)
    systems = {
        "salience": _clamp01(0.6 * g("attention") + 0.4 * g("novelty")),
        "reward": _clamp01(0.6 * g("reward_anticipation") + 0.4 * vpos),
        "threat": _clamp01(0.7 * g("threat") + 0.3 * g("arousal")),
        "memory": _clamp01(memorability),
        "social": g("social_resonance"),
        "semantic": _clamp01(1.0 - 0.6 * g("cognitive_effort") + 0.2 * g("attention") - 0.2),
        "executive": g("cognitive_effort"),
    }
    return {k: round(v, 4) for k, v in systems.items()}


# ───────────────────────── twin orchestration ────────────────────────────

_SEGMENTER_SYSTEM = """Split the given content into its natural narrative
beats (scenes / paragraphs / moments), in order. 3 to 8 beats. Each beat is a
short label plus a one-line summary of what happens in it. Preserve order.
Return only the beats."""

_CONTENT_PREAMBLE = """You are one synthetic audience member experiencing a
piece of content beat by beat, in order, without knowing what comes next.
For EVERY beat report your honest momentary reaction on each scale:
attention (0 ignored → 1 riveted), valence (-1 repelled → +1 delighted),
arousal (0 calm → 1 activated), novelty (0 seen it before → 1 surprising),
goal_relevance (0 nothing for me → 1 exactly my problem), social_resonance
(0 no urge to discuss → 1 must tell someone), threat (0 safe → 1 uneasy),
reward_anticipation (0 nothing promised → 1 can't wait), cognitive_effort
(0 effortless → 1 confusing). React as YOUR persona, not an average person.
Give exactly one appraisal object per beat, in beat order."""


async def segment_content(content: str, content_type: str) -> list[str]:
    try:
        completion = await async_client().chat.completions.create(
            model=TWIN_MODEL,
            messages=[
                {"role": "system", "content": _SEGMENTER_SYSTEM},
                {"role": "user", "content": f"Content type: {content_type}\n\n{content}"},
            ],
            temperature=0.2,
            response_format=response_format_for(ContentSegments),
        )
        segs = parse_completion(completion, ContentSegments).segments
        if 2 <= len(segs) <= 10:
            return segs
    except (openai.OpenAIError, Refusal, RuntimeError, ValueError):
        pass
    # fallback: paragraph split
    paras = [p.strip() for p in content.split("\n\n") if p.strip()]
    return paras[:8] if len(paras) >= 2 else [content[:200], content[200:400] or "(end)"]


async def _content_twin(
    persona: PersonaSeed,
    segments: list[str],
    load: str,
    semaphore: asyncio.Semaphore,
) -> Optional[TwinContentResponse]:
    numbered = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(segments))
    messages = [
        {"role": "system", "content": _CONTENT_PREAMBLE},
        _persona_system(persona, load),
        {"role": "user", "content": f"The content, beat by beat:\n{numbered}"},
    ]
    async with semaphore:
        try:
            completion = await async_client().chat.completions.create(
                model=TWIN_MODEL,
                messages=messages,
                temperature=LOAD_TO_TEMPERATURE[load],
                response_format=response_format_for(TwinContentResponse),
            )
            resp = parse_completion(completion, TwinContentResponse)
        except (openai.OpenAIError, Refusal, RuntimeError, ValueError):
            return None
    return resp if len(resp.appraisals) == len(segments) else None


#: What the segmenting stage is actually doing, per modality — the frontend
#: shows this live, and "splitting content into narrative beats" would be a lie
#: for an image.
_INGEST_DETAIL = {
    "image": "reading the visual hierarchy — what the eye lands on, in order",
    "video": "describing keyframes and aligning transcript cues",
    "audio": "grouping transcript cues into beats",
    "page": "walking page sections in DOM order",
    "document": "reading pages in document order",
    "text": "splitting content into narrative beats",
}


DIMS = [
    "attention", "valence", "arousal", "novelty", "goal_relevance",
    "social_resonance", "threat", "reward_anticipation", "cognitive_effort",
]


def aggregate_content_study(
    responses: list[TwinContentResponse],
    segments: list[str],
    load: str,
    sequence: "Optional[object]" = None,
) -> dict:
    """All the analysis layers over the raw appraisal tensor
    (twins × beats × dimensions). Pure — fully unit-testable.

    `sequence` is the BeatSequence the beats came from. It carries the AXIS,
    and the axis decides which statistics are meaningful: peak-end, change
    points as moments, retention and trajectory all presume an ordered
    experience, and the regions of a static image are simultaneous. Anything
    withheld is reported WITH ITS REASON in `withheld`, because an absent
    number and an inapplicable one look identical in a UI and only one of them
    is a finding.

    Omitting `sequence` keeps the old behaviour (everything computed), so
    existing callers and stored runs are unaffected.
    """
    def _supports(stat: str) -> bool:
        return sequence is None or sequence.supports(stat)

    n_seg = len(segments)
    # Mean curve + SIMULTANEOUS band per dimension.
    #
    # These were per-beat percentile intervals from independent `bootstrap_ci`
    # calls, which is wrong twice. Independent resampling at each beat asserts
    # that a twin's beat-3 rating says nothing about its beat-4 rating — false,
    # because twins have baselines — and pointwise 95% intervals drawn across a
    # curve are read as a band while covering the whole curve far less often
    # than they appear to. Measured coverage of an 8-beat curve: 50.3% at 8
    # twins, 62.7% at 20, against the 95% the chart is announcing. With nine
    # dimensions on screen that is dozens of invitations to find the one beat
    # that looks different, and one of them always does.
    #
    # `curve_cis` is now the sup-t band (97-99% measured simultaneous coverage)
    # because that is the one that gets drawn. The pointwise version is kept
    # alongside it, clearly named, for anyone comparing a single beat against a
    # single pre-registered prediction — the one question it is valid for.
    curves: dict[str, list[float]] = {}
    cis: dict[str, list[list[float]] | None] = {}
    pointwise: dict[str, list[list[float]] | None] = {}
    band_meta: dict[str, dict] = {}
    attention_by_beat: list[list[float]] = []
    for dim in DIMS:
        per_beat = [
            [getattr(r.appraisals[i], dim) for r in responses] for i in range(n_seg)
        ]
        if dim == "attention":
            attention_by_beat = per_beat  # the change-point null needs the raw ratings
        curves[dim] = [round(sum(col) / len(col), 4) for col in per_beat]
        band = simultaneous_band(per_beat)
        if band is None:
            # Too few twins to resample a curve. An absent band is honest; a
            # per-beat interval substituted here would be the exact overclaim
            # the band exists to stop.
            cis[dim] = None
            pointwise[dim] = None
            continue
        cis[dim] = band["band"]
        pointwise[dim] = band["pointwise"]
        band_meta[dim] = {
            "critical_value": band["critical_value"],
            "bonferroni_critical_value": band["bonferroni_critical_value"],
            "se": band["se"],
            "degenerate": band["degenerate"],
        }

    attention_curves = [[a.attention for a in r.appraisals] for r in responses]
    isc = intersubject_correlation(attention_curves)
    # Beat curves are short (3-8 points) — allow single-beat regimes. The raw
    # per-beat twin ratings go in as `observations` so the permutation null is
    # built by reshuffling each twin's beats, not by reshuffling the m beat
    # means: m means alone carry a p-value floor of ~2/m! and the max
    # statistic is invariant to permutation within each candidate segment, so
    # the curve-only null cannot reach 5% below m ≈ 7 and is conservative
    # above it (measured 0.0-0.4% at m ≤ 6, 2.1-4.2% at m = 8-10, against a
    # nominal 5%). With the twin ratings the same test holds 4.0-6.2% at every
    # beat and twin count measured. `n_obs` is passed for continuity of the
    # public signature; it no longer sets a threshold.
    cps = (
        change_points(
            curves["attention"],
            min_size=1,
            n_obs=len(responses),
            observations=attention_by_beat,
        )
        if _supports("change_points")
        else None
    )
    # Per twin, then aggregated — see peak_end_per_twin. The mean-curve version
    # is an ecological fallacy that reports the valence of a beat no twin
    # necessarily peaked on.
    memory = peak_end_per_twin(responses) if _supports("peak_end") else None
    trajectory = (
        emotional_trajectory(curves["valence"], curves["arousal"])
        if _supports("trajectory")
        else None
    )
    retention = retention_hazard(responses) if _supports("retention") else None
    overall_dims = {d: sum(curves[d]) / n_seg for d in DIMS}
    # Must not be max(recall_probability): that is a softmax over beats, so it
    # falls as ~1/n and the "memory" bar on the brain map tracked how many
    # beats the segmenter happened to emit. The same content split into 3 vs 8
    # beats moved it roughly 2×. encoding_strength is peak arousal — the actual
    # mechanism, and invariant to segmentation.
    memorability = (
        memory["encoding_strength"] if memory else overall_dims["arousal"]
    )
    systems = functional_systems(overall_dims, memorability)
    # Per-beat memorability must be on the SAME SCALE as the study-level
    # number, or the two "memory" bars on the brain map are not comparable —
    # and they were not. This was `recall_probability[i] * n_seg / 2`, where
    # recall_probability is a softmax that sums to 1 by construction: on any
    # flat arousal curve every beat scored EXACTLY 0.500 regardless of how
    # aroused the audience actually was, for every beat count.
    #
    # Study level is peak arousal; per beat is that beat's arousal. Both are
    # arousal in [0,1], both are invariant to the number of beats, and the
    # max over beats reproduces the study-level figure up to Jensen — which is
    # what "comparable" has to mean.
    per_segment_systems = [
        functional_systems({d: curves[d][i] for d in DIMS}, curves["arousal"][i])
        for i in range(n_seg)
    ]
    # Clamp: TwinContentResponse.would_share carries no numeric bounds in the
    # schema (unlike TwinReaction), so an out-of-range LLM value would otherwise
    # propagate straight into the headline mean. virality.py already clamps the
    # identical field; this makes the two consistent.
    share = [_clamp01(r.would_share) for r in responses]
    share_mean = sum(share) / len(share)

    weakest = min(range(n_seg), key=lambda i: curves["attention"][i])
    # Read the verdict off `grip`, not off a second set of thresholds on the
    # point estimate. Two independent cut-point ladders over the same number
    # could — and did — disagree, so the headline sentence could say attention
    # "holds" while the ISC block underneath reported "drifting".
    _GRIP_VERB = {"locked-in": "locks in", "engaged": "holds", "drifting": "fragments"}
    verdict = (
        f"Attention {_GRIP_VERB[isc['grip']]} "
        f"(ISC {isc['overall']:.2f}, p={isc['p_value']:.3f}); " if isc else ""
    ) + (
        f"memory will keep beat {memory['peak_index'] + 1} and the ending; " if memory else ""
    ) + f"weakest beat is {weakest + 1} ('{segments[weakest][:40]}')."

    return {
        "segments": segments,
        "twin_count": len(responses),
        "cognitive_load": load,
        "curves": curves,
        # Simultaneous — safe to read across the whole curve, which is how a
        # chart is read whatever the caption says.
        "curve_cis": cis,
        "curve_pointwise_cis": pointwise,
        "curve_band_meta": band_meta,
        "band_method": (
            "sup-t simultaneous confidence bands (studentized subject-level "
            "bootstrap): 95% coverage of the ENTIRE curve, not of each beat "
            "separately. Wider than pointwise intervals by design — pointwise "
            "bands over 8 beats were measured to cover the whole curve only "
            "50-63% of the time."
        ),
        "isc": isc,
        "change_points": cps,
        "memory": memory,
        "trajectory": trajectory,
        "retention": retention,
        "axis": getattr(sequence, "axis", None) and sequence.axis.value,
        "beat_source": getattr(sequence, "source", None),
        "beat_notes": getattr(sequence, "notes", None),
        "timestamps_ms": getattr(sequence, "timestamps_ms", None),
        "withheld": (sequence.to_dict()["withheld"] if sequence is not None else {}),
        "systems": systems,
        "per_segment_systems": per_segment_systems,
        "share_intent_mean": round(share_mean, 4),
        "share_intent_ci": bootstrap_ci(share),
        "gists": [r.gist for r in responses][:12],
        "verdict": verdict,
        "disclaimer": (
            "functional-system activations are model-derived scores from "
            "appraisal ratings — a visualization of measured constructs, not "
            "neuroimaging"
        ),
    }


async def stream_content_study(
    personas: list[PersonaSeed], request: ContentStudyRequest
) -> AsyncIterator[dict]:
    # Any modality, not just pasted text. `asset` is the new path; `content`
    # remains for existing callers and is treated as a text asset.
    asset = request.asset or ContentAsset(
        kind="text", text=request.content, content_type=request.content_type
    )
    yield {
        "type": "stage",
        "stage": "segmenting",
        "detail": _INGEST_DETAIL.get(asset.kind, "splitting content into beats"),
    }
    try:
        sequence = await extract_beats(asset)
    except UnsupportedAsset as exc:
        # Refusing beats guessing: a study run on beats that do not represent
        # the content produces confident numbers about something the audience
        # never saw.
        yield {"type": "error", "detail": str(exc)}
        return
    segments = sequence.beats
    roster = personas * request.twins_per_persona
    yield {
        "type": "start",
        "total": len(roster),
        "segments": segments,
        "axis": sequence.axis.value,
        "beat_source": sequence.source,
        "beat_notes": sequence.notes,
    }

    semaphore = asyncio.Semaphore(SWARM_CONCURRENCY)
    tasks = [
        asyncio.create_task(_content_twin(p, segments, request.cognitive_load, semaphore))
        for p in roster
    ]
    responses: list[TwinContentResponse] = []
    for fut in asyncio.as_completed(tasks):
        resp = await fut
        if resp is None:
            yield {"type": "agent_failed"}
            continue
        responses.append(resp)
        yield {
            "type": "agent",
            "attention": [a.attention for a in resp.appraisals],
            "valence": [a.valence for a in resp.appraisals],
            "arousal": [a.arousal for a in resp.appraisals],
            "share": resp.would_share,
        }
    if len(responses) < 3:
        yield {"type": "error", "detail": "Too few twins responded to analyse."}
        return

    yield {"type": "stage", "stage": "isc", "detail": "inter-subject correlation of attention trajectories"}
    if sequence.supports("change_points"):
        yield {
            "type": "stage",
            "stage": "changepoints",
            "detail": "permutation-calibrated change-point detection",
        }
    if sequence.supports("peak_end"):
        yield {"type": "stage", "stage": "memory", "detail": "per-twin peak-end + serial-position recall"}
    if sequence.supports("retention"):
        yield {"type": "stage", "stage": "retention", "detail": "discrete-time abandonment hazard"}
    yield {"type": "stage", "stage": "systems", "detail": "projecting onto functional cognitive systems"}
    result = aggregate_content_study(
        responses, segments, request.cognitive_load, sequence=sequence
    )
    result["content_preview"] = (asset.text or asset.brief or segments[0])[:140]
    result["modality"] = asset.kind
    yield {"type": "done", "result": result}
