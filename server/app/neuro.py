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
- Change-point detection (binary segmentation, BIC-penalised): finds the exact
  beats where audience attention structurally shifts — the drop-off moments.
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

from .analytics import bootstrap_ci
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


def change_points(
    xs: Sequence[float],
    min_size: int = 2,
    n_obs: int = 1,
    min_effect: float = 0.01,
) -> list[int]:
    """BIC-penalised binary segmentation: split where the residual sum of
    squares drops enough to justify the extra regime. Returns sorted indices of
    the first sample of each new regime.

    `n_obs` is the number of INDEPENDENT observations behind each point of
    `xs` — for the beat curves this runs on, the twin count. It is not
    decoration. BIC's penalty is k·ln(N) with N the number of observations the
    likelihood was computed from, and the whole consistency argument for BIC
    depends on N being the real sample size. Running the criterion on a curve of
    m beat-*means* while the data underneath is m·n_twins twin-beat ratings
    charges ln(m) where it owes ln(m·n_twins).

    The consequence is not a rounding error, because the likelihood-ratio term
    m·ln(RSS0/RSS1) is scale-free. Averaging shrinks the residual noise of the
    curve by √n_twins but leaves the ratio untouched, so as the twin count rises
    *every* systematic wobble — however practically meaningless — produces an
    arbitrarily large ratio and clears a penalty that never moved. Measured on
    pure-noise mean curves (8 beats, 20 twins, 200 trials) the old criterion
    declared a structural break in **36.5%** of them; with the penalty charged
    at the effective sample size it is 7.0%.

    Two further corrections:

    **`best_cost <= 1e-12` auto-ACCEPTED.** A split that drives the residual to
    zero short-circuited the test entirely, on the reasoning that a perfect fit
    must be real. It is the opposite: a perfect fit is what a *degenerate* split
    looks like. Verified — `change_points([0.5, 0.5, 0.500001, 0.500001],
    min_size=1)` returned `[2]`, announcing an attention collapse on a shift of
    one part in five hundred thousand. The residual is now floored so a perfect
    split still has to clear the penalty like everything else.

    **A practical-significance floor.** Because the criterion is scale-free it
    cannot, on its own, distinguish a real structural break from a real but
    negligible one. `min_effect` is a ROPE on the difference between the two
    segment means, in the units of the series (attention curves are 0-1, so the
    0.01 default is one point on a 0-100 scale). This is the same gate
    `kmeans_auto` applies to centroid spread and for the same reason.
    """
    n = len(xs)
    if n < 2 * min_size + 1:
        return []
    n_obs = max(1, int(n_obs))
    out: list[int] = []

    def recurse(lo: int, hi: int) -> None:
        seg = xs[lo:hi]
        m = len(seg)
        if m < 2 * min_size + 1:
            return
        base = _rss(seg)
        if base <= 1e-12:  # a perfectly flat parent has nothing to split
            return
        best_k, best_cost = None, base
        # `m - min_size` IS an admissible split — it leaves a right segment of
        # exactly min_size, the same size the left segment is allowed. The
        # exclusive bound silently required the right segment to hold at least
        # min_size + 1, so the last beat could never end a segment: a collapse
        # in the *ending* returned [] while the identical collapse in the
        # opening returned [1]. For content analysis that is the worst possible
        # asymmetry — peak-end says the ending is the part that gets
        # remembered, and peak_end_memory reads the same curve.
        for k in range(min_size, m - min_size + 1):
            cost = _rss(seg[:k]) + _rss(seg[k:])
            if cost < best_cost:
                best_k, best_cost = k, cost
        if best_k is None:
            return
        left, right = seg[:best_k], seg[best_k:]
        if abs(sum(left) / len(left) - sum(right) / len(right)) < min_effect:
            return
        # ΔBIC: m·ln(RSS0/RSS1) must beat 2 extra params (the second mean and
        # the split location) · ln(effective N). The floor on best_cost keeps a
        # zero-residual split finite instead of letting it bypass the test.
        statistic = m * math.log(base / max(best_cost, 1e-300))
        penalty = 2.0 * math.log(m * n_obs)
        if statistic > penalty:
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
    # mean curve + CI per dimension per beat
    curves: dict[str, list[float]] = {}
    cis: dict[str, list[list[float]]] = {}
    for dim in DIMS:
        mean_curve, ci_curve = [], []
        for i in range(n_seg):
            vals = [getattr(r.appraisals[i], dim) for r in responses]
            mean_curve.append(round(sum(vals) / len(vals), 4))
            ci = bootstrap_ci(vals)
            ci_curve.append([round(ci[0], 4), round(ci[1], 4)] if ci else None)
        curves[dim] = mean_curve
        cis[dim] = ci_curve

    attention_curves = [[a.attention for a in r.appraisals] for r in responses]
    isc = intersubject_correlation(attention_curves)
    # Beat curves are short (3-8 points) — allow single-beat regimes. n_obs is
    # the twin count: `curves["attention"]` is a mean across twins, so the
    # likelihood behind each point rests on len(responses) ratings, not one.
    # Without it the BIC penalty is charged at the wrong sample size and the
    # segmenter fires on averaging noise (36.5% of pure-noise curves).
    cps = (
        change_points(curves["attention"], min_size=1, n_obs=len(responses))
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
        "curve_cis": cis,
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
        yield {"type": "stage", "stage": "changepoints", "detail": "BIC-penalised change-point detection"}
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
