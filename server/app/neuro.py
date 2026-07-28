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
from .config import LOAD_TO_TEMPERATURE, SWARM_CONCURRENCY, TWIN_MODEL
from .oai import Refusal, async_client, parse_completion, response_format_for
from .schemas import (
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


DIMS = [
    "attention", "valence", "arousal", "novelty", "goal_relevance",
    "social_resonance", "threat", "reward_anticipation", "cognitive_effort",
]


def aggregate_content_study(
    responses: list[TwinContentResponse], segments: list[str], load: str
) -> dict:
    """All the analysis layers over the raw appraisal tensor
    (twins × beats × dimensions). Pure — fully unit-testable."""
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
    cps = change_points(curves["attention"], min_size=1, n_obs=len(responses))
    memory = peak_end_memory(curves["arousal"], curves["valence"])
    trajectory = emotional_trajectory(curves["valence"], curves["arousal"])
    overall_dims = {d: sum(curves[d]) / n_seg for d in DIMS}
    # Must not be max(recall_probability): that is a softmax over beats, so it
    # falls as ~1/n and the "memory" bar on the brain map tracked how many
    # beats the segmenter happened to emit. The same content split into 3 vs 8
    # beats moved it roughly 2×. encoding_strength is peak arousal — the actual
    # mechanism, and invariant to segmentation.
    memorability = (
        memory["encoding_strength"] if memory else overall_dims["attention"]
    )
    systems = functional_systems(overall_dims, memorability)
    per_segment_systems = [
        functional_systems(
            {d: curves[d][i] for d in DIMS},
            memory["recall_probability"][i] * n_seg / 2 if memory else curves["attention"][i],
        )
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
    yield {"type": "stage", "stage": "segmenting", "detail": "splitting content into narrative beats"}
    segments = await segment_content(request.content, request.content_type)
    roster = personas * request.twins_per_persona
    yield {"type": "start", "total": len(roster), "segments": segments}

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
    yield {"type": "stage", "stage": "changepoints", "detail": "BIC-penalised change-point detection"}
    yield {"type": "stage", "stage": "memory", "detail": "peak-end + serial-position recall model"}
    yield {"type": "stage", "stage": "systems", "detail": "projecting onto functional cognitive systems"}
    result = aggregate_content_study(responses, segments, request.cognitive_load)
    result["content_preview"] = request.content[:140]
    yield {"type": "done", "result": result}
