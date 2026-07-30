"""Turn a study's statistics into findings that answer the question that was asked.

WHAT WAS MISSING. Every study in this product ends the same way: a fixed
battery of numbers and a template f-string. `aggregate_content_study` builds
its verdict as `f"Attention {verb} (ISC {r:.2f}); memory will keep beat {i}; …"`
— the same sentence shape for a pharmaceutical explainer and a sneaker ad,
assembled before anyone said what they wanted to know. The mathematics
underneath is careful; the thing a researcher actually reads is a mail merge.

The gap is not "add an LLM summary". It is that nothing in the pipeline ever
knew what the researcher was trying to find out, so nothing could be specific
to it. This module adds that: a stated question flows in, and what comes out is
a ranked answer built from the study's own numbers.

THE ARCHITECTURE, and why it is inverted from the obvious one.

The obvious version hands the result blob and the question to a model and asks
for analysis. That fails in the way this codebase spends most of its comments
guarding against: the model will produce fluent, specific, confident prose
containing numbers that are not in the data, effects that did not clear their
intervals, and causal claims the design cannot support. Prose is exactly the
medium in which an unsupported claim is indistinguishable from a supported one.

So the model never sees a free-text summarising job. Instead:

  1. HARVEST — deterministic extraction of every quantitative claim the study
     actually licenses, each as an `Evidence` row with a value, an interval, a
     p-value where one exists, and its sample size. Pure code, no model.
  2. CONTROL — Benjamini-Hochberg across the whole family, because a neuro run
     yields dozens of these and reporting the significant subset of dozens of
     nulls means reporting findings forever from data with nothing in it. Each
     row comes out marked `supported` or not.
  3. RANK — relevance to the stated question, combined with effect size and
     support. Deterministic, so the ordering is reproducible and inspectable.
  4. SYNTHESISE — the model writes, but it may only write findings that CITE
     evidence ids, and the numbers rendered to the user come from the evidence
     table rather than from the model's text. A finding citing an id that does
     not exist is dropped before it reaches anyone.

That last constraint is the whole design. The model contributes judgement about
what matters and how to say it — genuinely useful, and not something arithmetic
can supply — while remaining structurally unable to invent a measurement.

WHAT IT REFUSES TO DO. If nothing survives step 2, the report says so. A study
whose numbers are all consistent with noise has a correct answer, and it is not
a list of observations phrased to sound like findings.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

import openai

from .analytics import benjamini_hochberg
from .config import PROFILER_MODEL
from .oai import Refusal, async_client, parse_completion, response_format_for
from .schemas import FindingsReport

# ─────────────────────────────── evidence ────────────────────────────────


@dataclass
class Evidence:
    """One quantitative claim the study licenses.

    `id` is short and stable because it is the citation key the model must use;
    long ids waste tokens in a prompt that carries dozens of them.
    """

    id: str
    label: str
    value: Optional[float]
    #: Rendered value when the number needs units or is categorical.
    display: str = ""
    ci: Optional[Sequence[float]] = None
    p_value: Optional[float] = None
    q_value: Optional[float] = None
    n: Optional[int] = None
    where: str = ""
    #: What a move in this number means — supplied by the probe, never by the
    #: model, so the direction of "good" is a property of the statistic rather
    #: than of whatever the model assumes about the domain.
    interpretation: str = ""
    #: Did it clear its own evidential bar. Set in `control`.
    supported: bool = False
    #: Why it did or did not, in one phrase.
    support_reason: str = ""
    #: |effect| on a comparable 0-1 scale, for ranking.
    magnitude: float = 0.0

    def to_prompt_row(self) -> str:
        bits = [f"[{self.id}] {self.label}"]
        bits.append(f"= {self.display or _fmt(self.value)}")
        if self.ci:
            bits.append(f"95% CI [{_fmt(self.ci[0])}, {_fmt(self.ci[1])}]")
        if self.p_value is not None:
            bits.append(f"p={self.p_value:.4g}")
        if self.q_value is not None:
            bits.append(f"q={self.q_value:.4g}")
        if self.n:
            bits.append(f"n={self.n}")
        bits.append("SUPPORTED" if self.supported else f"NOT SUPPORTED ({self.support_reason})")
        line = " · ".join(bits)
        if self.interpretation:
            line += f"\n      meaning: {self.interpretation}"
        return line

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "value": self.value,
            "display": self.display or _fmt(self.value),
            "ci": list(self.ci) if self.ci else None,
            "p_value": self.p_value,
            "q_value": self.q_value,
            "n": self.n,
            "where": self.where,
            "interpretation": self.interpretation,
            "supported": self.supported,
            "support_reason": self.support_reason,
            "magnitude": round(self.magnitude, 4),
        }


def _fmt(v: Any) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.3g}"
    return str(v)


def _dig(obj: Any, path: str) -> Any:
    """Resolve a dotted path, tolerating absent branches.

    Studies legitimately omit whole blocks — a spatial asset has no retention,
    a two-twin run has no ISC — so a missing path is an ordinary outcome here
    and not an error. Returning None lets one probe table describe every
    variant of a study kind.
    """
    cur = obj
    for part in path.split("."):
        if cur is None:
            return None
        if isinstance(cur, dict):
            cur = cur.get(part)
        elif isinstance(cur, list) and part.isdigit():
            idx = int(part)
            cur = cur[idx] if idx < len(cur) else None
        else:
            cur = getattr(cur, part, None)
    return cur


@dataclass(frozen=True)
class Probe:
    """A declarative pointer at one number in a result, plus how to read it."""

    id: str
    path: str
    label: str
    interpretation: str = ""
    ci: Optional[str] = None
    p: Optional[str] = None
    n: Optional[str] = None
    where: str = ""
    #: Where the "no effect" point sits, for magnitude and for interval-based
    #: support when there is no p-value. None ⇒ the number is descriptive and
    #: is reported without an evidential claim attached.
    null_value: Optional[float] = None
    #: Divisor mapping the raw effect onto ~[0,1] so magnitudes from different
    #: statistics can be ranked against each other at all.
    scale: float = 1.0
    fmt: Optional[Callable[[Any], str]] = None


def _probe_to_evidence(probe: Probe, result: dict) -> Optional[Evidence]:
    raw = _dig(result, probe.path)
    if raw is None:
        return None
    value = float(raw) if isinstance(raw, (int, float)) and not isinstance(raw, bool) else None
    ci = _dig(result, probe.ci) if probe.ci else None
    if ci is not None and not (isinstance(ci, (list, tuple)) and len(ci) == 2):
        ci = None
    p = _dig(result, probe.p) if probe.p else None
    n = _dig(result, probe.n) if probe.n else None
    ev = Evidence(
        id=probe.id,
        label=probe.label,
        value=value,
        display=probe.fmt(raw) if probe.fmt else (_fmt(raw) if value is None else ""),
        ci=[float(ci[0]), float(ci[1])] if ci else None,
        p_value=float(p) if isinstance(p, (int, float)) else None,
        n=int(n) if isinstance(n, (int, float)) else None,
        where=probe.where,
        interpretation=probe.interpretation,
    )
    if value is not None and probe.null_value is not None:
        ev.magnitude = min(1.0, abs(value - probe.null_value) / max(probe.scale, 1e-9))
    return ev


# ────────────────────────── per-kind probe tables ─────────────────────────
#
# Declarative rather than nine bespoke extractor functions. The point is that a
# statistic and its interval, its p-value and its interpretation are declared in
# ONE place — the failure mode with hand-written extractors is a number that
# gets harvested without the uncertainty that belongs to it, which is precisely
# the thing this module exists to prevent.

_CONTENT_PROBES: tuple[Probe, ...] = (
    Probe(
        id="isc", path="isc.overall", label="attention synchrony (ISC)",
        ci="isc.ci", p="isc.p_value", n="isc.n_twins", where="synchrony",
        null_value=0.0, scale=0.6,
        interpretation=(
            "how tightly the content controls WHERE attention goes. High means "
            "the audience moves together and the content is driving; low means "
            "minds wander independently and any average across them is a "
            "composite of people having different experiences."
        ),
    ),
    Probe(
        id="grip", path="isc.grip", label="attention verdict", where="synchrony",
        interpretation="locked-in / engaged / drifting, gated on the permutation test.",
    ),
    Probe(
        id="peakagree", path="memory.peak_agreement",
        label="agreement on where the peak was", n="memory.n_twins", where="memory",
        null_value=0.0, scale=1.0,
        interpretation=(
            "the share of twins whose personal peak fell on the same beat. Low "
            "means the audience has no shared high point — the piece works "
            "differently for different people, and a single 'best moment' is an "
            "artefact of averaging."
        ),
    ),
    Probe(
        id="encoding", path="memory.encoding_strength",
        label="encoding strength (peak arousal)", ci="memory.encoding_strength_ci",
        n="memory.n_twins", where="memory", null_value=0.5, scale=0.5,
        interpretation=(
            "how strongly the strongest moment will consolidate. This is the "
            "mechanism behind what gets remembered at all."
        ),
    ),
    Probe(
        id="remembered", path="memory.remembered_affect",
        label="remembered affect (peak-end)", ci="memory.remembered_affect_ci",
        n="memory.n_twins", where="memory", null_value=0.0, scale=1.0,
        interpretation=(
            "the feeling left behind once the experience is over, averaging the "
            "peak and the ending. Sign matters more than size: negative means "
            "the residue is aversive whatever happened in the middle."
        ),
    ),
    Probe(
        id="completion", path="retention.completion_rate",
        label="share reaching the end", where="retention", null_value=1.0, scale=1.0,
        interpretation="the fraction of the audience still present at the last beat.",
    ),
    Probe(
        id="worstbeat", path="retention.worst_beat",
        label="steepest drop-off beat (0-indexed)", where="retention",
        interpretation="the beat that lost the largest share of who was still watching.",
    ),
    Probe(
        id="share", path="share_intent_mean", label="share intent",
        ci="share_intent_ci", n="twin_count", where="amplification",
        null_value=0.5, scale=0.5,
        interpretation="how likely the audience is to pass this on unprompted.",
    ),
    Probe(
        id="dynamism", path="trajectory.dynamism", label="emotional dynamism",
        where="trajectory", null_value=0.0, scale=2.0,
        interpretation=(
            "total distance travelled through valence-arousal space. Near zero "
            "means the piece is affectively flat — nothing moves."
        ),
    ),
    Probe(
        id="valshift", path="trajectory.net_valence_shift",
        label="net valence shift, first beat to last", where="trajectory",
        null_value=0.0, scale=1.0,
        interpretation="whether the audience ends up better or worse disposed than they started.",
    ),
)

_SWARM_PROBES: tuple[Probe, ...] = (
    Probe(
        id="intent", path="mean_intent", label="mean intent", ci="intent_ci",
        n="twin_count", where="headline", null_value=0.5, scale=0.5,
        interpretation="how close the audience is to acting.",
    ),
    Probe(
        id="engage", path="mean_engagement", label="mean engagement",
        ci="engagement_ci", n="twin_count", where="headline",
        null_value=0.5, scale=0.5,
        interpretation="attention held, independently of whether it converts.",
    ),
    Probe(
        id="polar", path="polarization", label="action-mix entropy",
        where="consensus", null_value=0.0, scale=1.0,
        interpretation=(
            "0 = everyone does the same thing, 1 = evenly spread over every "
            "action. Middling values with high bimodality are two camps; "
            "middling values without it are just noise."
        ),
    ),
    Probe(
        id="bimodal", path="bimodality", label="bimodality coefficient (Sarle)",
        where="consensus", null_value=0.555, scale=0.3,
        interpretation="above 5/9 ≈ 0.555 suggests two distinct camps rather than one spread.",
    ),
    Probe(
        id="segk", path="segments.k", label="segments found", where="segments",
        interpretation="k chosen by the gap statistic; absent entirely when the audience is homogeneous.",
    ),
    Probe(
        id="seggap", path="segments.gap_vs_k1",
        label="segment separation above no-structure (SEs)", where="segments",
        null_value=0.0, scale=3.0,
        interpretation=(
            "how far the split sits above a structureless reference, in standard "
            "errors. Under ~1 the groups are a partition of noise."
        ),
    ),
    Probe(
        id="failed", path="twins_failed", label="twins that failed to respond",
        where="quality", null_value=0.0, scale=10.0,
        interpretation=(
            "dropped twins are not missing at random — a twin that timed out is "
            "not a twin that had no opinion — so a high count means every "
            "interval here is computed on a non-random subsample."
        ),
    ),
)

_COMPARE_PROBES: tuple[Probe, ...] = (
    Probe(
        id="winner", path="winner", label="winning variant", where="ranking",
        interpretation="highest mean intent — NOT necessarily a supported difference.",
    ),
    Probe(
        id="pwin", path="bayesian.p_win", label="P(winner is genuinely best)",
        where="ranking", null_value=0.5, scale=0.5,
        interpretation="posterior probability the leader really leads, not just in this sample.",
    ),
    Probe(
        id="conclusive", path="bayesian.conclusive", label="comparison conclusive",
        where="ranking",
        interpretation="whether the expected loss from picking the leader is inside the ROPE.",
    ),
)

_WALK_PROBES: tuple[Probe, ...] = (
    Probe(
        id="complete", path="completion_rate", label="completion rate",
        n="twin_count", where="funnel", null_value=1.0, scale=1.0,
        interpretation="share reaching the final step without abandoning.",
    ),
    Probe(
        id="convert", path="conversion_rate", label="conversion rate",
        n="twin_count", where="funnel", null_value=0.0, scale=1.0,
        interpretation="share ending in a convert action.",
    ),
    Probe(
        id="worststep", path="survival.worst_step_index",
        label="highest-hazard step", where="funnel",
        interpretation="the step losing the largest share of those who reached it.",
    ),
)

_PRICE_PROBES: tuple[Probe, ...] = (
    Probe(
        id="recprice", path="recommended_price", label="revenue-maximising price tested",
        where="pricing",
        interpretation="the tested point with the highest price × demand, not a continuous optimum.",
    ),
    Probe(
        id="elast", path="econometrics.elasticity", label="price elasticity of demand",
        where="pricing", null_value=-1.0, scale=1.0,
        interpretation=(
            "below −1 is elastic: a price rise loses more volume than it gains "
            "margin. Above −1 is inelastic and there is room to raise."
        ),
    ),
    Probe(
        id="elastr2", path="econometrics.elasticity_r2", label="log-log fit R²",
        where="pricing", null_value=0.0, scale=1.0,
        interpretation="how well a constant-elasticity curve describes these points at all.",
    ),
    Probe(
        id="contopt", path="econometrics.continuous_optimal_price",
        label="fitted continuous optimum", where="pricing",
        interpretation="from the demand fit — may sit outside the tested range, where it is extrapolation.",
    ),
)

_OBJECTION_PROBES: tuple[Probe, ...] = (
    Probe(
        id="dealbreak", path="dealbreaker_rate", label="dealbreaker rate",
        n="twin_count", where="objections", null_value=0.0, scale=1.0,
        interpretation="share for whom an objection is disqualifying rather than negotiable.",
    ),
    Probe(
        id="recommend", path="mean_recommend", label="mean recommend score",
        ci="recommend_ci", n="twin_count", where="objections",
        null_value=0.5, scale=0.5,
        interpretation="willingness to put their own name behind it.",
    ),
    Probe(
        id="objsil", path="cluster_silhouette", label="objection cluster quality",
        where="objections", null_value=0.0, scale=1.0,
        interpretation=(
            "how cleanly the objections separate into themes. Low means the "
            "themes are an imposed grouping of unrelated complaints."
        ),
    ),
)

_VIRALITY_PROBES: tuple[Probe, ...] = (
    Probe(
        id="r0", path="r0", label="R₀ (expected secondary shares)", ci="r0_ci",
        n="twin_count", where="cascade", null_value=1.0, scale=1.0,
        interpretation=(
            "above 1 a cascade can sustain itself; below 1 every seed dies out. "
            "The threshold is sharp and the interval around it is what matters — "
            "an R₀ of 1.1 whose interval spans 1 is not a growing cascade."
        ),
    ),
    Probe(
        id="takeoff", path="takeoff_probability", label="P(cascade takes off)",
        where="cascade", null_value=0.0, scale=1.0,
        interpretation="probability a single seed escapes extinction.",
    ),
    Probe(
        id="csize", path="expected_cascade_size", label="expected cascade size",
        ci="cascade_size_ci", where="cascade",
        interpretation="mean total reach per seed, including the seeds that die immediately.",
    ),
    Probe(
        id="censored", path="censored_frac", label="censored simulation fraction",
        where="quality", null_value=0.0, scale=0.5,
        interpretation=(
            "runs still alive at the generation cap. High values mean the size "
            "quantiles are under-estimates and should be read as lower bounds."
        ),
    ),
)

_OPTIMIZE_PROBES: tuple[Probe, ...] = (
    Probe(
        id="beatseed", path="beat_seed", label="champion beat the seed",
        where="optimisation",
        interpretation="whether the champion's credible interval clears the seed's.",
    ),
    # The honest lift, not the raw one. `intent_lift_vs_seed` is the champion's
    # margin measured on the very sample that crowned it — under a true null it
    # reads +0.087 and is positive in 99.3% of runs, so harvesting it as
    # evidence handed the synthesis a number that is positive almost whatever
    # happened. `selection_adjusted_lift` re-runs the selection inside each
    # split and evaluates on the held-out half (−0.0003 under the same null).
    # This is the same correction `_SEQUENCE_PROBES` already applies, and the
    # two study kinds should not disagree about whether the winner's curse is
    # something you subtract or something you mention.
    Probe(
        id="lift", path="selection_adjusted_lift", label="champion intent lift vs seed",
        ci="selection_adjusted_lift_ci", where="optimisation",
        null_value=0.0, scale=0.3,
        interpretation=(
            "how much the champion beats the seed once the winner's curse is "
            "removed — selection and evaluation on disjoint halves."
        ),
    ),
    Probe(
        id="selbias", path="selection_bias", label="selection bias removed",
        where="optimisation", null_value=0.0, scale=0.3,
        interpretation=(
            "the gap between the in-sample margin and the honest one — how hard "
            "the search was fitting noise."
        ),
    ),
    Probe(
        id="pbeats", path="p_best_beats_seed", label="P(champion act rate > seed)",
        where="optimisation", null_value=0.5, scale=0.5,
        interpretation=(
            "posterior probability the champion's act rate genuinely exceeds "
            "the seed's, from the Beta-Binomial posteriors."
        ),
    ),
    Probe(
        id="converge", path="convergence", label="convergence verdict",
        where="optimisation",
        interpretation="whether the search had enough evidence to stop.",
    ),
)

_SEQUENCE_PROBES: tuple[Probe, ...] = (
    Probe(
        id="gain", path="objective_gain", label="cross-fitted ordering gain",
        ci="objective_gain_ci", p="gain_p_value", n="twin_count", where="ordering",
        null_value=0.0, scale=1.0,
        interpretation=(
            "how much the recommended order beats the submitted one, with "
            "selection and evaluation on disjoint halves so the winner's curse "
            "is excluded."
        ),
    ),
    Probe(
        id="selbias", path="selection_bias", label="selection bias removed",
        where="ordering", null_value=0.0, scale=1.0,
        interpretation=(
            "the gap between the in-sample margin and the honest one — how hard "
            "the search was fitting noise."
        ),
    ),
    Probe(
        id="primacy", path="primacy_effect", label="primacy effect",
        where="position", null_value=0.0, scale=0.3,
        interpretation="extra gain from being first, independent of which message it is.",
    ),
    Probe(
        id="recency", path="recency_effect", label="recency effect",
        where="position", null_value=0.0, scale=0.3,
        interpretation="extra gain from being last.",
    ),
    Probe(
        id="disengage", path="disengage_rate", label="disengagement rate",
        where="ordering", null_value=0.0, scale=1.0,
        interpretation="share who stopped before the sequence finished.",
    ),
)

# The room is the only study whose treatment is OTHER AGENTS, so its evidence is
# about the deliberation rather than about the stimulus. Three of these rows
# exist to let the synthesis say the study FAILED: `placebo_gap`, `orderfx` and
# `infl_r2` are the ones that catch a room that converged because a language
# model is agreeable, because of who happened to speak first, or on an influence
# matrix that was never identifiable. A findings layer that only harvested the
# interesting numbers would report a confident consensus in all three cases.
_ROOM_PROBES: tuple[Probe, ...] = (
    Probe(
        id="roomgap", path="analysis.falsification.room_mean_gap",
        label="public-private gap (room mean)",
        ci="analysis.falsification.gap_ci", p="analysis.falsification.p_value",
        where="falsification", null_value=0.0, scale=0.3,
        interpretation=(
            "how far what the room SAID sat from what its members privately "
            "believed. Positive = the table was more in favour than the people "
            "at it. This is the one thing a real boardroom cannot measure "
            "about itself."
        ),
    ),
    Probe(
        id="conform", path="analysis.conformity.conformity_ratio",
        label="conformity vs Bayesian pooling",
        where="convergence", null_value=1.0, scale=1.0,
        interpretation=(
            "observed convergence rate over the rate a rational pooler of "
            "independent signals would show. 1.0 = they converged on "
            "information; above 1 = they converged on each other."
        ),
    ),
    Probe(
        id="placebogap", path="analysis.placebo.convergence_placebo",
        label="spread remaining in the placebo arm",
        where="control", null_value=1.0, scale=1.0,
        interpretation=(
            "final cross-member spread over opening spread, for members "
            "shown statements from a DIFFERENT room — so LOWER means the "
            "placebo converged harder. The control for model agreeableness: if "
            "it matches the real arm, the deliberation measured nothing and no "
            "finding about influence is admissible."
        ),
    ),
    Probe(
        id="orderfx", path="analysis.order_effect.eta_squared",
        label="speaking-order effect", p="analysis.order_effect.eta_p_value",
        where="control", null_value=0.0, scale=0.3,
        interpretation=(
            "how much where the room landed depended on who spoke first. "
            "Large = the consensus is an artefact of the agenda, not the "
            "argument."
        ),
    ),
    Probe(
        id="spectral", path="analysis.centrality.spectral_gap",
        label="influence spectral gap",
        where="influence", null_value=0.0, scale=1.0,
        interpretation=(
            "how fast the influence structure drives the room together. Near "
            "zero means the disagreement is structurally persistent — this "
            "room does not converge however long it talks."
        ),
    ),
    Probe(
        id="inflr2", path="analysis.influence.n_obs_per_member",
        label="observations per member behind the influence fit",
        where="influence",
        interpretation=(
            "the influence matrix is estimated per member from this many "
            "updates. Too few and it is not identifiable — the fit refuses "
            "rather than regularising into a confident-looking answer."
        ),
    ),
    Probe(
        id="consensus", path="analysis.centrality.consensus_forecast",
        label="forecast consensus if they kept talking",
        where="influence", null_value=0.5, scale=0.5,
        interpretation=(
            "the position the room's influence structure drives toward, "
            "weighting each member by how much the others defer to them "
            "rather than by whether they are right."
        ),
    ),
)


PROBES: dict[str, tuple[Probe, ...]] = {
    "content": _CONTENT_PROBES,
    "swarm": _SWARM_PROBES,
    "compare": _COMPARE_PROBES,
    "walk": _WALK_PROBES,
    "price": _PRICE_PROBES,
    "objection": _OBJECTION_PROBES,
    "virality": _VIRALITY_PROBES,
    "optimize": _OPTIMIZE_PROBES,
    "sequence": _SEQUENCE_PROBES,
    "room": _ROOM_PROBES,
}


# ─────────────────────── derived, list-shaped evidence ────────────────────
#
# The probe table handles scalars. The findings that actually change a decision
# are usually comparisons — THIS beat against the others, THIS variant against
# the control — and those have to be computed, with their uncertainty, rather
# than pointed at.


def _beat_extremes(result: dict) -> list[Evidence]:
    """Per-beat attention against the study's own simultaneous band.

    A beat is only called weak if it sits outside the band covering the WHOLE
    curve, which is the comparison a reader makes by eye anyway. Using the
    pointwise interval here would license "beat 4 is the weak one" on any curve
    with enough beats, since one of them is always lowest.
    """
    out: list[Evidence] = []
    curves = result.get("curves") or {}
    attention = curves.get("attention")
    bands = (result.get("curve_cis") or {}).get("attention")
    segments = result.get("segments") or []
    if not attention or len(attention) < 2:
        return out

    lo_i = min(range(len(attention)), key=lambda i: attention[i])
    hi_i = max(range(len(attention)), key=lambda i: attention[i])
    mean_all = sum(attention) / len(attention)

    for idx, tag, name in ((lo_i, "weakbeat", "weakest"), (hi_i, "strongbeat", "strongest")):
        band = bands[idx] if bands and idx < len(bands) and bands[idx] else None
        # Separated from the curve mean once the whole-curve band is accounted
        # for — not merely the lowest number present.
        separated = bool(band and (band[1] < mean_all or band[0] > mean_all))
        out.append(
            Evidence(
                id=tag,
                label=f"{name} beat: #{idx + 1} attention",
                value=float(attention[idx]),
                display=f"{attention[idx]:.3g} (beat {idx + 1} of {len(attention)})",
                ci=band,
                where="attention curve",
                n=result.get("twin_count"),
                magnitude=min(1.0, abs(attention[idx] - mean_all) / 0.3),
                interpretation=(
                    f"beat {idx + 1} is “{str(segments[idx])[:90]}”. "
                    + (
                        "Its simultaneous band clears the curve mean, so this is a "
                        "real structural feature and not the lowest of several "
                        "equivalent beats."
                        if separated
                        else "Its band overlaps the curve mean, so it is the "
                        f"{name} number without being distinguishable from the rest."
                    )
                ),
            )
        )
        out[-1].supported = separated
        out[-1].support_reason = (
            "band separated from curve mean" if separated else "band overlaps the curve mean"
        )
        out[-1].q_value = None
    return out


def _changepoint_evidence(result: dict) -> list[Evidence]:
    cps = result.get("change_points")
    if not cps:
        return []
    segments = result.get("segments") or []
    return [
        Evidence(
            id=f"cp{i}",
            label=f"structural attention shift entering beat {cp + 1}",
            value=float(cp),
            display=f"beat {cp + 1}",
            where="change points",
            n=result.get("twin_count"),
            supported=True,
            support_reason="cleared the BIC penalty at the effective sample size",
            magnitude=0.7,
            interpretation=(
                f"attention changes regime here — “{str(segments[cp])[:90]}” begins a "
                "measurably different stretch, which is where a re-edit has leverage."
            ),
        )
        for i, cp in enumerate(cps[:4])
    ]


def _dimension_evidence(result: dict) -> list[Evidence]:
    """Which appraisal dimensions are extreme across the whole piece.

    Nine dimensions are measured and the UI shows one. The others are where the
    mechanism lives: a piece can hold attention while producing no reward
    anticipation, and that is a different problem from one that loses attention.
    """
    curves = result.get("curves") or {}
    out: list[Evidence] = []
    interps = {
        "novelty": "how much of this the audience has seen before",
        "goal_relevance": "whether it speaks to a problem they actually have",
        "social_resonance": "the urge to tell someone",
        "threat": "unease, which suppresses approach behaviour",
        "reward_anticipation": "expectation that something good follows",
        "cognitive_effort": "how hard it is to follow; high values throttle everything else",
        "valence": "liking, on a −1..+1 scale",
        "arousal": "activation, independent of whether it is pleasant",
    }
    for dim, interp in interps.items():
        series = curves.get(dim)
        if not series:
            continue
        avg = sum(series) / len(series)
        # Valence is centred on 0; the rest on the 0-1 midpoint.
        null = 0.0 if dim == "valence" else 0.5
        out.append(
            Evidence(
                id=f"dim_{dim}",
                label=f"mean {dim.replace('_', ' ')}",
                value=round(avg, 4),
                where="appraisal dimensions",
                n=result.get("twin_count"),
                magnitude=min(1.0, abs(avg - null) / 0.5),
                interpretation=interp,
            )
        )
    return out


def _ranking_evidence(result: dict) -> list[Evidence]:
    """Variant ranking with lift, for compares."""
    out: list[Evidence] = []
    for i, row in enumerate((result.get("ranking") or [])[:6]):
        row = row if isinstance(row, dict) else getattr(row, "__dict__", {})
        lift = row.get("lift_vs_control_pct")
        out.append(
            Evidence(
                id=f"var{i}",
                label=f"variant “{row.get('name')}” mean intent",
                value=row.get("mean_intent"),
                display=(
                    f"{_fmt(row.get('mean_intent'))}"
                    + (f" ({lift:+.1f}% vs control)" if isinstance(lift, (int, float)) else " (control)")
                ),
                where="ranking",
                magnitude=min(1.0, abs(lift) / 50.0) if isinstance(lift, (int, float)) else 0.3,
                interpretation=(
                    "raw arm mean. Ranking on it alone lets a small arm outrank a "
                    "large one on noise — read it against the posterior."
                ),
            )
        )
    return out


def _objection_evidence(result: dict) -> list[Evidence]:
    out: list[Evidence] = []
    for i, theme in enumerate((result.get("top_objections") or [])[:6]):
        theme = theme if isinstance(theme, dict) else getattr(theme, "__dict__", {})
        share = theme.get("share") or theme.get("frequency")
        out.append(
            Evidence(
                id=f"obj{i}",
                label=f"objection: {str(theme.get('theme') or theme.get('label'))[:70]}",
                value=float(share) if isinstance(share, (int, float)) else None,
                display=f"{share:.0%} raised it" if isinstance(share, (int, float)) else "",
                where="objections",
                magnitude=float(share) if isinstance(share, (int, float)) else 0.3,
                interpretation="a recurring reason the audience gave for not proceeding.",
            )
        )
    return out


def _step_evidence(result: dict) -> list[Evidence]:
    """Per-step hazards for a walk — where the funnel actually leaks."""
    steps = (result.get("survival") or {}).get("steps") or []
    out: list[Evidence] = []
    for st in steps:
        hazard = st.get("hazard")
        if not isinstance(hazard, (int, float)) or hazard <= 0:
            continue
        out.append(
            Evidence(
                id=f"step{st.get('step_index', len(out))}",
                label=f"step {st.get('step_index', 0) + 1} hazard",
                value=float(hazard),
                display=f"{hazard:.0%} of those who reached it",
                n=st.get("entered"),
                where="funnel",
                magnitude=float(hazard),
                interpretation=(
                    "conditional drop-off: the share lost among those who got "
                    "here, which is the number that identifies a broken step. A "
                    "raw count would blame whichever step the most people saw."
                ),
            )
        )
    return sorted(out, key=lambda e: -(e.value or 0))[:5]


def _room_member_evidence(result: dict) -> list[Evidence]:
    """Per-member rows the scalar probes cannot reach: who hid their position,
    who carried the room, and whose absence would have changed where it landed.

    Everything lives under `result["analysis"]` — `room_analysis` returns its own
    top-level dict and `boardroom` nests it — so these paths carry the prefix and
    so do the probes. Getting that wrong is SILENT: `_dig` returns None for an
    absent branch, which is right for a study that legitimately skipped a block,
    so a probe table pointing one level too high harvests nothing while looking
    exactly like a room that had no findings.

    All three row types are gated on the room-level controls rather than reported
    on their own. A per-member influence weight from a refused fit is not a weak
    finding, it is not a finding — so those rows are withheld entirely rather
    than emitted with a caveat the synthesis is free to drop.
    """
    out: list[Evidence] = []
    analysis = result.get("analysis") or {}

    # If the run's own instrument check says the gap is below what the model can
    # resolve, no falsification row is admissible — not the room mean and not the
    # per-member breakdown. A 0.02 gap presented with an interval reads as a
    # small real effect; it is the absence of a measurement, and the difference
    # matters more here than anywhere else in the product because "the room is
    # lying to itself" is the most quotable thing this study could say.
    check = result.get("instrument_check") or {}
    fals = (
        {}
        if check.get("falsification_below_resolution")
        else (analysis.get("falsification") or {})
    )
    if fals.get("refused"):
        fals = {}

    # `per_member_gap` is a list of FLOATS aligned with `names`; the public and
    # private levels live on `roster`. Zipping is the only way to attach a name,
    # and an unnamed boardroom finding is worth nothing.
    gap_names = fals.get("names") or analysis.get("names") or []
    gaps = fals.get("per_member_gap") or []
    roster = {
        r.get("name"): r
        for r in (analysis.get("roster") or [])
        if isinstance(r, dict)
    }
    rows = [
        (nm, float(g))
        for nm, g in zip(gap_names, gaps)
        if isinstance(g, (int, float))
    ]
    for nm, gap in sorted(rows, key=lambda t: -abs(t[1]))[:3]:
        seat = roster.get(nm) or {}
        pub, pri = seat.get("final_public"), seat.get("final_private")
        direction = (
            "more supportive out loud than in private"
            if gap > 0
            else "more sceptical out loud than in private"
        )
        levels = (
            f" (public {float(pub):.2f} vs private {float(pri):.2f})"
            if isinstance(pub, (int, float)) and isinstance(pri, (int, float))
            else ""
        )
        out.append(
            Evidence(
                id=f"fals_{_slug(nm)}",
                label=f"{nm}: public-private gap",
                value=round(gap, 4),
                where="falsification",
                interpretation=f"said {abs(gap):.2f} {direction}{levels}.",
            )
        )

    controls_hold = _room_controls_hold(result)
    infl = analysis.get("influence") or {}
    cent = analysis.get("centrality") or {}
    if controls_hold and not infl.get("refused"):
        names = infl.get("names") or analysis.get("names") or []
        weights = cent.get("centrality") or []
        pairs = [
            (nm, float(w))
            for nm, w in zip(names, weights)
            if isinstance(w, (int, float))
        ]
        if pairs:
            top_name, top_w = max(pairs, key=lambda t: t[1])
            share = top_w / (sum(w for _, w in pairs) or 1.0)
            out.append(
                Evidence(
                    id=f"cent_{_slug(top_name)}",
                    label=f"{top_name}: share of the room's weight",
                    value=round(share, 4),
                    where="influence",
                    interpretation=(
                        f"the consensus weights {top_name} at {share:.0%} — the "
                        "others defer to them more than they defer to anyone. "
                        "This is structural position, not whether they are right."
                    ),
                )
            )

        # `absence`, not `counterfactuals`. Each entry carries `delta_room_mean`,
        # which is None for a member whose removal leaves a room that does not
        # settle — skipped rather than rendered as a zero move.
        absent = [
            a
            for a in (analysis.get("absence") or [])
            if isinstance(a, dict)
            and isinstance(a.get("delta_room_mean"), (int, float))
        ]
        for cf in sorted(absent, key=lambda a: -abs(a["delta_room_mean"]))[:2]:
            delta = float(cf["delta_room_mean"])
            out.append(
                Evidence(
                    id=f"cf_{_slug(cf.get('name', '?'))}",
                    label=f"room without {cf.get('name', '?')}",
                    value=round(delta, 4),
                    where="influence",
                    interpretation=(
                        f"remove them and the room's settled position moves "
                        f"{delta:+.2f}."
                    ),
                )
            )
    return out


def _room_controls_hold(result: dict) -> bool:
    """Did the deliberation measure anything at all.

    Reads the server's own `separation` label rather than recomputing it, and
    that choice fixed a real inversion. `convergence_ratio` is
    sd_final / sd_innate — the spread REMAINING — so a LOWER number means an arm
    converged HARDER. An earlier version here compared
    `real_convergence > placebo_convergence * 1.25`, which is wrong twice over:
    backwards in direction, and on field names that do not exist. It would have
    withheld every influence claim exactly when the control was good, and
    published them when the control had failed.

    `separation` is one of "none …", "marginal — within the position
    quantisation", or "clear — the real arm converged further". Only the last
    licenses a claim about who moved whom. Absent a placebo arm this returns
    True: the researcher chose to run without the control, which is their call,
    and `limits` says so.
    """
    pl = (result.get("analysis") or {}).get("placebo") or {}
    if not pl.get("ran"):
        return True
    separation = str(pl.get("separation") or "")
    if not separation:
        return True
    return separation.startswith("clear")


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())[:10] or "x"


DERIVED: dict[str, tuple[Callable[[dict], list[Evidence]], ...]] = {
    "content": (_beat_extremes, _changepoint_evidence, _dimension_evidence),
    "compare": (_ranking_evidence,),
    "objection": (_objection_evidence,),
    "walk": (_step_evidence,),
    "room": (_room_member_evidence,),
}


# ──────────────────────────── harvest / control ───────────────────────────


#: Rows a study's own instrument check can veto. A probe is declarative and
#: cannot see a precondition, so the veto lives here: the room's public/private
#: gap is withheld entirely when the run reports it below the resolution floor,
#: rather than being harvested and left for the synthesis to caveat.
_VETOED_WHEN: dict[str, tuple[str, ...]] = {
    "falsification_below_resolution": ("roomgap",),
}


def harvest(kind: str, result: dict) -> list[Evidence]:
    """Every quantitative claim the study licenses. Pure, no model."""
    out: list[Evidence] = []
    check = result.get("instrument_check") or {}
    vetoed = {
        pid
        for flag, ids in _VETOED_WHEN.items()
        if check.get(flag)
        for pid in ids
    }
    for probe in PROBES.get(kind, ()):
        if probe.id in vetoed:
            continue
        ev = _probe_to_evidence(probe, result)
        if ev is not None:
            out.append(ev)
    for fn in DERIVED.get(kind, ()):
        try:
            out.extend(fn(result))
        except (KeyError, TypeError, ValueError, IndexError):
            # A derived comparison that cannot be computed is not a failure of
            # the study. Skipping it loses one row; raising would lose the
            # entire report over a shape that was always optional.
            continue
    return out


def control(evidence: list[Evidence], alpha: float = 0.05) -> dict:
    """Mark what survives, correcting across the whole family.

    Three regimes, because the evidence is not homogeneous:

    · a p-value ⇒ Benjamini-Hochberg across every p-value in the family, and
      support means q ≤ α. Testing each at its own nominal 5% across a family
      of this size is how a study reports a finding every time.
    · an interval but no p-value ⇒ supported when the interval excludes the
      null. This is the same decision a reader makes looking at the chart.
    · neither ⇒ descriptive. Reported, never called a finding. `dynamism` and
      `mean novelty` belong here: real numbers, no evidential claim attached.
    """
    ps = [e.p_value for e in evidence]
    bh = benjamini_hochberg(ps, alpha=alpha)
    for ev, q, rejected in zip(evidence, bh["q_values"], bh["rejected"]):
        ev.q_value = q
        if ev.p_value is not None:
            ev.supported = bool(rejected)
            ev.support_reason = (
                f"q={q:.3g} ≤ {alpha} after FDR control over {bh['n_tests']} tests"
                if rejected
                else f"q={q:.3g} > {alpha} once corrected for {bh['n_tests']} simultaneous tests"
            )
        elif ev.ci and ev.value is not None:
            lo, hi = ev.ci[0], ev.ci[1]
            if hi - lo <= 1e-12:
                ev.supported = False
                ev.support_reason = "zero-width interval — no observed variation, not certainty"
            else:
                # The null for an interval-only row is whichever reference the
                # probe declared; absent one, zero.
                excludes = not (lo <= 0.0 <= hi)
                ev.supported = excludes
                ev.support_reason = (
                    "interval excludes zero" if excludes else "interval spans zero"
                )
        elif not ev.support_reason:
            ev.supported = False
            ev.support_reason = "descriptive — no interval or test attached"
    return bh


_WORD = re.compile(r"[a-z][a-z0-9_-]{2,}")
_STOP = {
    "the", "and", "for", "are", "was", "with", "that", "this", "how", "why",
    "what", "does", "did", "will", "would", "should", "can", "our", "their",
    "from", "into", "about", "which", "when", "who", "you", "your", "not",
    "but", "any", "all", "more", "most", "than", "then", "them",
    "have", "has", "had", "its", "were", "been", "being", "there", "here",
    "some", "such", "only", "just", "also", "much", "many", "very", "get",
    "got", "make", "makes", "made", "really", "actually", "anything",
}

# Suffixes stripped longest-first. Deliberately shallow: this is a tie-break on
# an evidence table of ~20 rows, not a search engine, and an aggressive stemmer
# on a vocabulary this small collides more often than it matches ("rating" and
# "rate" are the same concept; "price" and "precision" must not become one).
_SUFFIXES = ("ingly", "edly", "ing", "ies", "ied", "ess", "ed", "es", "ly", "s")


def _stem(word: str) -> str:
    """Crude suffix stripping so a question and an instrument label meet.

    Researchers write "pricing", "buyers", "dropped"; the evidence table says
    "price", "buyer", "drop-off". Raw substring matching missed every one of
    those pairs in the direction that matters — `"pricing" in "price
    elasticity"` is False — so a question about pricing ranked the pricing
    evidence no higher than anything else, which is the one job ranking has.
    """
    for suf in _SUFFIXES:
        if len(word) > len(suf) + 2 and word.endswith(suf):
            base = word[: -len(suf)]
            return base + "y" if suf in ("ies", "ied") else base
    return word


def _variants(word: str) -> tuple[str, ...]:
    """The forms of one word worth matching on: itself, its stem, and the stem
    with a restored silent 'e'.

    The third is not a nicety. English drops the 'e' before '-ing', so
    `_stem("pricing")` is "pric" while `_stem("price")` is "price" — the two
    forms of the SAME word stem to different strings, and a concept map keyed
    on either one misses the other. Without restoration "pricing" reached the
    price evidence only by the accident of "pric" being a substring of "price",
    and "sharing" reached none of the cascade rows at all.

    The same argument covers the other two regular spelling changes English
    makes at a suffix boundary, both of which occur in the questions this
    product is actually asked: a doubled final consonant ("dropped" → "dropp",
    which is not "drop", and drop-off is the most-asked-about thing in a
    retention study) and the agent "-er" ("buyers" → "buyer", which is not
    "buy"). Neither reaches its concept without being undone here.
    """
    stem = _stem(word)
    forms = {word, stem}
    if stem and stem != word and not stem.endswith("e"):
        forms.add(stem + "e")
    # undouble a final consonant: dropped → dropp → drop
    if len(stem) > 3 and stem[-1] == stem[-2] and stem[-1] not in "aeiou":
        forms.add(stem[:-1])
    # agent noun: buyer → buy, viewer → view
    for f in tuple(forms):
        if len(f) > 4 and f.endswith("er"):
            forms.add(f[:-2])
    return tuple(sorted(forms))


# Natural language → the vocabulary the instruments actually measure in.
#
# The gap this closes is not spelling, it is register. Nobody asks "is
# reward_anticipation low at beat 4"; they ask "does this make anyone want it".
# Without a bridge, the appraisal dimension that answers the question sorts
# below rows that merely happen to share a word with it. Keys are stems, so a
# question term is looked up after `_stem`.
_CONCEPT_TERMS: dict[str, tuple[str, ...]] = {
    "anxiety": ("threat",), "anxiou": ("threat",), "worri": ("threat",),
    "nervou": ("threat",), "afraid": ("threat",), "fear": ("threat",),
    "scare": ("threat",), "risk": ("threat",), "safe": ("threat",),
    "trust": ("threat", "objection", "dealbreaker"),
    "confus": ("cognitive effort", "effort"), "unclear": ("cognitive effort",),
    "complex": ("cognitive effort",), "hard": ("cognitive effort",),
    "difficult": ("cognitive effort",), "understand": ("cognitive effort",),
    "want": ("reward", "intent"), "desir": ("reward", "intent"),
    "excit": ("reward", "arousal"), "motivat": ("reward", "intent"),
    "buy": ("intent", "conversion", "price"), "purchas": ("intent", "conversion"),
    "convert": ("conversion", "intent"), "sign": ("conversion", "completion"),
    "bore": ("attention", "engagement"), "engag": ("attention", "engagement"),
    "interest": ("attention", "engagement"), "notic": ("attention", "novelty"),
    "distract": ("attention",), "skip": ("attention", "retention"),
    "surpris": ("novelty",), "fresh": ("novelty",), "familiar": ("novelty",),
    "remember": ("memory", "encoding", "peak-end"),
    "recall": ("memory", "encoding"), "memorab": ("memory", "encoding"),
    "forget": ("memory", "encoding"),
    "share": ("social", "share", "r₀", "cascade"),
    "viral": ("cascade", "r₀", "takeoff"), "spread": ("cascade", "r₀"),
    "recommend": ("recommend", "social"), "word-of-mouth": ("social", "cascade"),
    "price": ("price", "elasticity", "willingness"),
    "cost": ("price", "elasticity"), "expensiv": ("price", "elasticity"),
    "cheap": ("price", "elasticity"), "afford": ("price", "elasticity"),
    "leav": ("retention", "drop-off", "disengagement"),
    "drop": ("retention", "drop-off", "hazard"),
    "abandon": ("retention", "drop-off", "hazard", "completion"),
    "stop": ("retention", "drop-off", "disengagement"),
    "stay": ("retention", "completion"), "finish": ("retention", "completion"),
    "friction": ("hazard", "drop-off", "effort"),
    "step": ("funnel", "step", "hazard"), "flow": ("funnel", "step"),
    "objection": ("objection", "dealbreaker"),
    "concern": ("objection", "dealbreaker"), "block": ("dealbreaker",),
    "disagree": ("polarization", "bimodality", "segments"),
    "split": ("polarization", "bimodality", "segments"),
    "divid": ("polarization", "bimodality", "segments"),
    "agree": ("synchrony", "agreement", "consensus"),
    "segment": ("segments",), "group": ("segments",), "audienc": ("segments",),
    "feel": ("valence", "affect"), "like": ("valence", "affect"),
    "love": ("valence", "affect"), "hate": ("valence", "affect"),
    "emotion": ("valence", "arousal", "affect"),
    "beginning": ("primacy", "beat"), "start": ("primacy", "beat"),
    "opening": ("primacy", "beat"), "hook": ("primacy", "beat", "attention"),
    "end": ("recency", "peak-end", "beat"), "ending": ("recency", "peak-end"),
    "headlin": ("variant", "champion"), "copy": ("variant", "champion"),
    "wins": ("winner", "champion"), "win": ("winner", "champion"),
    "better": ("winner", "champion", "lift"), "best": ("winner", "champion"),
    "order": ("ordering", "sequence"), "sequenc": ("ordering", "sequence"),
}


def rank(evidence: list[Evidence], question: str) -> list[Evidence]:
    """Order by relevance to the question, then by how much is actually there.

    Deterministic on purpose. The model gets to decide what a finding MEANS;
    it does not get to decide which numbers it was shown, because a model that
    selects its own evidence can always find support for whatever it drafts
    first.

    Relevance is lexical overlap between the question and the row's label and
    interpretation, over STEMS rather than raw words, and widened by a concept
    map from natural language to the instrument vocabulary. Both corrections
    matter because researchers and instruments do not speak the same way:

    · Raw substring matching is directional and silently misses the direction
      that occurs. "pricing" does not appear in "price elasticity of demand",
      so the canonical example question — does the pricing section create
      anxiety — scored ZERO against the pricing evidence and zero against
      `threat`, and ranking degenerated to support-and-magnitude, i.e. to
      ignoring the question entirely. Stemming fixes the first half.
    · The second half is register, not spelling. No question asks about
      `reward_anticipation`; it asks whether anyone will want the thing. The
      concept map is what lets the dimension that answers a question outrank a
      row that merely shares a word with it.

    It stays crude on purpose, and it is allowed to be: it is a tie-break on top
    of support and effect size over ~20 rows, and every row is sent to the model
    regardless — the ordering decides emphasis, not inclusion. `rank` never
    filters, so no ranking bug can hide a contradicting measurement.

    Relevance is the FRACTION of the question's terms that landed, not a count
    capped at three. A count rewards long questions for being long: six terms
    matching two scored the same as two matching two, so padding a question
    raised every row's relevance equally and flattened the ordering it was
    supposed to sharpen.
    """
    raw = {w for w in _WORD.findall(question.lower()) if w not in _STOP}
    if not raw:
        # No question, or nothing but stopwords: relevance is undefined rather
        # than zero. Falling through with an empty term set would divide by
        # zero; scoring every row 0 relevance is the correct behaviour and
        # leaves support and magnitude to order the table.
        terms: dict[str, tuple[str, ...]] = {}
    else:
        terms = {}
        for w in raw:
            forms = _variants(w)
            # The row text is matched against every form of the term and against
            # whatever instrument words the concept map associates with ANY of
            # them — looking the concept up under one arbitrary form is how
            # "pricing" failed to reach a map keyed on "price".
            expansions = set(forms)
            for f in forms:
                expansions.update(_CONCEPT_TERMS.get(f, ()))
            terms[w] = tuple(sorted(expansions))

    def score(ev: Evidence) -> float:
        text = f"{ev.label} {ev.interpretation} {ev.where}".lower()
        stemmed = " ".join(_stem(t) for t in _WORD.findall(text))
        # One hit per QUESTION TERM, not per expansion — otherwise a term with
        # three synonyms outweighs a term with one purely by how well this
        # module happens to know the word.
        hits = sum(
            1 for exps in terms.values() if any(e in text or e in stemmed for e in exps)
        )
        relevance = hits / len(terms) if terms else 0.0
        return (
            2.0 * relevance
            + 1.0 * (1.0 if ev.supported else 0.0)
            + 0.75 * ev.magnitude
        )

    return sorted(evidence, key=lambda e: (-score(e), e.id))


# ───────────────────────────── synthesis ─────────────────────────────────


_FINDINGS_SYSTEM = """You are the analyst who reads a completed synthetic-audience
study and tells the researcher what it means for the specific question they
asked. You are rigorous, concrete, and you do not pad.

YOU ARE GIVEN an EVIDENCE TABLE. Every row is a number this study actually
computed, with its uncertainty and whether it survived multiple-comparison
control. You may reason about the stimulus and the audience, but every
quantitative claim you make must come from a row and cite its [id].

HARD RULES.
1. Cite evidence by id in `evidence_ids`. A finding with no citation is invalid.
2. NEVER state a number that is not in a row you cite. Do not recompute,
   rescale, convert to percentages, or round differently.
3. A row marked NOT SUPPORTED may be discussed ONLY as an absence — "we cannot
   distinguish this from chance" — never as a small or emerging effect.
4. If nothing is supported, say so plainly in `answer` and return few or no
   findings. A study of noise has a correct answer and it is not a list of
   observations phrased to sound like findings.
5. No causal language the design cannot carry. These are synthetic personas
   reacting to a stimulus, not a randomised trial on humans.

WHAT MAKES A FINDING GOOD.
- It is about THIS content and THIS audience. Name the beat, the variant, the
  step, the objection. "Beat 4 loses a third of the audience" beats "retention
  could be improved".
- It states the mechanism, not just the number: WHY does this beat lose people,
  given what the appraisal dimensions say happened there.
- Its recommended action is something a person could do on Monday to this
  specific asset.
- It answers the researcher's question, or explains why the study cannot.

Order findings by decision value, most consequential first."""


def _evidence_block(evidence: list[Evidence]) -> str:
    lines = []
    for ev in evidence:
        lines.append("  " + ev.to_prompt_row())
    return "\n".join(lines)


def _context_block(kind: str, result: dict, stimulus: str) -> str:
    bits = [f"STUDY TYPE: {kind}"]
    n = result.get("twin_count")
    if n:
        bits.append(f"AUDIENCE: {n} synthetic twins")
    if result.get("cognitive_load"):
        bits.append(f"COGNITIVE LOAD: {result['cognitive_load']}")
    if result.get("axis"):
        bits.append(
            f"BEAT AXIS: {result['axis']} (via {result.get('beat_source')})"
        )
    segments = result.get("segments")
    if segments:
        bits.append("BEATS:")
        for i, s in enumerate(segments):
            bits.append(f"  {i + 1}. {str(s)[:220]}")
    # The room's context is WHO was in it and what was put to them. Without the
    # cast the synthesis cannot name anyone, and an unnamed finding about a
    # boardroom ("one member privately disagreed") is worth nothing.
    cast = result.get("cast") or []
    if cast:
        # The result carries `motion_preview` at the top level (truncated for the
        # stream) and the full `motion` inside `analysis`. Reading only "motion"
        # found neither and silently put the room's own question out of reach of
        # the synthesis — the cast would be named against a blank motion.
        analysis = result.get("analysis") or {}
        motion = (
            analysis.get("motion")
            or result.get("motion")
            or result.get("motion_preview")
            or ""
        )
        bits.append(f"MOTION PUT TO THE ROOM:\n{str(motion)[:1200]}")
        bits.append("WHO IS AT THE TABLE (declared dispositions, self-described):")
        for m in cast:
            if not isinstance(m, dict):
                continue
            bits.append(
                f"  {m.get('name', '?')} — {m.get('role', '')}; "
                f"stake: {str(m.get('stake', ''))[:120]}; "
                f"declared openness {m.get('openness', '?')}, "
                f"seniority {m.get('seniority', '?')}"
            )
        bits.append(
            "The declared numbers are what each character SAID about itself. "
            "Fitted susceptibility is what its behaviour showed. Where they "
            "disagree, that disagreement is itself a finding."
        )
    withheld = result.get("withheld") or {}
    if withheld:
        bits.append(
            "NOT COMPUTABLE FOR THIS FORMAT (do not claim these): "
            + ", ".join(withheld.keys())
        )
    if stimulus:
        bits.append(f"THE STIMULUS UNDER TEST:\n{stimulus[:4000]}")
    return "\n".join(bits)


#: A stated question is optional; without one the engine still has to produce
#: something specific, so it falls back to the question the study kind exists
#: to answer rather than to a generic summary.
_DEFAULT_QUESTION = {
    "content": "What does this content do to an audience, and what is the single highest-leverage change to it?",
    "swarm": "How will this audience respond to this stimulus, and what stops them acting?",
    "compare": "Which variant should we ship, and is the difference real?",
    "walk": "Where does this flow lose people, and why there?",
    "price": "What should we charge, and how confident can we be?",
    "objection": "What stops this audience buying, and which objection is worth answering first?",
    "virality": "Will this spread on its own, and what would make it spread further?",
    "optimize": "Did the optimiser find genuinely better copy, and what did it learn?",
    "sequence": "What order should these messages go in, and does the order actually matter?",
    "room": "What would this room actually decide, who really decides it, and how much of the agreement is real?",
}


async def build_findings(
    kind: str,
    result: dict,
    question: str = "",
    stimulus: str = "",
    alpha: float = 0.05,
) -> dict:
    """Evidence → controlled → ranked → written. The whole pipeline.

    Returns a dict safe to merge into a study result. Never raises for model
    reasons: a study that completed is worth returning even if the synthesis on
    top of it did not, so failures come back as `error` alongside the evidence
    table, which is useful on its own.
    """
    evidence = harvest(kind, result)
    bh = control(evidence, alpha=alpha)
    asked = (question or "").strip()
    ordered = rank(evidence, asked or _DEFAULT_QUESTION.get(kind, ""))
    supported = [e for e in ordered if e.supported]

    base = {
        "question": asked,
        "evidence": [e.to_dict() for e in ordered],
        "multiplicity": {
            "n_tests": bh["n_tests"],
            "n_rejected": bh["n_rejected"],
            "alpha": alpha,
            "cutoff": bh["cutoff"],
            "method": bh["method"],
        },
        "n_supported": len(supported),
        "method": (
            "evidence harvested deterministically from the study's own "
            "statistics, Benjamini-Hochberg FDR control across the family, then "
            "synthesis constrained to cite harvested rows"
        ),
    }

    if not evidence:
        return {
            **base,
            "report": None,
            "error": "This study produced no harvestable quantitative claims.",
        }

    prompt_question = asked or _DEFAULT_QUESTION.get(kind, "What does this study show?")
    user = (
        f"THE RESEARCHER'S QUESTION:\n{prompt_question}\n\n"
        f"{_context_block(kind, result, stimulus)}\n\n"
        f"EVIDENCE TABLE ({len(evidence)} rows, {len(supported)} supported after "
        f"FDR control over {bh['n_tests']} tests):\n{_evidence_block(ordered)}\n"
    )

    try:
        completion = await async_client().chat.completions.create(
            # The profiler model, not the twin model. This is one
            # judgement-heavy call whose output is the thing the researcher
            # reads and acts on; every twin call in the run was cheaper than
            # getting this one wrong.
            model=PROFILER_MODEL,
            messages=[
                {"role": "system", "content": _FINDINGS_SYSTEM},
                {"role": "user", "content": user},
            ],
            temperature=0.3,
            response_format=response_format_for(FindingsReport),
        )
        report = parse_completion(completion, FindingsReport)
    except (openai.OpenAIError, Refusal, RuntimeError, ValueError) as exc:
        return {**base, "report": None, "error": f"Synthesis unavailable: {exc}"}

    return {**base, "report": validate_report(report, evidence)}


def validate_report(report: FindingsReport, evidence: list[Evidence]) -> dict:
    """Drop anything the evidence does not carry.

    This is the enforcement half of the citation rule, and it has to exist in
    code rather than only in the prompt. A model asked to cite will mostly
    cite; "mostly" is not a property you can put behind a number a researcher
    is going to act on.

    Findings citing no known id are removed outright. Findings citing a mix
    keep only the ids that resolve, and are demoted to low confidence if every
    row they stand on turned out to be unsupported — a finding built entirely
    on rows that did not clear FDR is exactly the overclaim being guarded
    against, and it reads as authoritative unless it is marked.
    """
    known = {e.id: e for e in evidence}
    kept: list[dict] = []
    dropped = 0
    for finding in report.findings:
        ids = [i for i in finding.evidence_ids if i in known]
        if not ids:
            dropped += 1
            continue
        rows = [known[i] for i in ids]
        any_supported = any(r.supported for r in rows)
        item = finding.model_dump()
        item["evidence_ids"] = ids
        if not any_supported and item["confidence"] != "low":
            item["confidence"] = "low"
            item["downgraded"] = (
                "every cited measurement is consistent with chance once "
                "corrected for multiple comparisons"
            )
        kept.append(item)

    return {
        "answer": report.answer,
        "findings": kept,
        "what_would_change_the_answer": report.what_would_change_the_answer,
        "limits": report.limits,
        "dropped_uncited": dropped,
    }
