"""Pydantic schemas shared across ingest, profiling, persona seeding, and the swarm.

Design stance (deliberate, documented):
- The LLM emits *behavioral segment signals* with confidence — never psychometric
  verdicts about a person.
- The Big Five vector is a *soft persona seed* derived by transparent rules in
  persona.py, carrying wide confidence bands. It parameterizes synthetic twins;
  it is not a clinical or individual claim.
"""

import re
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------- ingest

# The zone vocabulary is the whole "whitelist-only" architectural claim, and it
# was never actually enforced server-side. Lowercase alnum with dashes and
# underscores admits every zone the SDK ships with (hero, pricing-table, cta,
# checkout, content) and excludes the two things that made this a hole: an
# email address or other identifier used as a zone name, and a sentence of
# prose aimed at the model that reads the metrics block.
_ZONE_KEY = re.compile(r"[a-z0-9_-]{1,32}")


class FeaturePayload(BaseModel):
    segment_start: float
    segment_end: float
    event_count: int = Field(ge=0)
    scroll_velocity_mean: float = 0.0
    scroll_velocity_variance: float = 0.0
    scroll_direction_changes: int = 0
    micro_hesitation_count: int = 0
    hesitation_total_ms: float = 0.0
    rage_click_bursts: int = 0
    click_count: int = 0
    backtrack_ratio: float = Field(default=0.0, ge=0.0, le=1.0)

    # Zone names are AUTHOR-SUPPLIED strings that travel a long way: they are
    # json.dumps'd into the Layer-4 analyst prompt, and the analyst's output is
    # interpolated verbatim into every twin's *system* message. `/v1/ingest`
    # needs no API key, so without a key pattern this is an unauthenticated
    # path from a `data-cs` attribute on any page to instructions inside every
    # simulation — and a PII channel, since `data-cs="user-alice@corp.com"` was
    # equally acceptable. The pattern is the trust boundary; `_ZONE_KEY` is
    # enforced by the validator below.
    zone_dwell_ms: dict[str, float] = Field(default_factory=dict, max_length=64)
    zone_click_counts: dict[str, int] = Field(default_factory=dict, max_length=64)

    # cognitive-modeling series (optional; older SDKs don't send them).
    #
    # Bounded because this is the only unauthenticated write endpoint in the
    # product. A 10-second segment at the SDK's own sampling rates produces a
    # few hundred points; these ceilings are an order of magnitude above any
    # honest client and stop a single POST from being a memory amplifier.
    event_stream: list[list[float]] = Field(
        default_factory=list,
        max_length=20_000,
        description="[[t_ms, symbol_id]] symbolic event stream; alphabet matches cognition.EVENT_SYMBOLS",
    )
    velocity_series: list[list[float]] = Field(
        default_factory=list,
        max_length=20_000,
        description="[[t_ms, velocity]] scroll kinematics",
    )
    decision_latencies_ms: list[float] = Field(
        default_factory=list,
        max_length=5_000,
        description="quiet-period → click decision latencies",
    )

    @field_validator("zone_dwell_ms", "zone_click_counts")
    @classmethod
    def _reject_unsafe_zone_keys(cls, v: dict) -> dict:
        """Zone names must look like zone names.

        Rejecting rather than sanitising: a payload with a key that does not
        match is not a slightly-wrong payload from an honest SDK, it is either
        a misconfigured integration or an injection attempt, and both deserve
        a 422 rather than silent partial acceptance.
        """
        bad = [k for k in v if not _ZONE_KEY.fullmatch(k)]
        if bad:
            raise ValueError(
                "zone names must match [a-z0-9_-]{1,32} (lowercase); "
                f"rejected: {bad[:5]!r}"
            )
        return v


class IngestEnvelope(BaseModel):
    site_id: str = Field(min_length=1, max_length=128)
    consent: bool
    sdk_version: str
    page_path: str = Field(max_length=512)
    features: FeaturePayload
    panel_token: Optional[str] = Field(
        default=None,
        max_length=64,
        description="Present when the visitor is an enrolled research-panel member.",
    )


# ---------------------------------------------------------------- jobs & panel


class JobCreate(BaseModel):
    kind: Literal["swarm", "compare", "walk", "batch_swarm"]
    payload: dict = Field(
        description="The matching request body (SwarmRunRequest / CompareRequest / WalkRequest)."
    )


class PanelMemberCreate(BaseModel):
    label: str = Field(
        min_length=1,
        max_length=128,
        description="Admin-facing label (e.g. 'panelist #42, US cohort'). Never store PII here.",
    )


# ---------------------------------------------------------------- Layer 4 output


class BehavioralSignal(BaseModel):
    """Structured output of the Layer-4 analyst. Soft signals + evidence, never verdicts."""

    deliberation: Literal["low", "medium", "high"] = Field(
        description="How carefully the user appears to process content (dwell/hesitation pattern)."
    )
    frustration_signal: Literal["none", "mild", "elevated"] = Field(
        description="Rage clicks, erratic scrolling, backtracking."
    )
    exploration_style: Literal["scanner", "reader", "goal_directed", "wandering"] = Field(
        description="Overall navigation pattern for this segment."
    )
    price_sensitivity_signal: Literal["unknown", "low", "medium", "high"] = Field(
        description="Only if pricing-zone behavior gives evidence; otherwise 'unknown'."
    )
    friction_hotspot: Optional[str] = Field(
        default=None,
        description="Zone with the strongest struggle signal, or null.",
    )
    evidence: str = Field(
        description="One or two sentences citing the specific metrics behind these labels."
    )
    likely_mindset: str = Field(
        description="One in-character sentence describing the plausible mindset (used to seed twins)."
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="Honest overall confidence; thin data must mean low."
    )


# ---------------------------------------------------------------- persona seed


class Trait(BaseModel):
    score: float = Field(ge=0.0, le=1.0, description="0.5 = population baseline")
    confidence: float = Field(ge=0.0, le=1.0)


class TraitVector(BaseModel):
    """Soft Big Five persona seed. Derived by transparent rules, wide error bars.

    This parameterizes synthetic twins for simulation; it is explicitly NOT a
    psychometric assessment of any individual.
    """

    openness: Trait
    conscientiousness: Trait
    extraversion: Trait
    agreeableness: Trait
    neuroticism: Trait
    disclaimer: str = (
        "Soft simulation seed derived from aggregate interaction telemetry. "
        "Not a psychometric measurement."
    )


class PersonaSeed(BaseModel):
    persona_id: str
    label: str = Field(description="Human-readable segment name, e.g. 'deliberate price-checker'")
    signal: BehavioralSignal
    traits: TraitVector
    system_prompt: str = Field(description="Rendered twin conditioning prompt.")
    #: Where this persona came from. "observed" = derived from real behavioural
    #: telemetry. "inferred" = proposed from the stimulus when no telemetry
    #: existed. The distinction has to travel with the persona: a run against an
    #: imagined audience and one against a measured audience are different
    #: claims, and only the label stops a reader conflating them. Defaults to
    #: observed so personas stored before this existed keep their meaning.
    provenance: Literal["observed", "inferred"] = "observed"


class AudienceSegment(BaseModel):
    """One inferred audience segment. Supplies the CATEGORICAL signal only —
    trait numbers still come from persona.derive_traits, so the Big Five vector
    stays a documented function of a small input rather than something a model
    was free to invent."""

    label: str = Field(description="Short human name for the segment.")
    rationale: str = Field(description="Why this stimulus reaches them, citing something concrete.")
    likely_mindset: str = Field(description="One in-character sentence, present tense.")
    deliberation: Literal["low", "medium", "high"]
    frustration_signal: Literal["none", "mild", "elevated"]
    exploration_style: Literal["scanner", "reader", "goal_directed", "wandering"]
    price_sensitivity_signal: Literal["unknown", "low", "medium", "high"]
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="How strongly the stimulus supports this segment existing. Reading copy, not data.",
    )


class PageFetchRequest(BaseModel):
    """Resolve a URL to markup so a researcher can paste a link, not HTML.

    This is the ONE place a caller-supplied URL is accepted, and it stays
    separate from `ContentAsset` on purpose: the asset model remains
    URL-free, so no other code path can grow a fetch by accident. Everything
    dangerous about it is handled in `app.fetching`, which refuses anything
    that does not resolve to a publicly routable address.
    """

    url: str = Field(min_length=4, max_length=2048)


class AudienceComposeRequest(BaseModel):
    """Build or refine an audience panel from a plain-language instruction."""

    stimulus: str = Field(min_length=1, max_length=20_000)
    instruction: str = Field(
        default="",
        max_length=1_000,
        description="What to change, e.g. 'add price-conscious students'. Empty = infer from scratch.",
    )
    existing: list["PersonaSeed"] = Field(
        default_factory=list,
        max_length=8,
        description="The panel to refine. The FULL resulting panel is returned.",
    )


class AudienceInference(BaseModel):
    segments: list[AudienceSegment] = Field(description="3-5 behaviourally distinct segments.")


# ------------------------------------------------------- findings / question


class StudyRequest(BaseModel):
    """Base for every study: what the researcher is actually trying to find out.

    Until this field existed, no part of the pipeline knew what question a run
    was meant to answer, so no part of it could be specific to one. Every study
    computed the same battery and ended in the same template sentence, whether
    the researcher wanted to know why a signup flow leaks or whether an ad
    reads as threatening.

    It is optional, and stays optional. A blank question falls back to the one
    the study kind exists to answer, so nothing regresses for callers that do
    not set it.

    THE TWINS NEVER SEE IT, and that restriction is the point rather than an
    omission. Telling a synthetic audience "the researcher wants to know
    whether the pricing section creates anxiety" produces an audience that
    reports anxiety at the pricing section — the oldest failure mode in survey
    design, and one this platform is unusually exposed to because a language
    model is far more agreeable than a person. The question would contaminate
    the measurement it was asked to inform, and the resulting number would look
    exactly like a discovery.

    So it enters strictly downstream of measurement: it ranks the evidence and
    aims the synthesis. The numbers are identical whether or not a question was
    asked; what changes is which of them lead, and what they are read as
    meaning. That separation is what makes the answer specific without making
    it self-fulfilling.

    Not a prompt-injection surface in any interesting way: it is quoted into a
    system message as the researcher's stated goal, and the model's reply is
    schema-constrained and citation-validated against evidence harvested by
    code — so a hostile string can misdirect emphasis but cannot introduce a
    measurement, and cannot reach the twins at all.
    """

    research_question: str = Field(
        default="",
        max_length=2_000,
        description=(
            "What you want to know. e.g. 'Does the pricing section create "
            "anxiety for first-time buyers?' Steers twin attention and ranks "
            "the findings; blank falls back to the study's default question."
        ),
    )


class Finding(BaseModel):
    """One claim, tied to the measurements that license it."""

    headline: str = Field(
        description="One specific sentence. Name the beat/variant/step, not the category."
    )
    detail: str = Field(
        description="2-4 sentences: the mechanism, why it happens, what it implies."
    )
    evidence_ids: list[str] = Field(
        description="Ids from the evidence table. A finding with none is discarded."
    )
    direction: Literal["strength", "risk", "opportunity", "neutral"]
    confidence: Literal["high", "moderate", "low"]
    answers_question: bool = Field(
        description="True if this bears directly on what the researcher asked."
    )
    recommended_action: str = Field(
        description="Something doable to THIS asset. Empty string if none is warranted."
    )


class FindingsReport(BaseModel):
    answer: str = Field(
        description=(
            "Direct answer to the question asked, in 2-4 sentences. If the study "
            "cannot answer it, say that and say why."
        )
    )
    findings: list[Finding] = Field(description="Ordered by decision value, most consequential first.")
    what_would_change_the_answer: str = Field(
        description="The specific additional evidence that would move this conclusion."
    )
    limits: str = Field(
        description="What this design cannot establish, in the researcher's terms."
    )


# ---------------------------------------------------------------- swarm


class SwarmRunRequest(StudyRequest):
    scenario: str = Field(
        min_length=1,
        description=(
            "The stimulus, domain-agnostic: an ad variant, a video transcript with "
            "timestamps, a described UI flow, landing-page copy, etc."
        ),
    )
    session_ids: list[str] = Field(
        default_factory=list,
        description="Profiled sessions to seed personas from. Empty = use all profiled sessions.",
    )
    twins_per_persona: int = Field(default=3, ge=1, le=50)
    cognitive_load: Literal["low", "medium", "high"] = Field(
        default="low",
        description="Simulated cognitive state (fatigue/distraction) applied to every twin.",
    )


class TwinReaction(BaseModel):
    """Generic reaction schema — covers ad testing, content forecasting, and UX sims."""

    engagement: float = Field(ge=0.0, le=1.0, description="Sustained attention on the stimulus.")
    intent_score: float = Field(
        ge=0.0, le=1.0,
        description="Likelihood of the stimulus's target action (buy / click / keep watching / complete flow).",
    )
    action: Literal["convert", "continue", "hesitate", "abandon"]
    dropoff_point: Optional[str] = Field(
        default=None,
        description="Where attention or the flow was lost (timestamp, section, or UI step); null if none.",
    )
    friction_notes: str = Field(description="The single biggest friction this persona hit, in one sentence.")
    inner_monologue: str = Field(
        description="Brief in-character emotional appraisal written BEFORE deciding the action."
    )


class SwarmAggregate(BaseModel):
    run_id: str
    scenario_preview: str
    #: Twins whose reaction was actually USABLE. This is a survivor count, and
    #: for a long time it was the only count reported — so a run where 180 of
    #: 200 twins hit a 429 persisted as `twin_count: 20` and was byte-identical
    #: to a healthy 20-twin run, including in the calibration report. Read it
    #: together with `twins_requested` and `twins_failed`.
    twin_count: int
    twins_requested: Optional[int] = Field(
        default=None,
        description="Twin calls dispatched. None on runs recorded before this was tracked.",
    )
    twins_failed: Optional[int] = Field(
        default=None,
        description="Twin calls that errored or returned unusable output.",
    )
    cognitive_load: str
    mean_engagement: float
    mean_intent: float
    action_distribution: dict[str, int]
    top_dropoff_points: list[str]
    top_frictions: list[str]
    reactions: list[dict]
    # ── statistical layer (analytics.py) ─────────────────────────────
    intent_ci: Optional[list[float]] = Field(
        default=None, description="95% bootstrap confidence interval on mean intent."
    )
    engagement_ci: Optional[list[float]] = Field(default=None)
    polarization: float = Field(
        default=0.0,
        description="Normalised Shannon entropy of the action mix: 0 = unanimous, 1 = maximally split.",
    )
    bimodality: Optional[float] = Field(
        default=None,
        description="Sarle's coefficient on intent; > 0.555 indicates a genuinely split audience.",
    )
    consensus: str = Field(default="", description="Plain-language read of swarm agreement.")
    segments: Optional[dict] = Field(
        default=None,
        description=(
            "Audience segments discovered by k-means++ over (intent, engagement): "
            "{k, silhouette, groups: [{label, size, share, mean_intent, "
            "mean_engagement, dominant_action}]}. k is selected by silhouette and "
            "gated on practical centroid spread. None means the swarm is genuinely "
            "homogeneous — no invented segments."
        ),
    )


# ---------------------------------------------------------------- A/B compare


class ScenarioVariant(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    scenario: str = Field(min_length=1)


class CompareRequest(StudyRequest):
    variants: list[ScenarioVariant] = Field(min_length=2, max_length=8)
    session_ids: list[str] = Field(default_factory=list)
    #: A curated panel, supplied directly. Takes precedence over session_ids
    #: and over inference, so a researcher who has shaped an audience gets
    #: exactly that audience rather than one reconstructed per run.
    personas: list[PersonaSeed] = Field(default_factory=list, max_length=8)
    twins_per_persona: int = Field(default=3, ge=1, le=20)
    cognitive_load: Literal["low", "medium", "high"] = "low"
    adaptive: bool = Field(
        default=False,
        description=(
            "Batched Thompson-sampling allocation: waves of agents are routed to "
            "variants in proportion to their posterior probability of being best, "
            "with Bayesian early stopping once the leader's expected loss is "
            "inside the ROPE. Streaming endpoint only."
        ),
    )


class VariantRank(BaseModel):
    name: str
    mean_intent: float
    mean_engagement: float
    lift_vs_control_pct: Optional[float] = Field(
        default=None,
        description="Intent lift vs the first variant (the control), in percent. Null for the control itself.",
    )


class CompareResult(BaseModel):
    run_id: str
    cognitive_load: str
    twin_count_per_variant: int
    winner: str
    ranking: list[VariantRank] = Field(description="Sorted by mean_intent, best first.")
    variants: dict[str, SwarmAggregate]
    bayesian: Optional[dict] = Field(
        default=None,
        description=(
            "Beta-Binomial posterior analysis: per-arm P(best), 95% credible "
            "intervals and expected loss, plus whether the result is conclusive. "
            "A raw lift without this is not evidence."
        ),
    )


# ---------------------------------------------------------------- flow walkthrough


class WalkRequest(StudyRequest):
    steps: list[str] = Field(
        min_length=2,
        max_length=12,
        description="The flow, one described step per entry (screen, section, or moment).",
    )
    session_ids: list[str] = Field(default_factory=list)
    #: A curated panel, supplied directly. Takes precedence over session_ids
    #: and over inference, so a researcher who has shaped an audience gets
    #: exactly that audience rather than one reconstructed per run.
    personas: list[PersonaSeed] = Field(default_factory=list, max_length=8)
    twins_per_persona: int = Field(default=2, ge=1, le=10)
    cognitive_load: Literal["low", "medium", "high"] = "low"


class TwinStepReaction(BaseModel):
    """Per-step structured output while a twin walks a flow."""

    inner_monologue: str = Field(
        description="Brief in-character emotional appraisal of this step, written first."
    )
    engagement: float = Field(ge=0.0, le=1.0)
    friction_notes: str = Field(description="Biggest friction at this step, one sentence; 'none' if smooth.")
    action: Literal["continue", "hesitate", "abandon", "convert"] = Field(
        description="'hesitate' still advances but signals struggle; 'abandon'/'convert' end the walk."
    )


class WalkStepStat(BaseModel):
    step_index: int
    step: str
    entered: int
    abandoned: int
    converted: int
    # Optional because a step nobody reached has no engagement to report.
    # Coercing that to 0.0 made "unreached" indistinguishable from "everyone
    # hated it" — identical in the chart, opposite in meaning.
    mean_engagement: Optional[float] = None


class WalkAggregate(BaseModel):
    run_id: str
    twin_count: int
    cognitive_load: str
    context_window_steps: int = Field(
        description="Rolling memory window applied to each twin (the context throttle)."
    )
    completion_rate: float = Field(ge=0.0, le=1.0, description="Reached the end without abandoning.")
    conversion_rate: float = Field(ge=0.0, le=1.0)
    steps: list[WalkStepStat]
    walks: list[dict] = Field(description="Sample of per-twin step traces (capped).")
    survival: Optional[dict] = Field(
        default=None,
        description=(
            "Kaplan-Meier survival curve with per-step hazard rates and Greenwood "
            "confidence bands. `worst_step_index` is the true killer step — the one "
            "with the highest conditional drop risk, which raw counts get wrong."
        ),
    )


# ---------------------------------------------------------------- validation


class ActualsPayload(BaseModel):
    """Observed real-world outcomes for a run's stimulus, recorded post-hoc."""

    engagement: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    intent: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    note: Optional[str] = Field(default=None, max_length=512)


class MetricCalibration(BaseModel):
    """Calibration of predicted vs actual for one metric.

    MAE and bias alone cannot separate a systematically *compressed* predictor
    from a merely *noisy* one — a predictor with slope 0.15 and r = 1.00 and a
    calibrated one with slope 1.00 and r = 0.80 both post MAE 0.17 / bias 0.00.
    The regression terms below are what tell them apart, so they are reported
    together or not at all.

    The interval and reliability fields are the same principle applied one level
    up. `slope` is estimated from a handful of shipped runs, so a bare point
    estimate invites exactly the over-reading the verdict text refuses to make:
    the model carries `slope_ci` and `r_ci` because the verdict is gated on
    them, and a consumer that renders the adjective without the interval is
    showing a conclusion while hiding what licenses it. All of them are
    optional: they are None below n = 8, where an interval would be resampling
    noise rather than an uncertainty statement.
    """

    n: int
    mae: float = Field(description="Mean absolute error, predicted vs actual.")
    bias: float = Field(description="Mean(predicted - actual); positive = swarm over-predicts.")
    rmse: Optional[float] = Field(
        default=None, description="Root mean squared error — penalises large misses."
    )
    r: Optional[float] = Field(
        default=None, description="Pearson correlation of predicted with actual."
    )
    slope: Optional[float] = Field(
        default=None,
        description="Calibration slope from predicted ~ actual. <1 = compressed range.",
    )
    intercept: Optional[float] = Field(default=None, description="Calibration intercept.")
    slope_r2: Optional[float] = Field(default=None, description="R² of the calibration fit.")
    mae_ci: Optional[list[float]] = Field(default=None, description="95% bootstrap CI on MAE.")
    bias_ci: Optional[list[float]] = Field(default=None, description="95% bootstrap CI on bias.")
    slope_ci: Optional[list[float]] = Field(
        default=None,
        description=(
            "95% pairs-bootstrap CI on the calibration slope. The verdict is "
            "gated on this: 'compressed' is asserted only when the whole "
            "interval sits below 1. None below n=8."
        ),
    )
    r_ci: Optional[list[float]] = Field(
        default=None, description="95% pairs-bootstrap CI on Pearson r. None below n=8."
    )
    slope_bracket_ci: Optional[list[float]] = Field(
        default=None,
        description=(
            "The range of calibration slopes consistent with this data under ANY "
            "assumption about which axis is noisier — the actual arrives without "
            "replicate counts, so that ratio is unknowable and every point "
            "estimate encodes a guess about it. `verdict` is gated on this, not "
            "on slope_ci: an adjective survives only if no admissible ratio "
            "contradicts it. None below n=8."
        ),
    )
    reliability_curve: Optional[list[list[float]]] = Field(
        default=None,
        description=(
            "Isotonic (PAV) reliability diagram as [predicted, actual] knots — "
            "where along the range the predictor drifts, which a single slope "
            "cannot show. None below n=8."
        ),
    )
    brier_score: Optional[float] = Field(
        default=None, description="Mean squared error, heading the decomposition below."
    )
    brier_decomposition: Optional[dict[str, float]] = Field(
        default=None,
        description=(
            "CORP split of brier_score into miscalibration / discrimination / "
            "uncertainty. Separates 'wrong' from 'uninformative': a predictor "
            "can be perfectly calibrated and still useless. None below n=8."
        ),
    )
    verdict: Optional[str] = Field(
        default=None, description="Plain-language read of the calibration."
    )


class CalibrationReport(BaseModel):
    runs_with_actuals: int
    engagement: Optional[MetricCalibration]
    intent: Optional[MetricCalibration]


# ---------------------------------------------------------------- price sensitivity


class PriceSensitivityRequest(StudyRequest):
    product: str = Field(min_length=1, description="What's being sold, described in words.")
    prices: list[float] = Field(
        min_length=2, max_length=8, description="Candidate price points to test."
    )
    session_ids: list[str] = Field(default_factory=list)
    #: A curated panel, supplied directly. Takes precedence over session_ids
    #: and over inference, so a researcher who has shaped an audience gets
    #: exactly that audience rather than one reconstructed per run.
    personas: list[PersonaSeed] = Field(default_factory=list, max_length=8)
    twins_per_persona: int = Field(default=3, ge=1, le=20)
    cognitive_load: Literal["low", "medium", "high"] = "low"


class TwinPriceCurve(BaseModel):
    """Per-twin structured output: purchase intent at each price, same order as asked."""

    intents: list[float] = Field(
        description="Purchase intent 0..1 for each price, in the exact order given."
    )
    reasoning: str = Field(description="One sentence on what drove the price reactions.")


class PricePoint(BaseModel):
    price: float
    demand: float = Field(description="Mean purchase intent 0..1 across twins.")
    revenue_index: float = Field(description="price × demand, normalized so the best point = 1.0.")


class PriceSensitivityResult(BaseModel):
    run_id: str
    twin_count: int
    cognitive_load: str
    product_preview: str
    points: list[PricePoint]
    recommended_price: float = Field(description="The revenue-maximizing price point tested.")
    econometrics: Optional[dict] = Field(
        default=None,
        description=(
            "Price elasticity of demand from a log-log OLS fit (with R²), plus a "
            "linear-demand model whose revenue optimum is continuous rather than "
            "snapped to a tested price point."
        ),
    )


# ---------------------------------------------------------------- content assets


class TranscriptCue(BaseModel):
    t_ms: int = Field(ge=0, description="Cue start, milliseconds from content start.")
    text: str = Field(min_length=1, max_length=2_000)


class VideoFrame(BaseModel):
    t_ms: int = Field(ge=0, description="Keyframe timestamp, ms from start.")
    image_b64: str = Field(
        max_length=8_000_000,
        description="Base64 keyframe. Extracted client-side — see modality.py.",
    )
    media_type: Optional[str] = Field(default="image/jpeg", max_length=64)


class ContentAsset(BaseModel):
    """One piece of digital content, in whatever shape it actually exists.

    There is deliberately NO url field. Fetching a caller-supplied URL
    server-side is an SSRF primitive: it would let anyone with an API key reach
    the cloud metadata service, the private backend, or anything else routable
    from inside the network. The browser can fetch its own bytes.
    """

    kind: Literal["text", "image", "video", "audio", "page", "document"] = "text"
    #: Script, copy, transcript, or raw HTML depending on `kind`.
    text: Optional[str] = Field(default=None, max_length=200_000)
    content_type: Optional[str] = Field(default=None, max_length=64)
    brief: Optional[str] = Field(
        default=None,
        max_length=1_000,
        description="Optional context for the analyst, e.g. what the asset is for.",
    )
    image_b64: Optional[str] = Field(default=None, max_length=8_000_000)
    media_type: Optional[str] = Field(default=None, max_length=64)
    frames: list[VideoFrame] = Field(default_factory=list, max_length=32)
    transcript_cues: list[TranscriptCue] = Field(default_factory=list, max_length=2_000)
    pages: list[str] = Field(default_factory=list, max_length=64)


class ImageElement(BaseModel):
    label: str = Field(description="Short name for the element.")
    description: str = Field(description="What it shows and why the eye lands there.")


class ImageHierarchy(BaseModel):
    elements: list[ImageElement] = Field(
        description="3-6 attention regions in predicted scan order."
    )


class VideoFrameRead(BaseModel):
    description: str = Field(description="What is on screen and its emotional register.")


# ---------------------------------------------------------------- objection radar


class ContentStudyRequest(StudyRequest):
    """A neuro-impact study over any digital content.

    `asset` is the general path — text, image, video keyframes, audio
    transcript, a landing page, or a deck. `content`/`content_type` are the
    original text-only fields, kept so existing callers and stored runs keep
    working; when `asset` is absent they are treated as a text asset.
    """

    content: str = Field(
        default="",
        max_length=200_000,
        description="Script, transcript, storyboard, or copy. Ignored when `asset` is set.",
    )
    content_type: Literal["video_script", "ad_copy", "article", "storyboard"] = "video_script"
    asset: Optional[ContentAsset] = None
    session_ids: list[str] = Field(default_factory=list)
    #: A curated panel, supplied directly. Takes precedence over session_ids
    #: and over inference, so a researcher who has shaped an audience gets
    #: exactly that audience rather than one reconstructed per run.
    personas: list[PersonaSeed] = Field(default_factory=list, max_length=8)
    twins_per_persona: int = Field(default=3, ge=1, le=20)
    cognitive_load: Literal["low", "medium", "high"] = "low"


class ContentSegments(BaseModel):
    segments: list[str] = Field(description="Ordered narrative beats: 'Label — one-line summary'.")


class SegmentAppraisal(BaseModel):
    """One twin's momentary reaction to one content beat (appraisal theory)."""

    attention: float = Field(description="0 ignored .. 1 riveted")
    valence: float = Field(description="-1 repelled .. +1 delighted")
    arousal: float = Field(description="0 calm .. 1 activated")
    novelty: float = Field(description="0 seen before .. 1 surprising")
    goal_relevance: float = Field(description="0 irrelevant .. 1 exactly my problem")
    social_resonance: float = Field(description="0 no urge to discuss .. 1 must tell someone")
    threat: float = Field(description="0 safe .. 1 uneasy")
    reward_anticipation: float = Field(description="0 nothing promised .. 1 can't wait")
    cognitive_effort: float = Field(description="0 effortless .. 1 confusing")


class TwinContentResponse(BaseModel):
    appraisals: list[SegmentAppraisal] = Field(description="Exactly one per beat, in order.")
    gist: str = Field(description="One line: what this persona took away.")
    would_share: float = Field(description="0..1 likelihood of sharing/discussing it.")


class ViralityRequest(StudyRequest):
    content: str = Field(min_length=20, description="The shareable asset (post, ad, video script).")
    fanout: int = Field(default=6, ge=1, le=25, description="Contacts exposed per sharer (k).")
    session_ids: list[str] = Field(default_factory=list)
    #: A curated panel, supplied directly. Takes precedence over session_ids
    #: and over inference, so a researcher who has shaped an audience gets
    #: exactly that audience rather than one reconstructed per run.
    personas: list[PersonaSeed] = Field(default_factory=list, max_length=8)
    twins_per_persona: int = Field(default=3, ge=1, le=20)
    cognitive_load: Literal["low", "medium", "high"] = "low"


class TwinShare(BaseModel):
    would_share: float = Field(description="0..1 probability this persona actually shares it.")
    reason: str = Field(description="The single real reason, one line.")


class ObjectionRequest(StudyRequest):
    pitch: str = Field(min_length=1, description="The claim / pitch / copy to stress-test.")
    session_ids: list[str] = Field(default_factory=list)
    #: A curated panel, supplied directly. Takes precedence over session_ids
    #: and over inference, so a researcher who has shaped an audience gets
    #: exactly that audience rather than one reconstructed per run.
    personas: list[PersonaSeed] = Field(default_factory=list, max_length=8)
    twins_per_persona: int = Field(default=3, ge=1, le=20)
    cognitive_load: Literal["low", "medium", "high"] = "low"


class TwinObjection(BaseModel):
    sentiment: Literal["positive", "neutral", "negative"]
    biggest_objection: str = Field(
        description="The single biggest reason this persona hesitates or says no; 'none' if fully sold."
    )
    is_dealbreaker: bool = Field(description="True if that objection alone would stop them.")
    would_recommend: float = Field(description="Likelihood 0..1 they'd recommend it to a friend.")


class ObjectionTheme(BaseModel):
    objection: str = Field(description="Medoid phrasing — the most representative wording in the cluster.")
    count: int
    dealbreaker_count: int
    variants: list[str] = Field(
        default_factory=list,
        description="Other phrasings merged into this theme by semantic clustering.",
    )


class ObjectionResult(BaseModel):
    run_id: str
    twin_count: int
    cognitive_load: str
    pitch_preview: str
    sentiment_distribution: dict[str, int]
    dealbreaker_rate: float
    mean_recommend: float
    top_objections: list[ObjectionTheme]
    clustering_method: str = Field(
        default="exact",
        description="'semantic' when embeddings clustered the objections; 'exact' on fallback.",
    )
    recommend_ci: Optional[list[float]] = Field(
        default=None, description="95% bootstrap CI on mean recommendation likelihood."
    )
    cluster_silhouette: Optional[float] = Field(
        default=None,
        description="Mean silhouette of the chosen clustering (−1..1; higher = cleaner themes).",
    )
    cluster_threshold: Optional[float] = Field(
        default=None,
        description="Merge threshold selected by silhouette maximisation.",
    )
    theme_map: Optional[list[dict]] = Field(
        default=None,
        description=(
            "2-D kernel-PCA projection of the objection embedding cloud: each "
            "theme's {x, y, count, dealbreaker, label}, scaled to [-1, 1]. "
            "Positions are semantic — nearby themes are similar complaints."
        ),
    )


# ---------------------------------------------------------------- copy optimizer


class CopyOptimizerRequest(StudyRequest):
    """An evolutionary search over copy, not a one-shot score.

    Cost is population_size × generations × (personas × twins_per_persona)
    twin calls plus one breeder call per generation, so the bounds are
    deliberately tight — this is the most expensive study in the product.
    """

    brief: str = Field(
        min_length=1,
        max_length=4000,
        description=(
            "The creative brief: what the copy must achieve, for whom, and the "
            "hard constraints (claims allowed, tone, length, channel). The "
            "breeder sees this; the audience twins deliberately do not."
        ),
    )
    seed_variant: str = Field(
        min_length=1,
        max_length=4000,
        description="The copy you have today. Generation 0, and the baseline every mutant must beat.",
    )
    population_size: int = Field(default=4, ge=2, le=8)
    generations: int = Field(default=3, ge=1, le=5)
    session_ids: list[str] = Field(default_factory=list)
    #: A curated panel, supplied directly. Takes precedence over session_ids
    #: and over inference, so a researcher who has shaped an audience gets
    #: exactly that audience rather than one reconstructed per run.
    personas: list[PersonaSeed] = Field(default_factory=list, max_length=8)
    twins_per_persona: int = Field(default=2, ge=1, le=10)
    cognitive_load: Literal["low", "medium", "high"] = "low"
    min_evals_per_variant: int = Field(
        default=2,
        ge=1,
        le=20,
        description=(
            "Forced-exploration floor per variant per generation, spent before "
            "Thompson sampling allocates the rest. Below 2 a variant's posterior "
            "is a coin flip and the bandit can starve a good variant on one "
            "unlucky draw."
        ),
    )


class TwinCopyScore(BaseModel):
    """One twin's reaction to one variant. No bounds on the floats: an
    out-of-range value should be clamped, not thrown away with the whole twin."""

    intent: float = Field(description="0..1 likelihood this persona takes the action the copy asks for.")
    would_act: bool = Field(description="True only if they'd actually do it now (intent above 0.5).")
    strongest_phrase: str = Field(
        description="The phrase that landed hardest, quoted from the copy — feeds the breeder."
    )
    reaction: str = Field(description="One line, in character, on why.")


class CopyMutant(BaseModel):
    """One child produced by the breeder call."""

    text: str = Field(description="The new variant's full copy.")
    parent_ids: list[str] = Field(
        description="Ids of the variants this was bred from: two or more for a crossover, one for a mutation."
    )
    operation: Literal["crossover", "mutation"]
    rationale: str = Field(description="One line: what was taken from where, or what single thing was changed.")


class CopyMutantBatch(BaseModel):
    variants: list[CopyMutant] = Field(description="The next generation, exactly as many as requested.")


# ---------------------------------------------------------------- message sequence


class SequenceRequest(StudyRequest):
    """Order effects: which ORDER of these messages ends with the most intent."""

    messages: list[str] = Field(
        min_length=2,
        max_length=10,
        description=(
            "The messages to be delivered in sequence (drip emails, onboarding "
            "steps, pitch-deck sections), in the order you currently plan to "
            "send them. That submitted order is the baseline."
        ),
    )
    session_ids: list[str] = Field(default_factory=list)
    #: A curated panel, supplied directly. Takes precedence over session_ids
    #: and over inference, so a researcher who has shaped an audience gets
    #: exactly that audience rather than one reconstructed per run.
    personas: list[PersonaSeed] = Field(default_factory=list, max_length=8)
    twins_per_persona: int = Field(default=3, ge=1, le=20)
    cognitive_load: Literal["low", "medium", "high"] = "low"
    orderings_sampled: int = Field(
        default=6,
        ge=2,
        le=24,
        description=(
            "How many distinct orderings to actually walk. N! is intractable, so "
            "the precedence matrix is estimated from this sample and the best "
            "global ordering is then solved for. Capped at the number of twins — "
            "an ordering nobody walked contributes nothing."
        ),
    )


class TwinSequenceStep(BaseModel):
    """One twin's state after one message. Unbounded floats, clamped on receipt."""

    intent: float = Field(description="0..1 likelihood of acting at this moment.")
    fatigue: float = Field(description="0 fresh .. 1 done with this sequence.")
    disengaged: bool = Field(description="True the moment this persona would stop reading the rest.")
    reaction: str = Field(description="One line, in character.")


class TwinSequenceWalk(BaseModel):
    steps: list[TwinSequenceStep] = Field(
        description="Exactly one step per message, in the delivered order."
    )
    gist: str = Field(description="One line: what this persona was left with at the end.")


# ══════════════════════════════ the white room ══════════════════════════════
#
# Every other study in this product fans out to ISOLATED twins on purpose: no
# inter-agent chatter, so the spread you read is genuine disagreement rather
# than a herd. The white room deliberately breaks that isolation, which makes it
# a different instrument answering a different question — not "what does the
# audience think" but "what happens to what they think when they can hear each
# other".
#
# That difference is the whole design. Because the isolated condition already
# exists, the room can run BOTH on the same cast and report the delta, which is
# a within-subjects experiment with the control arm built in. Deliberation is
# the treatment; the private round-0 ballot is the control.


class RoomMemberSpec(BaseModel):
    """One character at the table.

    The disposition numbers are DECLARED, never measured — a model was asked to
    instantiate a character from prose and these are what it asserted. They are
    used as the room's *seating* parameters (who speaks when, who sees whom) and
    as priors the analysis is free to contradict: `openness` is what the
    character claims about itself, while `susceptibility` in the fitted results
    is what its actual behaviour showed. Keeping both, and never overwriting one
    with the other, is what lets the report say "the CFO described himself as
    open and moved less than anyone in the room".
    """

    name: str = Field(description="Short name, as it would appear on a place card.")
    role: str = Field(description="Their job or standing in this room.")
    archetype: str = Field(description="One line: what they are, in the abstract.")
    stake: str = Field(description="What they personally stand to gain or lose here.")
    system_prompt: str = Field(
        description="Second-person conditioning prompt. Voice, history, priors, what they fear."
    )
    openness: float = Field(description="0 immovable .. 1 persuaded by anything. Declared.")
    assertiveness: float = Field(description="0 speaks only if asked .. 1 dominates. Declared.")
    seniority: float = Field(description="0 junior .. 1 the room defers by default. Declared.")
    risk_appetite: float = Field(description="0 protects downside .. 1 chases upside. Declared.")
    expertise: list[str] = Field(description="Domains they are actually credible in.")


class RoomCast(BaseModel):
    members: list[RoomMemberSpec]


class RoomTurn(BaseModel):
    """What one member does on one round.

    THE PUBLIC/PRIVATE SPLIT IS THE POINT. `public_statement` is what they say
    at the table and every other member sees it; `private_position` is never
    shown to anyone, including in the next round's context. The gap between them
    is preference falsification (Kuran), and it is the one thing a real boardroom
    cannot measure about itself — everyone knows their own gap and nobody knows
    the room's.
    """

    public_statement: str = Field(description="What they say out loud. In character, 1-3 sentences.")
    public_position: float = Field(
        description="0 fully against the motion .. 1 fully for it — the position their WORDS convey."
    )
    private_position: float = Field(
        description="0..1 what they actually believe now. May differ from public_position."
    )
    private_note: str = Field(
        description="One line, unspoken. Why they said what they said, if it differs from what they think."
    )
    conceded_to: list[str] = Field(
        description="Names whose argument genuinely moved them. Self-reported; a diagnostic, never the influence estimate."
    )


class RoomRequest(StudyRequest):
    """Convene a room and put a motion to it."""

    motion: str = Field(
        min_length=10,
        max_length=4_000,
        description="The situation, decision or scenario the room is reacting to.",
    )
    cast_brief: str = Field(
        default="",
        max_length=4_000,
        description=(
            "Describe the characters in prose. Blank means infer a plausible "
            "room from the motion itself, which is marked as inferred."
        ),
    )
    members: list[RoomMemberSpec] = Field(
        default_factory=list,
        max_length=9,
        description="A cast the researcher already approved. Wins over cast_brief.",
    )
    seats: int = Field(default=5, ge=3, le=9, description="How many characters to cast.")
    rounds: int = Field(
        default=4,
        ge=2,
        le=8,
        description=(
            "Deliberation rounds after the private opening ballot. The influence "
            "matrix needs enough updates per member to be identifiable — see "
            "room.fit_social_influence, which refuses rather than regularising "
            "an underdetermined fit into a confident-looking answer."
        ),
    )
    replicates: int = Field(
        default=5,
        ge=1,
        le=8,
        description=(
            "Independent runs of the same room, each with a different seeded "
            "speaking order. Order is a confound, not a setting: one run cannot "
            "tell a real convergence from an artefact of who happened to speak "
            "first, and replicates are also what make the influence fit "
            "identifiable at realistic round counts. THE CHEAP AXIS — each "
            "replicate buys (rounds − 1) fresh observations per member, so 5 "
            "replicates identify more than 8 rounds do, at lower cost."
        ),
    )
    placebo_replicates: int = Field(
        default=1,
        ge=0,
        le=4,
        description=(
            "Runs where the statements each member is shown come from a "
            "DIFFERENT room. The critical control: a language model is "
            "agreeable, so convergence and a public/private gap can both be "
            "artefacts of its disposition rather than anything about these "
            "characters. If the placebo arm converges as hard, the deliberation "
            "measured nothing, and only this arm can tell you."
        ),
    )
    decision_threshold: Optional[float] = Field(
        default=None,
        description="If set, the position above which a member counts as voting FOR — enables coalition power indices.",
    )

    #: WHAT EACH SETTING CAN AND CANNOT SUPPORT — measured on synthetic rooms
    #: with a known influence matrix, not asserted. Read this before changing a
    #: default; several of these limits are structural and no amount of
    #: deliberation quality gets around them.
    #:
    #: The influence fit needs ≥1.5 observations per free parameter, where
    #: observations = replicates × (rounds − 1) and parameters = seats + 1.
    #: Below that the fitted weights lose to a uniform guess outright. The
    #: previous default of 3 replicates at 5 seats and 4 rounds sat at exactly
    #: 1.50 — on the refusal boundary, so one dropped replicate refused the
    #: whole fit. 5 replicates gives 2.50.
    #:
    #: A SINGLE W ENTRY is never quotable. An individual weight only beats a
    #: uniform-row guess above 9 observations per parameter, and the schema
    #: maximum — 9 seats × 8 rounds × 8 replicates — reaches 6.22. Only a
    #: 5-seat room at 8 rounds × 8 replicates (11.2) clears it. That is why
    #: `w_entries_supported` ships False on essentially every real run and why
    #: the network is drawn as a pattern rather than labelled with numbers.
    #: What DOES recover is everything derived from the whole matrix:
    #: equilibrium MAE 0.019 against a 0.085 baseline with rank correlation
    #: 0.96-0.99, which is why the counterfactuals are trustworthy while the
    #: edges they are computed from are not individually.
    #:
    #: CENTRALITY IS THE WEAKEST HEADLINE. Identifying the single most central
    #: member correctly runs 31-38% at 5 seats × 4 rounds × 3 replicates against
    #: a 20% chance rate, rising to 49-57% at 8 rounds × 8 replicates. Read it
    #: as a ranking with real uncertainty, never as "this person ran the room".
    #:
    #: TWO p-VALUE FLOORS ARE COMBINATORIAL. A permutation test cannot report a
    #: p below 2/(number of arrangements): the public/private gap floors at
    #: 2/2^seats, so a 5-seat room cannot reach p < 0.05 (floor 0.0625) however
    #: large the gap; and the speaking-order effect floors at 2/replicates!, so
    #: 3 replicates floor at 0.333 and 5 are the minimum that can reach 0.05.
    #: Both floors are reported next to their p-values.
    #:
    #: CONFORMITY power comes from replicates, not rounds — 48% → 91% going from
    #: 3 to 8 replicates, while 4 → 8 rounds changes nothing. Its false-positive
    #: rate is 2-7% against a nominal 5% and flat across designs.
