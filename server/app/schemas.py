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


# ---------------------------------------------------------------- swarm


class SwarmRunRequest(BaseModel):
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


class CompareRequest(BaseModel):
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


class WalkRequest(BaseModel):
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
    verdict: Optional[str] = Field(
        default=None, description="Plain-language read of the calibration."
    )


class CalibrationReport(BaseModel):
    runs_with_actuals: int
    engagement: Optional[MetricCalibration]
    intent: Optional[MetricCalibration]


# ---------------------------------------------------------------- price sensitivity


class PriceSensitivityRequest(BaseModel):
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


class ContentStudyRequest(BaseModel):
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


class ViralityRequest(BaseModel):
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


class ObjectionRequest(BaseModel):
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


class CopyOptimizerRequest(BaseModel):
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


class SequenceRequest(BaseModel):
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
