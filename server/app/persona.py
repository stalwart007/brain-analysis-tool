"""BehavioralSignal -> PersonaSeed.

The Big Five vector is derived by TRANSPARENT, auditable rules — not by the LLM.
Every mapping below is a documented modeling assumption with deliberately wide
confidence bands, capped by the upstream signal's own confidence. The vector's
only job is to give synthetic twins a consistent, domain-agnostic vocabulary
(the same seed works for ad testing, content forecasting, and UX simulation).
"""

import uuid

from .schemas import BehavioralSignal, PersonaSeed, Trait, TraitVector

BASELINE = 0.5  # population prior; traits move away from it only with evidence

# How far a single behavioral signal is allowed to push a trait from baseline.
_MAX_SHIFT = 0.25
# Trait confidence can never exceed this fraction of the signal's confidence —
# behavior->trait inference is inherently lossy and we say so.
_CONFIDENCE_DISCOUNT = 0.6

# Confidence bands, as fractions of the maximum achievable trait confidence
# (which is _CONFIDENCE_DISCOUNT: signal confidence 1.0 at evidence strength
# 1.0). Expressed relatively so they track the discount instead of silently
# decoupling from it if that constant is ever retuned.
_BAND_STRONG = 0.60 * _CONFIDENCE_DISCOUNT   # 0.36
_BAND_WEAK = 0.30 * _CONFIDENCE_DISCOUNT     # 0.18

# The licence a twin has to act on each trait, by band. This is the payload of
# the confidence number: a trait we barely evidenced must not steer behaviour
# with the same force as one we measured directly.
_CONFIDENCE_HEDGE = {
    "strong": "well evidenced — let this drive your reactions",
    "moderate": "moderately evidenced — a real leaning, not a rule",
    "weak": (
        "barely evidenced — treat as roughly average unless the stimulus "
        "forces the question"
    ),
}


def _trait(shift: float, signal_confidence: float, evidence_strength: float) -> Trait:
    score = min(1.0, max(0.0, BASELINE + shift * evidence_strength))
    confidence = round(signal_confidence * _CONFIDENCE_DISCOUNT * evidence_strength, 3)
    return Trait(score=round(score, 3), confidence=confidence)


def derive_traits(signal: BehavioralSignal) -> TraitVector:
    c = signal.confidence

    # Conscientiousness ~ deliberation (careful processing, methodical dwell).
    deliberation_shift = {"low": -_MAX_SHIFT, "medium": 0.0, "high": _MAX_SHIFT}[
        signal.deliberation
    ]
    # Neuroticism ~ stress reactivity proxied by frustration expression.
    frustration_shift = {"none": -0.1, "mild": 0.1, "elevated": _MAX_SHIFT}[
        signal.frustration_signal
    ]
    # Openness ~ exploratory navigation vs goal-locked behavior.
    openness_shift = {
        "wandering": _MAX_SHIFT,
        "scanner": 0.1,
        "reader": 0.0,
        "goal_directed": -0.1,
    }[signal.exploration_style]

    return TraitVector(
        openness=_trait(openness_shift, c, 0.8),
        conscientiousness=_trait(deliberation_shift, c, 1.0),
        # No defensible behavioral proxy from this telemetry -> baseline, near-zero confidence.
        extraversion=_trait(0.0, c, 0.1),
        agreeableness=_trait(0.0, c, 0.1),
        neuroticism=_trait(frustration_shift, c, 0.9),
    )


def _label(signal: BehavioralSignal) -> str:
    style = signal.exploration_style.replace("_", "-")
    mood = {
        "none": "calm",
        "mild": "impatient",
        "elevated": "frustrated",
    }[signal.frustration_signal]
    price = (
        f", {signal.price_sensitivity_signal} price sensitivity"
        if signal.price_sensitivity_signal != "unknown"
        else ""
    )
    return f"{mood} {style} ({signal.deliberation} deliberation{price})"


def confidence_band(confidence: float) -> str:
    """Which evidential band a trait confidence falls in: strong/moderate/weak."""
    if confidence >= _BAND_STRONG:
        return "strong"
    if confidence >= _BAND_WEAK:
        return "moderate"
    return "weak"


def _render_trait(name: str, trait: Trait) -> str:
    band = confidence_band(trait.confidence)
    return f"- {name}={trait.score} [{_CONFIDENCE_HEDGE[band]}; confidence {trait.confidence}]"


def _render_system_prompt(signal: BehavioralSignal, traits: TraitVector) -> str:
    """Render the twin's system prompt, INCLUDING each trait's confidence band.

    The confidence has to reach the prompt or the whole trait vector is
    dishonest. Emitting a bare `openness=0.7, conscientiousness=0.75,
    neuroticism=0.725` lets a trait inferred from a 0.1-confidence signal
    (trait confidence 0.06 — essentially a coin flip dressed as a measurement)
    steer the twin exactly as hard as one inferred from a 0.9-confidence
    signal. That would erase the module's entire justification: the docstring
    above stakes its honesty on "deliberately wide confidence bands", and a
    confidence band no consumer reads is not a limitation, it is decoration.

    The bands are attached inline as behavioural licence rather than as a bare
    number, because a number in a prompt is not an instruction — a model shown
    `confidence=0.06` has no calibrated notion of how much to discount, whereas
    "barely evidenced — treat as roughly average" is actionable. Extraversion
    and agreeableness are included now precisely *because* they are pinned at
    baseline with near-zero confidence: stating that we have no read on them is
    more informative than omitting them and letting the model invent one.
    """
    trait_lines = "\n".join(
        _render_trait(name, trait)
        for name, trait in (
            ("openness", traits.openness),
            ("conscientiousness", traits.conscientiousness),
            ("extraversion", traits.extraversion),
            ("agreeableness", traits.agreeableness),
            ("neuroticism", traits.neuroticism),
        )
    )
    return (
        "You are a synthetic research persona in a simulation. You are NOT an "
        "assistant here: you react in character, imperfectly and humanly — real "
        "people skim, get bored, misread things, and abandon flows.\n\n"
        f"Behavioral profile: deliberation={signal.deliberation}, "
        f"frustration={signal.frustration_signal}, "
        f"exploration={signal.exploration_style}, "
        f"price_sensitivity={signal.price_sensitivity_signal}.\n"
        "Trait tendencies (0-1, 0.5=average). Each carries how much evidence "
        "actually supports it — weight your behaviour accordingly:\n"
        f"{trait_lines}\n"
        f"Current mindset: {signal.likely_mindset}\n\n"
        "Stay in character. Write your inner monologue first, then decide."
    )


def seed_persona(signal: BehavioralSignal) -> PersonaSeed:
    traits = derive_traits(signal)
    return PersonaSeed(
        persona_id=uuid.uuid4().hex[:10],
        label=_label(signal),
        signal=signal,
        traits=traits,
        system_prompt=_render_system_prompt(signal, traits),
    )
