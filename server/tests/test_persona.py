from app.persona import derive_traits, seed_persona
from app.schemas import BehavioralSignal


def _signal(**overrides) -> BehavioralSignal:
    base = dict(
        deliberation="high",
        frustration_signal="elevated",
        exploration_style="goal_directed",
        price_sensitivity_signal="high",
        friction_hotspot="pricing-table",
        evidence="12 micro-hesitations and 3 rage-click bursts over pricing zone.",
        likely_mindset="I want the price, and this page is annoying me.",
        confidence=0.8,
    )
    base.update(overrides)
    return BehavioralSignal(**base)


def test_traits_move_with_signals():
    traits = derive_traits(_signal())
    assert traits.conscientiousness.score > 0.5  # high deliberation
    assert traits.neuroticism.score > 0.5  # elevated frustration
    # No behavioral proxy for these -> pinned near baseline with negligible confidence
    assert traits.extraversion.score == 0.5
    assert traits.extraversion.confidence < 0.1


def test_trait_confidence_capped_by_signal_confidence():
    lo = derive_traits(_signal(confidence=0.2))
    hi = derive_traits(_signal(confidence=0.9))
    assert lo.conscientiousness.confidence < hi.conscientiousness.confidence
    # discount: trait confidence must never exceed 60% of signal confidence
    assert hi.conscientiousness.confidence <= 0.9 * 0.6 + 1e-9


def test_low_confidence_signal_stays_near_baseline_in_confidence_not_score():
    traits = derive_traits(_signal(confidence=0.1))
    assert traits.neuroticism.confidence <= 0.06


def test_seed_persona_renders_prompt_and_label():
    seed = seed_persona(_signal())
    assert seed.persona_id
    assert "frustrated" in seed.label
    assert "synthetic research persona" in seed.system_prompt
    assert "Not a psychometric measurement" in seed.traits.disclaimer
