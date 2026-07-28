import pytest

from app.swarm import aggregate_walks

STEPS = ["Landing hero", "Signup form", "Checkout"]


def _record(i, action, engagement=0.5):
    return {
        "step_index": i,
        "step": STEPS[i],
        "action": action,
        "engagement": engagement,
        "friction_notes": "none",
        "inner_monologue": "…",
    }


def test_aggregate_walks_funnel():
    walks = [
        {  # abandons at the form
            "persona_label": "a",
            "outcome": "abandoned",
            "records": [_record(0, "continue", 0.8), _record(1, "abandon", 0.2)],
        },
        {  # converts at checkout
            "persona_label": "b",
            "outcome": "converted",
            "records": [
                _record(0, "continue", 0.6),
                _record(1, "hesitate", 0.4),
                _record(2, "convert", 0.9),
            ],
        },
        {  # completes without converting
            "persona_label": "c",
            "outcome": "completed",
            "records": [
                _record(0, "continue"),
                _record(1, "continue"),
                _record(2, "continue"),
            ],
        },
    ]
    agg = aggregate_walks(walks, STEPS, "high", 2)

    assert agg.twin_count == 3
    assert agg.completion_rate == pytest.approx(2 / 3, abs=1e-3)
    assert agg.conversion_rate == pytest.approx(1 / 3, abs=1e-3)
    assert agg.context_window_steps == 2

    entered = [s.entered for s in agg.steps]
    assert entered == [3, 3, 2]  # one twin never reaches checkout
    assert agg.steps[1].abandoned == 1
    assert agg.steps[2].converted == 1
    assert agg.steps[0].mean_engagement == pytest.approx((0.8 + 0.6 + 0.5) / 3, abs=1e-3)


def test_aggregate_walks_empty_raises():
    with pytest.raises(RuntimeError):
        aggregate_walks([], STEPS, "low", 20)
