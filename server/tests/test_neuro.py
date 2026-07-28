"""Known-answer tests for the neuro-impact math (ISC, change points,
peak-end memory, trajectory, functional systems, aggregation)."""

import math
import random

import pytest

from app.neuro import (
    aggregate_content_study,
    change_points,
    emotional_trajectory,
    functional_systems,
    intersubject_correlation,
    peak_end_memory,
    pearson,
)
from app.schemas import SegmentAppraisal, TwinContentResponse


def test_pearson_known_values():
    assert pearson([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)
    assert pearson([1, 2, 3], [6, 4, 2]) == pytest.approx(-1.0)
    assert pearson([1, 1, 1], [1, 2, 3]) is None  # zero variance


def test_isc_identical_curves_lock_in():
    base = [0.2, 0.8, 0.5, 0.9, 0.3]
    isc = intersubject_correlation([base] * 5)
    assert isc is not None
    assert isc["overall"] == pytest.approx(1.0)
    assert isc["grip"] == "locked-in"


def test_isc_uncorrelated_curves_drift():
    rng = random.Random(4)
    curves = [[rng.random() for _ in range(12)] for _ in range(8)]
    isc = intersubject_correlation(curves)
    assert isc is not None
    assert abs(isc["overall"]) < 0.3
    assert isc["grip"] == "drifting"


def test_change_points_find_planted_step():
    xs = [0.9] * 10 + [0.2] * 10  # attention collapses at index 10
    cps = change_points(xs)
    assert cps == [10]


def test_change_points_none_in_flat_series():
    assert change_points([0.5] * 20) == []


def test_peak_end_memory_weights_peak_and_ending():
    arousal = [0.2, 0.9, 0.3, 0.2, 0.4]
    valence = [0.0, 0.8, 0.1, 0.0, -0.5]
    m = peak_end_memory(arousal, valence)
    assert m is not None
    assert m["peak_index"] == 1
    # remembered affect = mean(valence@peak, final valence) = (0.8 - 0.5)/2
    assert m["remembered_affect"] == pytest.approx(0.15)
    rp = m["recall_probability"]
    assert sum(rp) == pytest.approx(1.0, abs=1e-3)
    # peak and ending must out-recall the forgettable middle
    assert rp[1] > rp[2] and rp[4] > rp[3]


def test_trajectory_dynamism_flat_vs_rollercoaster():
    flat = emotional_trajectory([0.1] * 6, [0.3] * 6)
    ride = emotional_trajectory([-0.8, 0.8, -0.8, 0.8, -0.8, 0.8], [0.9, 0.2, 0.9, 0.2, 0.9, 0.2])
    assert flat is not None and ride is not None
    assert flat["dynamism"] == pytest.approx(0.0)
    assert ride["dynamism"] > 5.0
    assert flat["quadrant_occupancy"].get("content", 0) == 1.0


def test_functional_systems_bounds_and_mapping():
    s = functional_systems(
        {
            "attention": 1.0, "valence": 1.0, "arousal": 0.0, "novelty": 1.0,
            "goal_relevance": 1.0, "social_resonance": 0.9, "threat": 0.0,
            "reward_anticipation": 1.0, "cognitive_effort": 0.0,
        },
        memorability=0.7,
    )
    assert s["salience"] == pytest.approx(1.0)
    assert s["reward"] == pytest.approx(1.0)
    assert s["threat"] == pytest.approx(0.0)
    assert s["social"] == pytest.approx(0.9)
    assert all(0.0 <= v <= 1.0 for v in s.values())


def _twin(curve_attention, curve_valence, curve_arousal, share=0.5):
    apps = [
        SegmentAppraisal(
            attention=a, valence=v, arousal=r, novelty=0.5, goal_relevance=0.5,
            social_resonance=0.4, threat=0.1, reward_anticipation=0.5,
            cognitive_effort=0.3,
        )
        for a, v, r in zip(curve_attention, curve_valence, curve_arousal)
    ]
    return TwinContentResponse(appraisals=apps, gist="fine", would_share=share)


def test_aggregate_content_study_end_to_end():
    """Synchronised audience with an attention collapse at beat 3 and an
    emotional peak at beat 2: every analysis layer must find its feature."""
    segs = ["Hook", "Reveal", "Feature dump", "CTA"]
    rng = random.Random(6)
    twins = [
        _twin(
            [0.9 + rng.uniform(-0.05, 0.05), 0.95 + rng.uniform(-0.04, 0.04),
             0.25 + rng.uniform(-0.05, 0.05), 0.5 + rng.uniform(-0.05, 0.05)],
            [0.3, 0.8, 0.0, 0.4],
            [0.5, 0.9, 0.2, 0.4],
            share=0.7,
        )
        for _ in range(9)
    ]
    r = aggregate_content_study(twins, segs, "low")
    assert r["isc"]["overall"] > 0.5  # synchronized audience
    assert 2 in r["change_points"]  # the collapse at beat index 2
    assert r["memory"]["peak_index"] == 1  # the emotional peak
    assert r["systems"]["salience"] > 0.5
    assert len(r["per_segment_systems"]) == 4
    assert "weakest beat is 3" in r["verdict"]
    assert r["share_intent_mean"] == pytest.approx(0.7)
