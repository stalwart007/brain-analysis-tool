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


# ── peak-end must be within-subject, and retention must be a hazard ───────


class _Appraisal:
    def __init__(self, **kw):
        for d in (
            "attention", "valence", "arousal", "novelty", "goal_relevance",
            "social_resonance", "threat", "reward_anticipation", "cognitive_effort",
        ):
            setattr(self, d, kw.get(d, 0.5))


class _Resp:
    def __init__(self, appraisals):
        self.appraisals = appraisals
        self.would_share = 0.5
        self.gist = "g"


def test_peak_end_is_computed_within_twin_not_on_the_mean_curve():
    """The ecological fallacy this replaces. Two audience camps: half peak on
    beat 0, half on beat 2. The MEAN arousal curve peaks on beat 1 — a beat
    NEITHER camp peaked on — so remembered_affect was the valence of a moment
    nobody had a peak experience of.
    """
    from app.neuro import peak_end_memory, peak_end_per_twin

    early = _Resp([_Appraisal(arousal=a, valence=v) for a, v in [(0.9, 0.8), (0.5, 0.0), (0.1, -0.8)]])
    late = _Resp([_Appraisal(arousal=a, valence=v) for a, v in [(0.1, 0.8), (0.5, 0.0), (0.9, -0.8)]])
    responses = [early, early, late, late]

    # the mean curve is flat-topped and peaks in the middle
    mean_arousal = [0.5, 0.5, 0.5]
    mean_valence = [0.8, 0.0, -0.8]
    naive = peak_end_memory(mean_arousal, mean_valence)
    assert naive is not None and naive["peak_index"] == 0  # argmax of ties

    per = peak_end_per_twin(responses)
    assert per is not None
    # the distribution shows BOTH camps, which the mean curve destroyed
    assert per["peak_index_distribution"] == [2, 0, 2]
    assert per["peak_agreement"] == 0.5
    assert per["n_twins"] == 4


def test_peak_agreement_is_one_when_the_audience_shares_a_peak():
    from app.neuro import peak_end_per_twin

    same = _Resp([_Appraisal(arousal=a, valence=0.5) for a in (0.1, 0.9, 0.2)])
    per = peak_end_per_twin([same, same, same])
    assert per is not None
    assert per["peak_agreement"] == 1.0
    assert per["peak_index"] == 1


def test_encoding_strength_is_not_understated_by_jensen():
    """max(mean arousal) <= mean(max arousal). Averaging first systematically
    under-reports how aroused individuals actually got."""
    from app.neuro import peak_end_per_twin

    a = _Resp([_Appraisal(arousal=x, valence=0.0) for x in (0.9, 0.1, 0.1)])
    b = _Resp([_Appraisal(arousal=x, valence=0.0) for x in (0.1, 0.1, 0.9)])
    per = peak_end_per_twin([a, b])
    assert per is not None
    assert per["encoding_strength"] == pytest.approx(0.9, abs=1e-6)  # not 0.5


def test_retention_is_a_hazard_with_a_shrinking_risk_set():
    """A beat that loses half of a small remaining audience is a different
    finding from one that loses the same COUNT out of everyone."""
    from app.neuro import retention_hazard

    stays = _Resp([_Appraisal(attention=0.8) for _ in range(4)])
    drops_at_2 = _Resp([_Appraisal(attention=a) for a in (0.8, 0.8, 0.05, 0.05)])
    r = retention_hazard([stays, stays, drops_at_2, drops_at_2])
    assert r is not None
    steps = {s["beat"]: s for s in r["steps"]}
    assert steps[0]["hazard"] == 0.0
    assert steps[2]["entered"] == 4 and steps[2]["dropped"] == 2
    assert steps[2]["hazard"] == pytest.approx(0.5)
    assert steps[3]["entered"] == 2  # risk set shrank
    assert r["completion_rate"] == pytest.approx(0.5)
    assert r["worst_beat"] == 2


def test_retention_names_no_worst_beat_when_nobody_dropped():
    """An unguarded argmax over all-zero hazards would name beat 0 and read as
    a finding."""
    from app.neuro import retention_hazard

    stays = _Resp([_Appraisal(attention=0.9) for _ in range(4)])
    r = retention_hazard([stays, stays, stays])
    assert r is not None and r["worst_beat"] is None
    assert r["completion_rate"] == 1.0


def test_abandonment_is_absorbing():
    """Someone who has stopped watching does not come back for beat 4 and rate
    it, so a later recovery must not resurrect them."""
    from app.neuro import retention_hazard

    bounces = _Resp([_Appraisal(attention=a) for a in (0.9, 0.05, 0.9, 0.9)])
    r = retention_hazard([bounces, bounces])
    assert r is not None
    assert r["completion_rate"] == 0.0
    steps = {s["beat"]: s for s in r["steps"]}
    assert steps[1]["dropped"] == 2
    # the curve keeps its full length, but with the risk set empty the hazard
    # is UNDEFINED rather than zero — a tail of zeros would read as "nobody
    # left here" when it means "nobody was left to leave"
    assert len(r["steps"]) == 4
    assert steps[2]["entered"] == 0 and steps[2]["hazard"] is None
    assert steps[3]["hazard"] is None
