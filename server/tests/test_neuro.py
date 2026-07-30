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


# ── change_points: the permutation calibration ────────────────────────────


def _beat_ratings(curve, twins=20, sd=0.15, seed=0, quantise=False):
    """Per-beat twin ratings whose beat-means track `curve` — the shape the
    production caller passes as `observations`."""
    rng = random.Random(seed)
    out = []
    for mu in curve:
        col = [min(1.0, max(0.0, rng.gauss(mu, sd))) for _ in range(twins)]
        if quantise:
            col = [round(v, 1) for v in col]
        out.append(col)
    return out


def test_change_points_holds_a_nominal_level_at_every_twin_count():
    """The defect was that the false-positive rate tracked the twin count.

    The BIC criterion's statistic is a MAXIMUM over admissible splits, so it is
    not χ²-distributed, its null depends on the segment length, and recursive
    segmentation piled on uncorrected multiplicity. Measured on pure-noise
    beat-mean curves at nominal 5%, it fired at 19.8% / 9.5% / 5.4% / 3.4% for
    3 / 8 / 20 / 40 twins on an 8-beat curve, and 63.3% / 43.6% / 33.7% / 24.9%
    on a 3-beat one — a number that moved between 3% and 63% depending only on
    how much data you had. Against the permutation null it measured 3.8-7.0%
    across every twin and beat count tried (600 trials/cell).

    150 trials/cell here keeps the suite fast; the ceiling is loose enough to
    absorb that Monte Carlo error and tight enough to catch a return to 20%+.
    """
    for twins in (3, 8, 20):
        for beats in (4, 8):
            fired = 0
            trials = 150
            for s in range(trials):
                obs = _beat_ratings([0.5] * beats, twins=twins, seed=7000 + s)
                curve = [sum(c) / len(c) for c in obs]
                if change_points(curve, min_size=1, n_obs=twins, observations=obs):
                    fired += 1
            assert fired / trials < 0.13, (twins, beats, fired / trials)


def test_change_points_still_finds_a_real_step_it_should():
    """Calibration must not be bought with power. A 0.12 step at beat 5 of 8 is
    detected 72% of the time at 8 twins, 98.8% at 20 and 100% at 40 — measured
    higher than the uncalibrated BIC's 63.0% / 80.6%, because the twin-level
    null replaces a scale-free ratio with a statistic on the observation scale.
    """
    curve = [0.5] * 5 + [0.62] * 3
    hits = 0
    for s in range(40):
        obs = _beat_ratings(curve, twins=20, seed=900 + s)
        mean_curve = [sum(c) / len(c) for c in obs]
        cps = change_points(mean_curve, min_size=1, n_obs=20, observations=obs)
        if any(abs(c - 5) <= 1 for c in cps):
            hits += 1
    assert hits >= 34  # ~98.8% measured; 34/40 leaves room for the tail


def test_change_points_is_deterministic():
    """The permutation seed is derived from the segment's own values, so the
    same curve must give the same answer every time — a study re-read from
    history cannot disagree with the study as it streamed."""
    curve = [0.9] * 4 + [0.35] * 4
    obs = _beat_ratings(curve, twins=12, seed=5)
    first = change_points(curve, min_size=1, n_obs=12, observations=obs)
    for _ in range(4):
        assert change_points(curve, min_size=1, n_obs=12, observations=obs) == first
    assert first, "sanity: this curve does have a break"


def test_change_points_quantised_ratings_do_not_break_the_level():
    """Twin scores arrive quantised to 0.1, which shrinks the randomization
    support and could inflate the level. Measured 4.3-5.2% on the quantised
    variant against 4.8-6.0% unquantised."""
    fired = 0
    trials = 150
    for s in range(trials):
        obs = _beat_ratings([0.5] * 8, twins=20, seed=6000 + s, quantise=True)
        curve = [sum(c) / len(c) for c in obs]
        if change_points(curve, min_size=1, n_obs=20, observations=obs):
            fired += 1
    assert fired / trials < 0.13


def test_change_points_curve_only_null_is_conservative_not_wrong():
    """Without ratings the test is structurally conservative, and that is a
    documented limitation rather than a bug: the max statistic is invariant to
    permutation within each candidate segment, so p cannot fall below
    2/C(m,k) — 0.33 on a 4-beat curve, which no signal can beat. It must fail
    CLOSED (no split claimed) rather than open."""
    assert change_points([0.5, 0.5, 0.9, 0.9], min_size=1) == []
    # and with the ratings the same curve resolves
    obs = _beat_ratings([0.5, 0.5, 0.9, 0.9], twins=20, sd=0.05, seed=2)
    assert change_points([0.5, 0.5, 0.9, 0.9], min_size=1, observations=obs) == [2]


def test_change_points_rejects_a_negligible_but_real_shift():
    """A calibrated test still cannot tell a real break from a negligible one,
    so the min_effect ROPE stays. One part in 500,000 is not an attention
    collapse however clean the statistics are."""
    curve = [0.5, 0.5, 0.500001, 0.500001]
    obs = _beat_ratings(curve, twins=20, sd=0.02, seed=1)
    assert change_points(curve, min_size=1, observations=obs) == []
