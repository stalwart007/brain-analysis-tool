"""Tests that the statistics engine is actually wired into each simulation's
aggregation — i.e. the numbers surface on the objects the API returns."""

import pytest

from app.schemas import TwinObjection, TwinReaction
from app.studies import aggregate_objections, aggregate_price_curves
from app.swarm import aggregate_reactions, aggregate_walks


def _reaction(action: str, intent: float, engagement: float = 0.5) -> TwinReaction:
    return TwinReaction(
        engagement=engagement,
        intent_score=intent,
        action=action,  # type: ignore[arg-type]
        dropoff_point=None,
        friction_notes="n/a",
        inner_monologue="…",
    )


def test_swarm_aggregate_exposes_confidence_and_consensus():
    unanimous = [_reaction("abandon", 0.1) for _ in range(12)]
    agg = aggregate_reactions(unanimous, "stimulus", "low")
    assert agg.polarization == 0.0  # single action ⇒ no entropy
    assert agg.consensus == "strong consensus"
    assert agg.intent_ci == [0.1, 0.1]  # constant sample ⇒ degenerate CI


def test_swarm_aggregate_detects_a_split_audience():
    """The failure a mean hides: two camps averaging to a bland midpoint."""
    split = [_reaction("convert", 0.95) for _ in range(10)] + [
        _reaction("abandon", 0.05) for _ in range(10)
    ]
    agg = aggregate_reactions(split, "stimulus", "low")
    assert agg.mean_intent == pytest.approx(0.5, abs=0.01)  # the misleading average
    # Entropy is normalised against the full four-action vocabulary, so a clean
    # two-way split is ~log2/log4 = 0.5 — not 1.0. Normalising by the number of
    # *observed* actions would give 1.0 here and make this split
    # indistinguishable from a flat spread across all four actions, which is
    # the opposite finding.
    #
    # 0.518 rather than 0.500 because plug-in entropy is biased DOWN at these
    # sample sizes and now carries the Miller-Madow correction. The bias runs
    # in the dangerous direction for this product — it manufactures consensus
    # out of small samples — so it is corrected and the corrected value pinned.
    assert agg.polarization == pytest.approx(0.518, abs=0.001)
    assert agg.bimodality is not None and agg.bimodality > 0.555
    assert agg.consensus == "polarised"
    lo, hi = agg.intent_ci
    assert lo < 0.5 < hi


def test_walk_aggregate_attaches_survival_curve():
    steps = ["landing", "form", "checkout"]

    def record(i, action):
        return {
            "step_index": i,
            "step": steps[i],
            "action": action,
            "engagement": 0.5,
            "friction_notes": "n/a",
            "inner_monologue": "…",
        }

    walks = [
        {"persona_label": "a", "outcome": "abandoned", "records": [record(0, "continue"), record(1, "abandon")]},
        {"persona_label": "b", "outcome": "completed", "records": [record(0, "continue"), record(1, "continue"), record(2, "continue")]},
    ]
    agg = aggregate_walks(walks, steps, "high", 2)
    assert agg.survival is not None
    # step 1 saw 1 abandon out of 2 entrants ⇒ hazard 0.5, and is the worst step
    assert agg.survival["steps"][1]["hazard"] == pytest.approx(0.5)
    assert agg.survival["worst_step_index"] == 1


def test_price_aggregate_attaches_econometrics():
    prices = [10.0, 20.0, 30.0, 40.0, 50.0]
    # a clean linear demand schedule so the fit is exact
    curves = [[1.2 - 0.02 * p for p in prices]]
    result = aggregate_price_curves(curves, prices, "low", "a widget")
    econ = result.econometrics
    assert econ["ok"] is True
    assert econ["elasticity"] is not None
    assert econ["continuous_optimal_price"] == pytest.approx(30.0, abs=0.5)


def _objection(text: str, dealbreaker: bool = False, rec: float = 0.4) -> TwinObjection:
    return TwinObjection(
        sentiment="neutral",
        biggest_objection=text,
        is_dealbreaker=dealbreaker,
        would_recommend=rec,
    )


def test_objection_semantic_clustering_merges_paraphrases():
    """The concrete defect being fixed: three phrasings of one complaint used to
    rank as three separate size-1 themes."""
    objs = [
        _objection("It's too expensive", dealbreaker=True),
        _objection("The price is too high"),
        _objection("Costs way too much"),
        _objection("I don't trust the company"),
    ]
    # embeddings placed so the three price complaints are near-parallel
    embeddings = [
        [1.0, 0.0, 0.0],
        [0.99, 0.06, 0.0],
        [0.98, 0.10, 0.0],
        [0.0, 0.0, 1.0],
    ]
    result = aggregate_objections(objs, "low", "pitch", embeddings=embeddings)
    assert result.clustering_method == "semantic"
    top = result.top_objections[0]
    assert top.count == 3  # merged, not three separate entries
    assert top.dealbreaker_count == 1
    assert len(top.variants) == 2  # the other phrasings recorded
    assert len(result.top_objections) == 2  # price theme + trust theme


def test_objection_falls_back_to_exact_grouping_without_embeddings():
    objs = [_objection("Too expensive"), _objection("Too expensive"), _objection("No trust")]
    result = aggregate_objections(objs, "low", "pitch", embeddings=None)
    assert result.clustering_method == "exact"
    assert result.top_objections[0].count == 2


def test_objection_recommend_interval_follows_the_bootstrap_guard():
    """The interval appears when the sample can support one, and not before.

    This used to be asserted inside the grouping-fallback test above, on three
    objections — so it was really asserting that `bootstrap_ci` would hand back
    an interval at n = 3, which is exactly the behaviour that was wrong: at that
    size the "95% CI" was the sample range, and the range of two draws covers
    the true mean less than half the time. The guard now refuses, and the
    contract worth pinning is the transition, not either side alone.
    """
    three = [_objection("Too expensive"), _objection("Too expensive"), _objection("No trust")]
    assert aggregate_objections(three, "low", "pitch", embeddings=None).recommend_ci is None

    many = [_objection("Too expensive") for _ in range(6)] + [
        _objection("No trust") for _ in range(4)
    ]
    assert aggregate_objections(many, "low", "pitch", embeddings=None).recommend_ci is not None


def test_objection_ignores_none_responses_as_themes():
    objs = [_objection("none"), _objection("N/A"), _objection("Too pricey")]
    result = aggregate_objections(objs, "low", "pitch")
    assert len(result.top_objections) == 1
    assert result.twin_count == 3  # but they still count toward the sample


def test_swarm_aggregate_discovers_segments_in_split_audience():
    """A polarised swarm must yield two discovered segments; the k, sizes and
    labels must reflect the actual camps."""
    lovers = [_reaction("convert", 0.92, 0.9) for _ in range(8)]
    haters = [_reaction("abandon", 0.08, 0.1) for _ in range(8)]
    agg = aggregate_reactions(lovers + haters, "stimulus", "low")
    assert agg.segments is not None
    assert agg.segments["k"] == 2
    groups = agg.segments["groups"]
    labels = {g["label"] for g in groups}
    assert labels == {"enthusiasts", "skeptics"}
    assert all(g["size"] == 8 for g in groups)
    enth = next(g for g in groups if g["label"] == "enthusiasts")
    assert enth["dominant_action"] == "convert"


def test_swarm_aggregate_reports_no_segments_for_homogeneous_swarm():
    same = [_reaction("continue", 0.5, 0.5) for _ in range(12)]
    agg = aggregate_reactions(same, "stimulus", "low")
    assert agg.segments is None  # honesty: no invented structure


def test_objection_theme_map_positions_are_semantic():
    """Themes must land at the centroid of their members' PCA projections, so
    the two price-complaints sit together and the trust complaint sits apart."""
    objs = [
        _objection("Too expensive", dealbreaker=True),
        _objection("Price is very high"),
        _objection("I do not trust this company"),
        _objection("Seems completely untrustworthy"),
    ]
    embeddings = [
        [1.0, 0.02, 0.0],
        [0.99, 0.05, 0.0],
        [0.0, 1.0, 0.03],
        [0.02, 0.98, 0.0],
    ]
    result = aggregate_objections(objs, "low", "pitch", embeddings=embeddings)
    assert result.theme_map is not None
    assert len(result.theme_map) == 2
    a, b = result.theme_map
    # the two themes must be far apart on the semantic map
    dist = ((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2) ** 0.5
    assert dist > 0.5
    price_theme = next(t for t in result.theme_map if t["count"] == 2 and t["dealbreaker"])
    assert "expens" in price_theme["label"].lower() or "price" in price_theme["label"].lower()


def test_aggregate_reports_the_shortfall_not_just_the_survivors():
    """A run where 180 of 200 twins hit a 429 used to persist as
    `twin_count: 20`, byte-identical to a healthy 20-twin run — including in
    the calibration report. Every interval is computed over the survivors, so
    the reader has to be told that is what they are looking at.
    """
    from app.schemas import TwinReaction
    from app.swarm import aggregate_reactions

    survivors = [
        TwinReaction(
            engagement=0.5,
            intent_score=0.5,
            action="continue",
            dropoff_point=None,
            friction_notes="none",
            inner_monologue="fine",
        )
        for _ in range(20)
    ]
    agg = aggregate_reactions(survivors, "stimulus", "low", requested=200)
    assert agg.twin_count == 20
    assert agg.twins_requested == 200
    assert agg.twins_failed == 180


def test_a_healthy_run_reports_no_failures():
    from app.schemas import TwinReaction
    from app.swarm import aggregate_reactions

    reactions = [
        TwinReaction(
            engagement=0.5,
            intent_score=0.5,
            action="continue",
            dropoff_point=None,
            friction_notes="none",
            inner_monologue="fine",
        )
        for _ in range(12)
    ]
    agg = aggregate_reactions(reactions, "stimulus", "low", requested=12)
    assert agg.twins_failed == 0 and agg.twins_requested == 12
