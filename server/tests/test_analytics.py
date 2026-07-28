"""Known-answer tests for the statistics engine.

Every expected value here is derived by hand from the textbook formula, not
captured from a previous run — so these tests actually verify the mathematics
rather than freezing whatever the code happened to output.
"""

import math

import pytest

from app.analytics import (
    agglomerative_cluster,
    bayesian_ab,
    bimodality_coefficient,
    bootstrap_ci,
    consensus_label,
    cosine_similarity,
    excess_kurtosis,
    kaplan_meier,
    medoid,
    price_elasticity,
    shannon_entropy_normalised,
    skewness,
)

# ───────────────────────────────── descriptive statistics


def test_entropy_extremes():
    # all mass on one outcome ⇒ perfect consensus
    assert shannon_entropy_normalised([10, 0, 0, 0]) == 0.0
    # 50/50 split over two outcomes ⇒ maximal disagreement
    assert shannon_entropy_normalised([5, 5]) == pytest.approx(1.0)
    # uniform over four outcomes is also maximal (normalised by log k)
    assert shannon_entropy_normalised([3, 3, 3, 3]) == pytest.approx(1.0)
    # lopsided split sits between the extremes
    assert 0.0 < shannon_entropy_normalised([9, 1]) < 1.0


def test_skewness_and_kurtosis_known_values():
    # symmetric two-point distribution: skew 0, excess kurtosis exactly -2
    xs = [0.0] * 20 + [1.0] * 20
    assert skewness(xs) == pytest.approx(0.0, abs=1e-12)
    assert excess_kurtosis(xs) == pytest.approx(-2.0, abs=1e-12)


def test_bimodality_detects_split_swarm():
    """A perfectly split swarm must exceed Sarle's 5/9 threshold.

    Sarle's correction 3(n−1)²/((n−2)(n−3)) is defined against the
    BIAS-CORRECTED moments G1/G2, not the population g1/g2. For a symmetric
    two-point sample g1 = 0 and g2 = −2, so

        G1 = 0
        G2 = ((n+1)·(−2) + 6)·(n−1) / ((n−2)(n−3))
        BC = (G1² + 1) / (G2 + 3(n−1)²/((n−2)(n−3)))

    At n = 40: G2 = −2.10811…, correction = 3·39²/(38·37) = 3.24537…, so
    BC ≈ 0.8793.

    Feeding the correction population moments instead yields 0.8030 here, and
    deflates BC at small n badly enough that a perfect 50/50 split is
    undetectable at n = 12 — see the n = 12 case below, which is the size an
    actual swarm runs at.
    """
    n = 40
    split = [0.0] * 20 + [1.0] * 20
    bc = bimodality_coefficient(split)
    correction = 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    big_g2 = ((n + 1) * -2.0 + 6.0) * (n - 1) / ((n - 2) * (n - 3))
    assert bc == pytest.approx(1.0 / (big_g2 + correction), rel=1e-9)
    assert bc > 0.555  # flagged as polarised

    # The case that matters: the canonical two-camp audience at a realistic
    # swarm size. Under the mixed-convention form this scores 0.492 and stays
    # silent, which is the failure this guards.
    assert bimodality_coefficient([0.1] * 6 + [0.9] * 6) > 0.555

    # a tightly concentrated (unimodal) sample must NOT be flagged
    unimodal = [0.5] * 36 + [0.45, 0.55, 0.48, 0.52]
    assert bimodality_coefficient(unimodal) < 0.555

    # below n = 12 the coefficient has no power, so it refuses rather than
    # returning a number that reads as a measurement
    assert bimodality_coefficient([0.1, 0.2, 0.3]) is None
    assert bimodality_coefficient([0.0] * 4 + [1.0] * 4) is None
    assert bimodality_coefficient([0.5] * 14) == 0.0  # zero variance


def test_consensus_label_reads_the_room():
    assert consensus_label(0.1, 0.2) == "strong consensus"
    assert consensus_label(0.9, 0.8) == "polarised"
    assert consensus_label(0.5, 0.2) == "leaning"


# ───────────────────────────────── bootstrap


def test_bootstrap_ci_on_constant_sample_is_degenerate():
    # every resample of a constant sample has the same mean
    assert bootstrap_ci([0.5] * 10) == (0.5, 0.5)


def test_bootstrap_ci_brackets_the_mean_and_is_deterministic():
    values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    lo, hi = bootstrap_ci(values, iterations=1000, seed=42)
    assert lo < 0.5 < hi  # sample mean is 0.5
    assert bootstrap_ci(values, iterations=1000, seed=42) == (lo, hi)  # reproducible
    assert bootstrap_ci([0.5]) is None  # too small to resample


# ───────────────────────────────── Bayesian A/B


def test_bayesian_ab_identifies_a_decisive_winner():
    result = bayesian_ab({"A": (1, 100), "B": (99, 100)}, draws=4000, seed=3)
    assert result["winner"] == "B"
    assert result["p_win"] > 0.99
    assert result["conclusive"] is True

    arms = {a["name"]: a for a in result["arms"]}
    # posterior mean = (1+s)/(2+n): A → 2/102, B → 100/102
    assert arms["A"]["posterior_mean"] == pytest.approx(2 / 102, abs=1e-4)
    assert arms["B"]["posterior_mean"] == pytest.approx(100 / 102, abs=1e-4)
    # the winner's expected regret is ~0; the loser's is large
    assert arms["B"]["expected_loss"] < 0.01
    assert arms["A"]["expected_loss"] > 0.5
    # credible interval must bracket the posterior mean
    assert arms["B"]["ci_low"] < arms["B"]["posterior_mean"] < arms["B"]["ci_high"]


def test_bayesian_ab_refuses_to_call_a_tie():
    result = bayesian_ab({"A": (50, 100), "B": (50, 100)}, draws=4000, seed=5)
    assert 0.35 < result["p_win"] < 0.65  # neither arm dominates
    assert result["conclusive"] is False


def test_bayesian_ab_flags_small_samples_as_inconclusive():
    """The headline defect this fixes: a big raw 'lift' on a tiny sample.

    3/5 vs 1/5 is a 200% naive lift, but the posteriors overlap heavily, so the
    expected loss stays well above the ROPE and we must NOT declare a winner.
    """
    result = bayesian_ab({"A": (1, 5), "B": (3, 5)}, draws=4000, seed=7)
    assert result["winner"] == "B"
    assert result["conclusive"] is False
    assert "NOT yet conclusive" in result["verdict"]


# ───────────────────────────────── pricing econometrics


def test_elasticity_recovers_a_known_power_law():
    """Q = 2·P^(-1.5) ⇒ log-log slope is exactly -1.5 with R² = 1."""
    prices = [10.0, 20.0, 40.0, 80.0]
    demand = [2.0 * p**-1.5 for p in prices]
    out = price_elasticity(prices, demand)
    assert out["ok"] is True
    assert out["elasticity"] == pytest.approx(-1.5, abs=1e-6)
    assert out["elasticity_r2"] == pytest.approx(1.0, abs=1e-6)
    assert out["classification"].startswith("elastic")


def test_linear_demand_gives_interior_revenue_optimum():
    """Q = 1.2 - 0.02·P ⇒ R(P) = 1.2P - 0.02P², maximised at P* = 30.

    Note 30 is *between* the tested points in the sense that an argmax over the
    grid's revenue would also land at 30 here, but the fit yields it analytically
    and would return a non-tested price for other schedules.
    """
    prices = [10.0, 20.0, 30.0, 40.0, 50.0]
    demand = [1.2 - 0.02 * p for p in prices]
    out = price_elasticity(prices, demand)
    assert out["demand_fit_r2"] == pytest.approx(1.0, abs=1e-9)
    assert out["continuous_optimal_price"] == pytest.approx(30.0, abs=1e-6)


def test_elasticity_needs_enough_points():
    assert price_elasticity([10.0, 20.0], [0.5, 0.4])["ok"] is False


# ───────────────────────────────── survival analysis


def test_kaplan_meier_hazard_beats_raw_counts():
    """The correctness argument for hazard over drop-off counts.

    Step 0 loses 8 agents out of 100 (hazard 0.08); step 1 loses only 5 — but
    out of the 10 that got there (hazard 0.50). Raw counts would blame step 0;
    the hazard correctly indicts step 1.
    """
    km = kaplan_meier(entered=[100, 10], dropped=[8, 5])
    s0, s1 = km["steps"]
    assert s0["hazard"] == pytest.approx(0.08)
    assert s1["hazard"] == pytest.approx(0.50)
    assert km["worst_step_index"] == 1  # NOT step 0, despite 8 > 5
    # S = (1-0.08)(1-0.50) = 0.46
    assert km["overall_survival"] == pytest.approx(0.46, abs=1e-4)


def test_kaplan_meier_survival_is_monotone_with_widening_bands():
    km = kaplan_meier(entered=[100, 80, 40], dropped=[20, 40, 4])
    survivals = [s["survival"] for s in km["steps"]]
    assert survivals == pytest.approx([0.8, 0.4, 0.36], abs=1e-4)
    assert all(
        survivals[i] >= survivals[i + 1] for i in range(len(survivals) - 1)
    )  # survival can never increase
    widths = [s["survival_ci_high"] - s["survival_ci_low"] for s in km["steps"]]
    assert widths[2] > 0  # Greenwood band exists and is non-degenerate


def test_kaplan_meier_handles_empty_steps():
    km = kaplan_meier(entered=[10, 0], dropped=[10, 0])
    assert km["overall_survival"] == pytest.approx(0.0)


# ───────────────────────────────── clustering


def test_cosine_similarity_basics():
    assert cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)
    assert cosine_similarity([0, 0], [1, 1]) == 0.0  # degenerate vector


def test_agglomerative_merges_near_duplicates_only():
    """This is the actual bug being fixed: two phrasings of the same objection
    must land in one cluster, while a genuinely different objection stays out."""
    vectors = [
        [1.0, 0.0, 0.0],  # 0: "too expensive"
        [0.99, 0.05, 0.0],  # 1: "the price is too high"  (near-duplicate of 0)
        [0.0, 1.0, 0.0],  # 2: "I don't trust it"        (distinct)
        [0.02, 0.99, 0.0],  # 3: "seems untrustworthy"    (near-duplicate of 2)
    ]
    clusters = agglomerative_cluster(vectors, threshold=0.82)
    assert len(clusters) == 2
    grouped = sorted(sorted(c) for c in clusters)
    assert grouped == [[0, 1], [2, 3]]


def test_agglomerative_keeps_dissimilar_items_apart():
    vectors = [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]
    assert len(agglomerative_cluster(vectors, threshold=0.82)) == 3


def test_medoid_picks_the_most_representative_member():
    # 0 and 1 sit together; 2 is the outlier, so it must never be the medoid
    vectors = [[1.0, 0.0], [0.98, 0.2], [0.0, 1.0]]
    assert medoid([0, 1, 2], vectors) in (0, 1)
    assert medoid([0], vectors) == 0


def test_empty_inputs_are_safe():
    assert agglomerative_cluster([]) == []
    assert shannon_entropy_normalised([]) == 0.0
    assert math.isfinite(cosine_similarity([1, 2], [3, 4]))


# ───────────────────────────────── silhouette & auto-threshold


def test_silhouette_separates_good_from_bad_clusterings():
    from app.analytics import silhouette_score

    vectors = [
        [1.0, 0.02], [0.99, 0.05], [0.98, 0.01],   # tight group A
        [0.02, 1.0], [0.05, 0.99], [0.01, 0.98],   # tight group B
    ]
    good = silhouette_score(vectors, [[0, 1, 2], [3, 4, 5]])
    bad = silhouette_score(vectors, [[0, 1, 3], [2, 4, 5]])  # deliberately mixed
    assert good is not None and good > 0.9
    assert bad is not None and bad < good
    assert silhouette_score(vectors, [[0, 1, 2, 3, 4, 5]]) is None  # k=1 undefined
    # singleton clusters take s=0 by convention, not an error
    assert silhouette_score(vectors, [[0], [1], [2], [3], [4], [5]]) == 0.0


def test_auto_cluster_picks_the_natural_partition():
    from app.analytics import auto_cluster

    vectors = [
        [1.0, 0.0, 0.0], [0.99, 0.06, 0.0], [0.97, 0.1, 0.0],  # price complaints
        [0.0, 1.0, 0.0], [0.03, 0.99, 0.0],                    # trust complaints
    ]
    clusters, threshold, score = auto_cluster(vectors)
    assert sorted(sorted(c) for c in clusters) == [[0, 1, 2], [3, 4]]
    assert score is not None and score > 0.8
    assert 0.6 <= threshold <= 0.9


def test_auto_cluster_small_input_falls_back():
    from app.analytics import auto_cluster

    clusters, threshold, score = auto_cluster([[1.0, 0.0], [0.0, 1.0]])
    assert threshold == 0.82 and score is None
    assert len(clusters) == 2


# ───────────────────────────────── Thompson sampling


def test_thompson_allocation_favours_the_stronger_arm():
    from app.analytics import thompson_allocate

    counts = thompson_allocate({"A": (80, 100), "B": (5, 100)}, wave=40, seed=1)
    assert counts["A"] + counts["B"] == 40
    assert counts["A"] >= 36  # posterior mass overwhelmingly on A


def test_thompson_allocation_explores_under_ignorance():
    from app.analytics import thompson_allocate

    counts = thompson_allocate({"A": (0, 0), "B": (0, 0)}, wave=60, seed=2)
    assert counts["A"] + counts["B"] == 60
    assert counts["A"] > 10 and counts["B"] > 10  # uniform posteriors ⇒ both explored


def test_thompson_allocation_is_deterministic_per_seed():
    from app.analytics import thompson_allocate

    a = thompson_allocate({"A": (3, 10), "B": (6, 10)}, wave=20, seed=9)
    b = thompson_allocate({"A": (3, 10), "B": (6, 10)}, wave=20, seed=9)
    assert a == b


# ───────────────────────────────── k-means segmentation


def test_kmeans_auto_finds_two_clear_blobs():
    from app.analytics import kmeans_auto

    blob_a = [(0.1 + i * 0.01, 0.1 + i * 0.01) for i in range(8)]
    blob_b = [(0.9 - i * 0.01, 0.9 - i * 0.01) for i in range(8)]
    result = kmeans_auto(blob_a + blob_b)
    assert result is not None
    labels, k, silhouette = result
    assert k == 2
    assert silhouette > 0.7  # blobs are far apart, silhouette must be high
    # every point in blob_a shares a label; every point in blob_b the other
    assert len(set(labels[:8])) == 1 and len(set(labels[8:])) == 1
    assert labels[0] != labels[8]


def test_kmeans_auto_refuses_homogeneous_data():
    """No structure in the data ⇒ report None, don't invent segments."""
    from app.analytics import kmeans_auto

    tight = [(0.5 + (i % 3) * 0.001, 0.5 - (i % 2) * 0.001) for i in range(12)]
    assert kmeans_auto(tight) is None
    assert kmeans_auto([(0.5, 0.5)] * 4) is None  # n < 6


# ───────────────────────────────── PCA semantic map


def test_pca_recovers_dominant_axis():
    from app.analytics import pca_2d

    # points along a straight line in 4-D: PC1 carries everything, PC2 ≈ 0
    line = [[t, 2 * t, -t, 0.5 * t] for t in [0.0, 1.0, 2.0, 3.0, 4.0]]
    coords = pca_2d(line)
    assert coords is not None
    ys = [abs(y) for _, y in coords]
    assert max(ys) < 0.05  # no second-axis variance
    xs = [x for x, _ in coords]
    assert xs == sorted(xs) or xs == sorted(xs, reverse=True)  # order preserved


def test_pca_separates_two_groups():
    from app.analytics import pca_2d

    group1 = [[1.0, 0.0, 0.05 * i] for i in range(4)]
    group2 = [[0.0, 1.0, 0.05 * i] for i in range(4)]
    coords = pca_2d(group1 + group2)
    assert coords is not None
    g1x = [x for x, _ in coords[:4]]
    g2x = [x for x, _ in coords[4:]]
    # the groups must land on opposite sides of PC1
    assert max(g1x) < min(g2x) or min(g1x) > max(g2x)


def test_pca_small_input_returns_none():
    from app.analytics import pca_2d

    assert pca_2d([[1.0, 2.0], [3.0, 4.0]]) is None
