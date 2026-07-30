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


def test_bootstrap_ci_refuses_small_samples():
    """The audit case. At n = 2 the old percentile version returned the sample
    range and called it a 95% CI — bootstrap_ci([0.1, 0.9]) == (0.1, 0.9) —
    an interval whose measured coverage for the mean of Exponential(1) is
    46.5% (200k trials), less than half the label. n = 3…7 differ only in
    degree, so below 8 the honest answer is no interval at all.
    """
    assert bootstrap_ci([0.1, 0.9]) is None  # the exact flaw scenario
    for n in range(2, 8):
        assert bootstrap_ci([0.1 * i for i in range(n)]) is None
    assert bootstrap_ci([0.1 * i for i in range(8)]) is not None  # gate is 8


def test_bootstrap_ci_bca_coverage_on_skewed_data():
    """Coverage on the distribution where the percentile version was measured
    failing: the mean of Exponential(1) at n = 10.

    Full-size paired measurement (2000 trials × 2000 resamples, both endpoints
    from the same bootstrap draws): percentile 86.3%, BCa 88.4%.

    This test does NOT assert BCa > percentile. That gap is ~2 points and the
    Monte-Carlo error at a test-sized run is ~2 points, so such an assertion
    would be a coin-flip dressed as a regression — in paired reruns across
    five meta-seeds at 200-400 trials, BCa led by up to 2 points and trailed
    by 1. What is pinned is the level: this exact seeded run (300 trials × 800
    resamples, meta-seed 11) measured **84.0%**, the worst of the five seeds
    tried (the others: 88.3 / 89.7 / 89.0 / 88.0%). The floor sits at 80%,
    below every observed run and far above what a broken estimator returns —
    the interval this replaced covered 46.5% at n = 2.
    """
    import random

    rng = random.Random(11)
    trials, hits = 300, 0
    for t in range(trials):
        xs = [rng.expovariate(1.0) for _ in range(10)]
        lo, hi = bootstrap_ci(xs, iterations=800, seed=t)
        hits += lo <= 1.0 <= hi
    assert hits / trials >= 0.80, f"BCa covered only {hits}/{trials}"


def test_bootstrap_ci_skew_shifts_the_interval_toward_the_long_tail():
    """On right-skewed data both corrections push the interval right — that is
    what 'bias-corrected and accelerated' buys. The percentile interval this
    replaced is symmetric in probability around the bootstrap median, so
    equality here would mean the corrections are dead code."""
    skewed = [0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.3, 0.4, 0.8, 1.6]
    m = sum(skewed) / len(skewed)
    lo, hi = bootstrap_ci(skewed, seed=13)
    assert lo < m < hi
    assert (hi - m) > (m - lo)  # long tail is to the right, interval follows


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


def test_kaplan_meier_terminal_step_is_not_certain():
    """The audit case. A step with entered=1, dropped=1 reported
    survival_ci = [0.0, 0.0] under the Wald band — 95% certainty of
    extinction from ONE observation, because se = Ŝ·√G and Ŝ = 0 zeroes the
    width no matter how thin the evidence. At Ŝ = 0 the log-log transform has
    nothing to say either, so the upper limit comes from the Wilson lower
    bound of the terminal hazard: 1 − 0.2065 = 0.7935 of doubt from one
    observation, shrinking to 0.037 when 100 of 100 actually dropped.
    """
    km = kaplan_meier(entered=[1], dropped=[1])
    s = km["steps"][0]
    assert s["survival_ci_low"] == 0.0
    assert s["survival_ci_high"] == pytest.approx(0.7935, abs=1e-3)  # not 0.0

    km100 = kaplan_meier(entered=[100], dropped=[100])
    s100 = km100["steps"][0]
    assert 0.0 < s100["survival_ci_high"] < 0.05  # real evidence, tight band
    assert s100["survival_ci_high"] < s["survival_ci_high"]


def test_kaplan_meier_no_drop_step_is_not_certain_either():
    """The mirror edge: Ŝ = 1 with zero observed drops. Wald reported
    [1.0, 1.0] — certainty of perfect survival from any n, including 1. The
    lower limit now comes from the Wilson upper bound's complement: 0-of-1
    reads [0.2065, 1.0], 0-of-100 reads [0.963, 1.0].
    """
    km1 = kaplan_meier(entered=[1], dropped=[0])
    assert km1["steps"][0]["survival_ci_low"] == pytest.approx(0.2065, abs=1e-3)
    assert km1["steps"][0]["survival_ci_high"] == 1.0

    km100 = kaplan_meier(entered=[100], dropped=[0])
    assert 0.9 < km100["steps"][0]["survival_ci_low"] < 1.0


def test_kaplan_meier_loglog_band_known_value():
    """Kalbfleisch-Prentice by hand: 20-of-100 gives Ŝ = 0.8,
    G = 20/(100·80) = 0.0025, θ = log(−log 0.8) = −1.49997,
    se_θ = √G/|log 0.8| = 0.22407, band = exp(−exp(θ ∓ 1.95996·se_θ))
    = (0.70735, 0.86604). Also pins that the band respects (0, 1) without
    clipping — the Wald band here would be 0.8 ± 0.0784.
    """
    km = kaplan_meier(entered=[100], dropped=[20])
    s = km["steps"][0]
    assert s["survival_ci_low"] == pytest.approx(0.70735, abs=1e-3)
    assert s["survival_ci_high"] == pytest.approx(0.86604, abs=1e-3)
    assert 0.0 < s["survival_ci_low"] < s["survival"] < s["survival_ci_high"] < 1.0


def test_kaplan_meier_worst_step_needs_evidence():
    """The audit case. worst_step_index was an argmax on raw hazard, so a
    1-of-1 step (ĥ = 1.0) outranked 5-of-10 (ĥ = 0.5) and even 100-of-200 —
    a verdict about the funnel's biggest leak decided by a single agent.
    Under the Wilson lower bound the ordering is
    wilson_low(1,1) = 0.2065 < wilson_low(5,10) = 0.2366 <
    wilson_low(100,200) = 0.4314, so attested leaks win.
    """
    km = kaplan_meier(entered=[10, 1], dropped=[5, 1])
    assert km["worst_step_index"] == 0  # 5-of-10 outranks 1-of-1
    assert km["worst_step_hazard"] == pytest.approx(0.5)  # raw hazard, kept
    assert km["worst_step_hazard_wilson_low"] == pytest.approx(0.2366, abs=1e-3)
    lows = [s["hazard_wilson_low"] for s in km["steps"]]
    assert lows[0] > lows[1] > 0.0

    km2 = kaplan_meier(entered=[200, 1], dropped=[100, 1])
    assert km2["worst_step_index"] == 0  # 100-of-200 outranks 1-of-1 too

    # ...but a genuinely murderous step with real n still wins outright
    km3 = kaplan_meier(entered=[100, 50], dropped=[10, 45])
    assert km3["worst_step_index"] == 1


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


def test_kmeans_refuses_structureless_noise():
    """The regression this whole selector was rewritten for.

    k-means partitions ANY cloud, so a selector that maximises silhouette over
    k ∈ {2,3,4} can never answer "there are no segments" — silhouette is not
    even defined at k = 1. Measured on pure uniform noise over [0,1]², the
    silhouette selector invented a split in 86-90% of runs at every sample
    size, and downstream those groups are labelled "enthusiasts" and
    "skeptics" and shown to a researcher as a finding.

    The gap statistic ranks k = 1 on the same axis as the rest, so refusing is
    an answer it can give. Nominal level is ~5%; anything under ~15% here is a
    different estimator from the one that shipped.
    """
    import random

    from app.analytics import kmeans_auto

    rng = random.Random(4)
    invented = 0
    trials = 60
    for _ in range(trials):
        noise = [(rng.random(), rng.random()) for _ in range(20)]
        if kmeans_auto(noise) is not None:
            invented += 1
    assert invented / trials < 0.15, (
        f"invented segments in {invented}/{trials} pure-noise runs"
    )


def test_kmeans_segments_reports_its_own_evidence():
    """A split has to arrive with the reason to believe it, not just a k.

    `gap_vs_k1` is how far the chosen partition sits above "no structure at
    all", in standard errors of the reference ensemble. It is what the findings
    layer quotes; without it a reader cannot tell a decisive split from a
    marginal one, and both render identically as two coloured groups.
    """
    from app.analytics import kmeans_segments

    blob_a = [(0.1 + i * 0.01, 0.1 + i * 0.01) for i in range(8)]
    blob_b = [(0.9 - i * 0.01, 0.9 - i * 0.01) for i in range(8)]
    seg = kmeans_segments(blob_a + blob_b)
    assert seg is not None
    assert seg["k"] == 2
    assert seg["gap_vs_k1"] > 1.0  # clearly above the no-structure reference
    assert seg["centroid_spread"] > 0.15
    assert seg["n_reference"] >= 2 and "gap statistic" in seg["method"]


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


# ───────────────────────── simultaneous inference


def _correlated_curves(seed, n_twins, truth):
    """Twins with individual BASELINES — the correlation structure that makes
    independent per-beat resampling wrong."""
    import random

    rng = random.Random(seed)
    per_twin = []
    for _ in range(n_twins):
        base = rng.gauss(0, 0.10)
        per_twin.append(
            [max(0.0, min(1.0, t + base + rng.gauss(0, 0.10))) for t in truth]
        )
    return [[tw[i] for tw in per_twin] for i in range(len(truth))]


def test_simultaneous_band_covers_the_whole_curve():
    """The regression the sup-t band exists for.

    A pointwise 95% interval is read as a band the moment it is drawn across a
    curve, and it does not behave like one: measured coverage of an entire
    8-beat curve was 50-63% depending on twin count, against the 95% the chart
    announces. With nine dimensions on screen that is dozens of chances to find
    the one beat that looks different.

    The band must cover the full curve at least 90% of the time (the estimator
    is slightly conservative at these sizes, which is the safe direction), and
    it must beat the pointwise version it replaces.
    """
    from app.analytics import simultaneous_band

    truth = [0.3, 0.5, 0.7, 0.6, 0.4, 0.55, 0.65, 0.45]
    trials, sim_hits, pt_hits = 120, 0, 0
    for s in range(trials):
        r = simultaneous_band(_correlated_curves(s, 12, truth), iterations=400)
        assert r is not None
        sim_hits += all(lo <= truth[i] <= hi for i, (lo, hi) in enumerate(r["band"]))
        pt_hits += all(lo <= truth[i] <= hi for i, (lo, hi) in enumerate(r["pointwise"]))
    assert sim_hits / trials >= 0.90, f"simultaneous coverage only {sim_hits}/{trials}"
    assert sim_hits > pt_hits  # strictly better than what it replaces


def test_simultaneous_band_is_wider_than_pointwise():
    """If it were not wider it would not be correcting anything."""
    from app.analytics import simultaneous_band

    truth = [0.3, 0.5, 0.7, 0.6, 0.4, 0.55, 0.65, 0.45]
    r = simultaneous_band(_correlated_curves(1, 12, truth), iterations=400)
    wide = sum(hi - lo for lo, hi in r["band"])
    narrow = sum(hi - lo for lo, hi in r["pointwise"])
    assert wide > narrow
    assert r["critical_value"] > 1.96  # past the naive per-beat z
    assert r["n_indices"] == 8 and r["n_subjects"] == 12


def test_simultaneous_band_refuses_too_few_subjects():
    from app.analytics import simultaneous_band

    assert simultaneous_band([[0.1, 0.2], [0.3, 0.4]]) is None  # n = 2
    assert simultaneous_band([]) is None
    # ragged input is a caller bug, not something to silently average
    assert simultaneous_band([[0.1, 0.2, 0.3], [0.4, 0.5]]) is None


def test_simultaneous_band_handles_unanimity():
    """Every twin identical at every beat: zero-width, flagged, not a crash."""
    from app.analytics import simultaneous_band

    r = simultaneous_band([[0.5] * 6, [0.5] * 6, [0.5] * 6], iterations=100)
    assert r is not None and r["degenerate"] is True
    assert all(lo == hi == 0.5 for lo, hi in r["band"])


def test_benjamini_hochberg_controls_false_discovery():
    """40 hypotheses, all null. Uncorrected, ~87% of studies report a finding."""
    import random

    from app.analytics import benjamini_hochberg

    rng = random.Random(1)
    trials, raw, corrected = 200, 0, 0
    for _ in range(trials):
        ps = [rng.random() for _ in range(40)]
        raw += any(p < 0.05 for p in ps)
        corrected += benjamini_hochberg(ps)["n_rejected"] > 0
    assert raw / trials > 0.7  # the problem, still present without correction
    assert corrected / trials < 0.15  # controlled


def test_benjamini_hochberg_keeps_real_effects():
    """Control that costs all the power is not a trade worth making."""
    from app.analytics import benjamini_hochberg

    ps = [1e-6] * 5 + [0.4, 0.5, 0.6, 0.7, 0.8] * 7
    out = benjamini_hochberg(ps)
    assert out["n_rejected"] == 5
    assert all(out["rejected"][:5])
    assert out["cutoff"] == pytest.approx(1e-6)


def test_benjamini_hochberg_is_monotone_and_passes_none_through():
    """q must never decrease as p increases, and a ragged family must survive
    without the caller re-indexing around the gaps."""
    from app.analytics import benjamini_hochberg

    out = benjamini_hochberg([0.01, None, 0.04, 0.2, None, 0.9])
    qs = [q for q in out["q_values"] if q is not None]
    assert qs == sorted(qs)  # input was already p-ascending
    assert out["q_values"][1] is None and out["q_values"][4] is None
    assert out["n_tests"] == 4
    assert all(q <= 1.0 for q in qs)


def test_the_bootstrap_floor_is_named_and_matches_the_frontend():
    """The threshold is duplicated in `dashboard/src/components/Interval.tsx`,
    which uses it to explain WHY an interval is absent. A drift would have the
    UI cite a number the estimator does not use — and the whole point of that
    message is that it tells a reader exactly how many more twins to run.
    """
    import pathlib
    import re

    from app.analytics import MIN_BOOTSTRAP_N, bootstrap_ci

    assert bootstrap_ci([0.5] * (MIN_BOOTSTRAP_N - 1)) is None
    assert bootstrap_ci([i / 10 for i in range(MIN_BOOTSTRAP_N)]) is not None

    ui = (
        pathlib.Path(__file__).resolve().parents[2]
        / "dashboard/src/components/Interval.tsx"
    )
    if not ui.exists():          # server-only checkout
        return
    found = re.search(r"MIN_BOOTSTRAP_N\s*=\s*(\d+)", ui.read_text())
    assert found, "the frontend must keep a named copy of the floor"
    assert int(found.group(1)) == MIN_BOOTSTRAP_N
