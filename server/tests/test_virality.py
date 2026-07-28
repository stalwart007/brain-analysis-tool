"""Known-answer tests for the Galton-Watson virality engine."""

import pytest

from app.virality import (
    extinction_probability,
    final_size_bins,
    simulate_cascades,
    virality_result,
)


def test_subcritical_process_always_dies():
    # R0 = 4·0.2 = 0.8 < 1 ⇒ extinction certain
    assert extinction_probability([0.2], 4) == pytest.approx(1.0, abs=1e-6)


def test_supercritical_extinction_matches_closed_form():
    """k=2, p=0.75: q solves q=(0.25+0.75q)² ⇒ 0.5625q²−0.625q+0.0625=0
    ⇒ q = 1/9 exactly (the other root is 1)."""
    q = extinction_probability([0.75], 2)
    assert q == pytest.approx(1.0 / 9.0, abs=1e-6)


def test_heterogeneity_raises_extinction_at_equal_mean():
    """Jensen: a 50/50 mix of p=0 and p=0.8 (mean 0.4) must go extinct more
    often than homogeneous p=0.4 with the same R0."""
    q_mix = extinction_probability([0.4], 5)
    q_het = extinction_probability([0.0, 0.8], 5)
    assert q_het > q_mix


def test_simulation_alive_fraction_matches_theory():
    """Long-run survival fraction ≈ 1−q. For k=2, p=0.75: 1−1/9 ≈ 0.888."""
    sims = simulate_cascades([0.75], 2, n_sims=2000, seed=11)
    assert sims["generations"][-1]["alive_frac"] == pytest.approx(8 / 9, abs=0.03)


def test_simulation_subcritical_mean_size_matches_1_over_1_minus_r0():
    """k=4, p=0.15 ⇒ R0=0.6 ⇒ E[size]=1/(1−0.6)=2.5."""
    sims = simulate_cascades([0.15], 4, n_sims=3000, seed=7)
    mean_size = sum(sims["final_sizes"]) / len(sims["final_sizes"])
    assert mean_size == pytest.approx(2.5, abs=0.15)


def test_simulation_is_deterministic_under_seed():
    a = simulate_cascades([0.3, 0.6], 3, n_sims=200, seed=5)
    b = simulate_cascades([0.3, 0.6], 3, n_sims=200, seed=5)
    assert a["final_sizes"] == b["final_sizes"]


def test_final_size_bins_cover_all_samples():
    finals = [1, 1, 2, 3, 5, 9, 17, 40]
    bins = final_size_bins(finals)
    assert sum(int(c) for _, _, c in bins) == len(finals)
    for lo, hi, _ in bins:
        assert hi == lo * 2


def test_virality_result_full_surface():
    ps = [0.05, 0.1, 0.2, 0.15, 0.08, 0.12] * 3  # clearly subcritical at k=6
    r = virality_result(ps, 6, "low")
    assert r["r0"] == pytest.approx(6 * sum(ps) / len(ps), abs=1e-6)
    assert not r["supercritical"]
    assert r["extinction_q_hetero"] == pytest.approx(1.0, abs=1e-4)
    assert r["takeoff_probability"] == pytest.approx(0.0, abs=1e-4)
    assert r["expected_cascade_size"] is not None
    assert r["critical_fanout"] == pytest.approx(1 / (sum(ps) / len(ps)), abs=0.1)
    assert len(r["generations"]) == 20
    assert r["share_ci"] is not None and r["r0_ci"] is not None
    assert r["n_sims"] == 2000 and r["seed"] == 11
    assert "not a prediction" in r["disclaimer"]


def test_virality_result_supercritical_has_takeoff():
    r = virality_result([0.5] * 12, 4, "low")  # R0 = 2
    assert r["supercritical"]
    assert r["expected_cascade_size"] is None
    assert 0.0 < r["extinction_q_hetero"] < 1.0
    assert r["takeoff_probability"] > 0.5
