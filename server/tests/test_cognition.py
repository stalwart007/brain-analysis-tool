"""Known-answer tests for the computational cognition engine. Each test uses
synthetic data whose ground truth is known analytically or by construction, so
these verify the math — not just that code runs."""

import math
import random

import pytest

from app.cognition import (
    EVENT_SYMBOLS,
    ez_diffusion,
    full_cognitive_profile,
    habituation_fit,
    hmm_fit,
    hmm_select,
    iter_cognitive_profile,
    lomb_scargle,
    lz76_complexity,
    markov_transitions,
    powerlaw_vs_exponential,
    sample_entropy,
    spectral_profile,
)

# ────────────────────────────── EZ-diffusion ─────────────────────────────


def test_ez_diffusion_recovers_published_example():
    """Wagenmakers et al. 2007 worked example: Pc=.802, VRT=.112, MRT=.723
    → v≈.09993, a≈.1399, Ter≈.30003."""
    r = ez_diffusion(mean_rt_s=0.723, var_rt_s2=0.112, p_correct=0.802, n=100)
    assert r is not None
    assert r["drift_rate"] == pytest.approx(0.09994, abs=2e-4)
    assert r["boundary_separation"] == pytest.approx(0.1399, abs=2e-3)
    assert r["non_decision_time_s"] == pytest.approx(0.3000, abs=2e-3)


def test_ez_diffusion_low_accuracy_gives_negative_drift():
    r = ez_diffusion(mean_rt_s=0.8, var_rt_s2=0.15, p_correct=0.2, n=50)
    assert r is not None and r["drift_rate"] < 0


def test_ez_diffusion_refuses_degenerate_input():
    assert ez_diffusion(0.5, 0.0, 0.8, 100) is None
    assert ez_diffusion(0.5, 0.1, 0.8, 3) is None


# ────────────────────────────────── HMM ──────────────────────────────────


def _planted_two_state_sequence(n=400, seed=5):
    """Two sticky latent states: state 0 emits symbol 0 (scroll) 90% of the
    time, state 1 emits symbol 3 (hesitate) 90% of the time."""
    rng = random.Random(seed)
    state, seq, truth = 0, [], []
    for _ in range(n):
        if rng.random() < 0.05:
            state = 1 - state
        truth.append(state)
        main = 0 if state == 0 else 3
        seq.append(main if rng.random() < 0.9 else rng.randrange(len(EVENT_SYMBOLS)))
    return seq, truth


def test_hmm_recovers_planted_structure():
    seq, truth = _planted_two_state_sequence()
    fit = hmm_fit(seq, 2, len(EVENT_SYMBOLS))
    assert fit is not None
    # log-likelihood must be non-decreasing across EM iterations (theory!)
    trace = fit["ll_trace"]
    assert all(b >= a - 1e-6 for a, b in zip(trace, trace[1:]))
    # decoded path must agree with the planted states (up to label swap)
    path = fit["state_path"]
    agree = sum(1 for p, t in zip(path, truth) if p == t) / len(truth)
    assert max(agree, 1 - agree) > 0.85
    # sticky transitions must be recovered as sticky
    A = fit["transition"]
    assert A[0][0] > 0.8 and A[1][1] > 0.8


def test_hmm_bic_prefers_two_states_for_two_state_data():
    seq, _ = _planted_two_state_sequence(n=300)
    sel = hmm_select(seq, len(EVENT_SYMBOLS), k_candidates=(2, 3, 4))
    assert sel is not None
    assert sel["n_states"] == 2  # complexity must pay for itself
    labels = [s["dominant_event"] for s in sel["states"]]
    assert set(labels) == {"scroll", "hesitate"}


def test_hmm_refuses_tiny_sequences():
    assert hmm_fit([0, 1, 0], 4, len(EVENT_SYMBOLS)) is None


# ─────────────────────── Markov / information theory ─────────────────────


def test_markov_stationary_of_known_chain():
    """Deterministic alternation 0,1,0,1… has stationary (.5,.5) restricted to
    the two symbols and near-zero entropy rate (fully predictable)."""
    seq = [0, 1] * 100
    m = markov_transitions(seq, 2)
    assert m["stationary"][0] == pytest.approx(0.5, abs=0.01)
    assert m["entropy_rate_bits"] < 0.15  # Laplace smoothing keeps it just >0
    assert m["predictability"] > 0.85


def test_lz76_periodic_vs_random():
    periodic = [0, 1, 2, 3] * 50
    rng = random.Random(3)
    noisy = [rng.randrange(4) for _ in range(200)]
    cp, cr = lz76_complexity(periodic), lz76_complexity(noisy)
    assert cp is not None and cr is not None
    assert cr > 2.5 * cp  # random stream is far more complex than a loop


def test_sample_entropy_orders_regular_vs_irregular():
    regular = [math.sin(i / 3.0) for i in range(120)]
    rng = random.Random(9)
    irregular = [rng.random() for _ in range(120)]
    sr, si = sample_entropy(regular), sample_entropy(irregular)
    assert sr is not None and si is not None
    assert si > sr  # white noise beats a sine wave in irregularity


# ────────────────────────────── spectral ─────────────────────────────────


def test_lomb_scargle_finds_planted_frequency_in_irregular_samples():
    """A 0.5 Hz sine sampled at *jittered* times must peak near 0.5 Hz."""
    rng = random.Random(11)
    ts = sorted(rng.uniform(0, 30) for _ in range(150))
    ys = [math.sin(2 * math.pi * 0.5 * t) for t in ts]
    freqs = [i * 0.05 for i in range(1, 61)]
    p = lomb_scargle(ts, ys, freqs)
    assert freqs[max(range(len(p)), key=lambda i: p[i])] == pytest.approx(0.5, abs=0.06)


def test_spectral_profile_bands_sum_to_one():
    rng = random.Random(2)
    ts = sorted(rng.uniform(0, 20000) for _ in range(80))
    vs = [math.sin(2 * math.pi * 0.4 * (t / 1000)) + 0.2 * rng.random() for t in ts]
    sp = spectral_profile(ts, vs)
    assert sp is not None
    assert sp["drift_band"] + sp["rhythm_band"] + sp["burst_band"] == pytest.approx(1.0, abs=0.01)
    assert sp["peak_frequency_hz"] == pytest.approx(0.4, abs=0.1)


# ────────────────────────────── foraging ─────────────────────────────────


def test_powerlaw_mle_recovers_alpha():
    """Inverse-CDF sampling of a pure power law with α=2.5 must recover α."""
    rng = random.Random(7)
    xs = [1.0 * (1 - rng.random()) ** (-1 / 1.5) for _ in range(3000)]  # α=2.5
    r = powerlaw_vs_exponential(xs)
    assert r is not None
    assert r["alpha"] == pytest.approx(2.5, abs=0.1)
    assert r["regime"] == "levy_like"


def test_exponential_data_classified_brownian():
    rng = random.Random(13)
    xs = [rng.expovariate(1.0) + 0.05 for _ in range(2000)]
    r = powerlaw_vs_exponential(xs)
    assert r is not None
    assert r["regime"] == "brownian_like"


# ───────────────────────────── habituation ───────────────────────────────


def test_habituation_recovers_decay_rate_and_half_life():
    ts = [i * 2.0 for i in range(30)]
    ys = [3.0 * math.exp(-0.05 * t) for t in ts]
    h = habituation_fit(ts, ys)
    assert h is not None
    assert h["decay_rate_per_s"] == pytest.approx(0.05, abs=1e-3)
    assert h["half_life_s"] == pytest.approx(math.log(2) / 0.05, abs=0.5)
    assert h["r2"] > 0.999


def test_flat_engagement_is_not_habituating():
    h = habituation_fit([float(i) for i in range(20)], [1.0] * 20)
    assert h is not None and not h["habituating"]


# ─────────────────────────── full pipeline ───────────────────────────────


def _rich_synthetic_session(seed=21):
    rng = random.Random(seed)
    t, events, vel = 0.0, [], []
    for i in range(240):
        t += rng.expovariate(1 / 180.0)
        sym = rng.choices(range(len(EVENT_SYMBOLS)), weights=[5, 3, 2, 1, 0.3, 0.7, 1])[0]
        events.append([t, sym])
        vel.append([t, abs(math.sin(t / 900.0)) * math.exp(-t / 60000.0) + 0.05 * rng.random()])
    lats = [rng.gauss(650, 160) for _ in range(30)]
    return events, vel, [max(120.0, x) for x in lats]


def test_full_profile_runs_every_model_on_rich_data():
    events, vel, lats = _rich_synthetic_session()
    p = full_cognitive_profile(
        event_stream=events,
        velocity_series=vel,
        decision_latencies_ms=lats,
        completion_ratio=0.8,
    )
    assert set(p["models_run"]) == {
        "ez_diffusion", "hmm", "information_dynamics", "spectral", "foraging", "habituation",
    }
    assert p["models_skipped"] == []
    s = p["summary"]
    assert "processing_efficiency" in s and "state_occupancy" in s
    assert s["disclaimer"].startswith("behavioral")


def test_pipeline_degrades_gracefully_with_no_data():
    p = full_cognitive_profile()
    assert p["models_run"] == []
    assert len(p["models_skipped"]) == 6


def test_pipeline_streams_stage_frames_in_order():
    events, vel, lats = _rich_synthetic_session()
    frames = list(
        iter_cognitive_profile(
            event_stream=events, velocity_series=vel,
            decision_latencies_ms=lats, completion_ratio=0.8,
        )
    )
    stages = [f["stage"] for f in frames]
    for st in ["ddm", "hmm", "information", "spectral", "foraging", "habituation"]:
        assert st in stages
    assert stages[-1] == "complete"
    # the HMM stage must surface its real EM trace mid-stream
    conv = next(f for f in frames if f["stage"] == "hmm" and f.get("status") == "converged")
    assert len(conv["ll_trace"]) >= 2
