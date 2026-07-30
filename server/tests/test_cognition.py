"""Known-answer tests for the computational cognition engine. Each test uses
synthetic data whose ground truth is known analytically or by construction, so
these verify the math — not just that code runs."""

import math
import random

import pytest

from app.cognition import (
    EVENT_SYMBOLS,
    SAMPEN_MAX_SE,
    SAMPEN_MIN_N,
    ez_diffusion,
    full_cognitive_profile,
    habituation_fit,
    hmm_fit,
    hmm_select,
    iter_cognitive_profile,
    lomb_scargle,
    lz76_complexity,
    lz76_profile,
    markov_transitions,
    powerlaw_vs_exponential,
    sample_entropy,
    sample_entropy_detail,
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


# ── audit: LZ76 normalised against its own shuffled surrogates ───────────


def test_lz76_is_anchored_on_surrogates_not_an_asymptote():
    """n / log_a(n) is an n → ∞ limit, and session lengths are not near it.

    Before, on 4-symbol sequences: a perfectly periodic loop scored 0.4089 at
    n=30 while i.i.d. random scored 1.0435 (mean of 200 draws, sd 0.078) — the
    key documented as "1.0 ≈ random" actually ranged over ~[0.4, 1.04], and a
    maximally regular session read as mid-scale.

    After (200 seeded shuffles of the sequence itself): periodic 0.3826 at
    n=30 and 0.0937 at n=200, random centred on 1.0 at both lengths.
    """
    periodic30 = ([0, 1, 2, 3] * 8)[:30]
    rng = random.Random(3)
    random30 = [rng.randrange(4) for _ in range(30)]

    p30 = lz76_profile(periodic30)
    r30 = lz76_profile(random30)
    assert p30["lz76_complexity"] == pytest.approx(0.3826, abs=1e-3)
    assert r30["lz76_complexity"] == pytest.approx(1.0367, abs=1e-3)
    # the periodic sequence must now sit at the BOTTOM of the range, not mid
    assert p30["lz76_complexity"] < 0.5 < r30["lz76_complexity"]
    # and be unmistakable against its own null
    assert p30["lz76_z"] < -5.0 and abs(r30["lz76_z"]) < 3.0

    # the anchor holds at the other end of the length range
    rng = random.Random(3)
    assert lz76_profile([0, 1, 2, 3] * 50)["lz76_complexity"] == pytest.approx(0.0937, abs=1e-3)
    assert lz76_profile([rng.randrange(4) for _ in range(200)])[
        "lz76_complexity"
    ] == pytest.approx(1.0, abs=0.05)


def test_lz76_surrogate_ensemble_is_seeded_and_degenerate_input_is_flagged():
    rng = random.Random(17)
    seq = [rng.randrange(5) for _ in range(120)]
    assert lz76_profile(seq) == lz76_profile(seq)  # deterministic, seeded
    assert lz76_complexity(seq) == lz76_profile(seq)["lz76_complexity"]
    # one symbol has exactly one possible ordering: no sequential information
    only_one = lz76_profile([2] * 30)
    assert only_one["lz76_complexity"] == 0.0 and only_one["lz76_z"] is None
    assert lz76_complexity([0, 1, 0]) is None  # below the length floor


# ── audit: SampEn reported only where it is estimable ────────────────────


def test_sample_entropy_refuses_short_noisy_series():
    """It used to answer at kinematic lengths, and those answers were the
    regular tail of the sampling distribution.

    Measured before, seeded Gaussian noise (m=2, r=0.2σ; reference 2.19 at
    n=1500): report-rate 5.5% at n=12, 17.7% at n=20, 73.8% at n=50, with the
    reported values averaging 0.66 / 1.10 / 2.07 — a biased subsample carrying
    52% / 36% / 23% relative dispersion. After the gate nothing is reported
    below n=100, which is where the 95% CI starts covering the reference at
    its nominal rate (95.0 / 97.0 / 96.5 / 94.5% at n=100 / 120 / 150 / 200,
    against 36.4% at n=50).
    """
    for n in (20, 50, 75, SAMPEN_MIN_N - 1):
        for s in range(6):
            rng = random.Random(4242 + 7 * n + s)
            assert sample_entropy([rng.gauss(0, 1) for _ in range(n)]) is None, n

    rng = random.Random(4242 + 7 * 200)
    d = sample_entropy_detail([rng.gauss(0, 1) for _ in range(200)])
    assert d is not None and d["regime"] == "interior"
    assert d["value"] == pytest.approx(1.8639, abs=1e-3)
    assert d["se"] <= SAMPEN_MAX_SE
    assert d["ci_95"][0] < d["value"] < d["ci_95"][1]


def test_sample_entropy_still_measures_an_exactly_regular_short_series():
    """The length gate must not swallow the one case that IS exact.

    A == B means every m-match extended: SampEn = 0, with one-sided
    uncertainty only, bounded by the rule of three at 3/B. Period-6 over 30
    points has B = 52 matches ⇒ upper 0.0594; the period-8 triangle over 24
    points has B = 20 ⇒ upper 0.1625. Over 3600 seeded Gaussian-noise trials
    at n = 12…200 this branch fired 0 times, so it is not a hole for noise.
    """
    d = sample_entropy_detail([(i % 6) / 5.0 for i in range(30)])
    assert d is not None and d["regime"] == "boundary_rule_of_three"
    assert d["value"] == 0.0 and d["matches_m"] == d["matches_m1"] == 52
    assert d["ci_95"][0] == 0.0
    assert d["ci_95"][1] == pytest.approx(0.0594, abs=1e-3)

    t = sample_entropy_detail([abs((i % 8) - 4) / 4.0 for i in range(24)])
    assert t is not None and t["value"] == 0.0
    assert t["ci_95"][1] == pytest.approx(0.1625, abs=1e-3)


# ── audit: entropy-rate bias in `predictability` ─────────────────────────


def test_predictability_is_not_manufactured_from_short_random_sequences():
    """Plug-in entropy is biased down ⇒ predictability is biased up.

    Measured on uniform i.i.d. sequences (truth 0.000), 500 seeded trials per
    cell, BEFORE the Miller-Madow correction (i.e. with the Jeffreys rewrite
    alone): 0.098 / 0.060 / 0.030 at T=20/40/80 over k=3 symbols, and
    0.106 / 0.089 / 0.062 over k=5. After: 0.017 / 0.011 / 0.005 (k=3) and
    0.008 / 0.005 / 0.003 (k=5). The 200-trial subsets below are those same
    nulls, cheap enough to run in every suite.
    """
    for k, T, ceiling in ((3, 20, 0.03), (4, 40, 0.03), (5, 80, 0.02)):
        preds = []
        for trial in range(200):
            rng = random.Random(1000 * k + 10 * T + trial)
            preds.append(
                markov_transitions([rng.randrange(k) for _ in range(T)], 7)["predictability"]
            )
        mean_pred = sum(preds) / len(preds)
        assert mean_pred < ceiling, (k, T, mean_pred)


def test_the_correction_leaves_a_genuinely_predictable_sequence_alone():
    """Deterministic rows have K̂ = 1, so Miller-Madow adds exactly nothing."""
    short, long = markov_transitions([0, 1] * 15, 7), markov_transitions([0, 1] * 100, 7)
    assert short["predictability"] == pytest.approx(0.7943, abs=1e-3)
    assert long["predictability"] == pytest.approx(0.9548, abs=1e-3)
    assert long["predictability"] > short["predictability"]  # the prior washes out
    # transition counts ship with the estimate: 29 is not enough to lean on
    assert (short["n_transitions"], short["predictability_reliable"]) == (29, False)
    assert (long["n_transitions"], long["predictability_reliable"]) == (199, True)


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


# ── audit: Tobit censoring point, honest R², interval on λ ───────────────


def _censored_decay(trial: int, n: int = 30, lam: float = 0.12, a: float = 3.0):
    """y = A·exp(−λt)·exp(ε), ε ~ N(0, 0.15), with the collector reporting 0
    for anything below its resolution limit of 0.25 (≈29% of the record)."""
    rng = random.Random(31337 + trial)
    ts = [float(i) for i in range(n)]
    ys = []
    for t in ts:
        y = a * math.exp(-lam * t) * math.exp(rng.gauss(0, 0.15))
        ys.append(y if y >= 0.25 else 0.0)
    return ts, ys


def test_tobit_e_step_truncates_at_the_real_censoring_point():
    """A zero reading means y < LOD. LOD/2 is only the starting substitution.

    Conditioning the E-step on ln(LOD/2) told it the censored mass sat a
    factor of two below the instrument's threshold, so the imputed tail was
    placed too low and the slope came out too steep. Over 500 seeded trials of
    the record above, mean λ̂ was 0.13483 against a true 0.120 — +12.4%,
    systematic (trial-to-trial sd 0.0057, so the bias is ~2.6 sd). Truncating
    at ln(LOD) gives 0.11892, −0.9%. The 200-trial subset here measures
    0.11897.
    """
    lams = [habituation_fit(*_censored_decay(i), n_boot=0)["decay_rate_per_s"] for i in range(200)]
    mean_lam = sum(lams) / len(lams)
    assert mean_lam == pytest.approx(0.1190, abs=2e-3)
    assert abs(mean_lam - 0.12) < 0.004  # ≲3%, versus +12.4% before
    assert abs(mean_lam - 0.12) < abs(0.13483 - 0.12)


def test_tobit_r2_is_scored_on_observed_points_only():
    """Imputed points sit exactly on the fitted line by construction, so
    including them scores the model against itself: mean R² 0.974 over all
    points versus 0.962 over the uncensored ones across the 500 trials. The
    reported r2 must be reproducible from the returned line using only the
    points the instrument actually measured."""
    ts, ys = _censored_decay(0)
    f = habituation_fit(ts, ys, n_boot=0)
    assert f is not None and f["censored_fraction"] > 0.2

    b0 = math.log(f["initial_engagement_median"])
    lam = f["decay_rate_per_s"]
    obs = [(t, math.log(y)) for t, y in zip(ts, ys) if y > 0]
    mu = sum(l for _, l in obs) / len(obs)
    ss_tot = sum((l - mu) ** 2 for _, l in obs)
    ss_res = sum((l - (b0 - lam * t)) ** 2 for t, l in obs)
    assert f["r2"] == pytest.approx(1 - ss_res / ss_tot, abs=1e-3)


def test_tobit_reports_an_interval_on_decay_rate_and_half_life():
    """A censored MLE from a 30-point record is not a 5-decimal fact.

    Seeded pairs bootstrap, 200 resamples: measured coverage of the true λ was
    92.8% over 500 trials at a nominal 95%. The cheaper 60-resample setting
    used here measured 92.5% over the first 40 trials; the assertion is set
    well below that so it pins the interval's existence and usefulness, not
    bootstrap noise.
    """
    f = habituation_fit(*_censored_decay(0), n_boot=80)
    lo, hi = f["decay_rate_ci"]
    assert lo < f["decay_rate_per_s"] < hi
    # half-life is log2/λ at the same endpoints, so the order reverses
    hl_lo, hl_hi = f["half_life_ci"]
    assert hl_lo == pytest.approx(math.log(2) / hi, abs=0.01)
    assert hl_hi == pytest.approx(math.log(2) / lo, abs=0.01)
    assert hl_lo < f["half_life_s"] < hl_hi

    covered = sum(
        1
        for i in range(40)
        if (ci := habituation_fit(*_censored_decay(i), n_boot=60)["decay_rate_ci"])
        and ci[0] <= 0.12 <= ci[1]
    )
    assert covered >= 32  # 80% floor under a measured 92.5%


def test_tobit_censoring_point_is_stated_not_assumed():
    """Thresholding and rounding censor at different places.

    The collector thresholds — a velocity sample is |Δy|/Δt with Δy in whole
    pixels, so a zero means nothing moved — and ln(LOD) is right for it. An
    instrument that rounds to a quantum q censors at q/2 instead, and says so
    with `censoring_limit`. On y = 3·exp(−0.15t) rounded to 0.1 (true λ=0.15):
    default 0.1406, `censoring_limit=0.05` 0.14303.
    """
    ts = [float(i) for i in range(41)]
    ys = [round(3.0 * math.exp(-0.15 * t), 1) for t in ts]
    default = habituation_fit(ts, ys, n_boot=0)
    rounded = habituation_fit(ts, ys, n_boot=0, censoring_limit=0.05)
    assert default["censoring_limit_used"] == pytest.approx(0.1)
    assert rounded["censoring_limit_used"] == pytest.approx(0.05)
    assert default["decay_rate_per_s"] == pytest.approx(0.1406, abs=1e-4)
    assert rounded["decay_rate_per_s"] == pytest.approx(0.14303, abs=1e-4)


def test_tobit_changes_nothing_on_an_uncensored_record():
    """No zeros ⇒ no E-step at all, and the observed-only R² is the R²."""
    ts = [i * 2.0 for i in range(30)]
    f = habituation_fit(ts, [3.0 * math.exp(-0.05 * t) for t in ts], n_boot=0)
    assert f["decay_rate_per_s"] == pytest.approx(0.05, abs=1e-4)
    assert f["initial_engagement"] == pytest.approx(3.0, abs=1e-3)
    assert f["censored_fraction"] == 0.0 and f["r2"] == pytest.approx(1.0, abs=1e-4)


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
