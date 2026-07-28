"""Computational cognitive modeling over behavioral telemetry.

This is the layer that makes the tool a *brain-pattern analysis* instrument in
the only scientifically honest sense available to a browser: established
cognitive-science models fitted to interaction dynamics. Nothing here is an
LLM guess — every number is a parameter of a published model with a citation:

- EZ-diffusion (Wagenmakers, van der Maas & Grasman 2007): closed-form
  drift-diffusion decision model — drift rate (processing efficiency),
  boundary separation (decision caution), non-decision time.
- Hidden Markov Model (Baum-Welch EM + Viterbi, BIC model selection):
  latent behavioral-state inference over the event stream — the session
  decoded into states like focused / scanning / hesitant / frustrated.
- Information dynamics: Markov transition structure, entropy rate,
  Lempel-Ziv (LZ76) sequence complexity, sample entropy of kinematics.
- Spectral analysis: Lomb-Scargle periodogram (handles the irregular
  sampling real telemetry has) with band-power summaries.
- Foraging dynamics: continuous power-law MLE (Clauset et al. 2009) with a
  likelihood-ratio test vs exponential — Lévy-like vs Brownian exploration.
- Habituation: exponential engagement-decay fit with half-life.

Every function is pure, deterministic (fixed seeds), stdlib-only, and returns
uncertainty alongside point estimates wherever the estimator provides one.
Labels are *behavioral*, never clinical or psychometric verdicts.
"""

from __future__ import annotations

import math
import random
from typing import Iterator, Optional, Sequence

# Event-symbol alphabet shared with the collector. Order defines symbol ids.
EVENT_SYMBOLS = ["scroll", "dwell", "click", "hesitate", "rage", "backtrack", "zone"]
_SYMBOL_ID = {s: i for i, s in enumerate(EVENT_SYMBOLS)}

_EPS = 1e-12

# Minimum observations per free parameter before an HMM is considered
# identifiable. The usual latent-variable rule of thumb is 5-10; 5 is the
# permissive end and still refuses the cases the old `T < 2*K` guard waved
# through (see _hmm_fit_single).
_MIN_OBS_PER_PARAM = 5

_SQRT_2PI = math.sqrt(2.0 * math.pi)


def _norm_pdf(z: float) -> float:
    return math.exp(-0.5 * z * z) / _SQRT_2PI


def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


# ───────────────────────────── EZ-diffusion ──────────────────────────────


#: Minimum decision count before EZ-diffusion is attempted at all. The
#: estimator inverts an RT *variance*, and a variance from 4 observations is
#: not a measurement — the old n ≥ 4 gate emitted v, a and Ter to four decimal
#: places from four numbers, with no interval. Wagenmakers et al. assume a
#: proper RT distribution; 12 is the floor at which the bootstrap interval
#: below stops spanning the entire plausible range.
EZ_MIN_DECISIONS = 12

_EZ_BOOTSTRAP = 400


def _ez_solve(
    mean_rt_s: float, var_rt_s2: float, p_correct: float, n: int, scale: float
) -> Optional[tuple[float, float, float, float, float]]:
    """The closed form itself. Returns (v, a, Ter, MDT, pc) or None if the
    inputs cannot identify the model."""
    if mean_rt_s <= 0 or var_rt_s2 <= 0:
        return None
    # edge correction (standard EZ practice): keep Pc off {0, .5, 1}
    pc = min(max(p_correct, 1.0 / (2 * n)), 1.0 - 1.0 / (2 * n))
    if abs(pc - 0.5) < 1e-9:
        pc += 1.0 / (2 * n)
    s2 = scale * scale
    L = math.log(pc / (1 - pc))
    x = L * (L * pc * pc - L * pc + pc - 0.5) / var_rt_s2
    if x <= 0:
        return None
    v = math.copysign(1.0, pc - 0.5) * scale * (x ** 0.25)  # drift rate
    a = s2 * L / v  # boundary separation
    y = -v * a / s2
    mdt = (a / (2 * v)) * (1 - math.exp(y)) / (1 + math.exp(y))  # mean decision time
    ter = mean_rt_s - mdt  # non-decision time
    return v, a, ter, mdt, pc


def ez_diffusion(
    mean_rt_s: float,
    var_rt_s2: float,
    p_correct: float,
    n: int,
    scale: float = 0.1,
    latencies_s: Optional[Sequence[float]] = None,
) -> Optional[dict]:
    """Closed-form drift-diffusion estimates from latency moments.

    mean_rt_s / var_rt_s2: mean and variance of decision latencies (seconds).
    p_correct: fraction of 'successful' decisions (here: interactions that
    completed rather than were abandoned/backtracked). Returns None when the
    inputs cannot identify the model (too few samples, degenerate variance).

    When the raw `latencies_s` are supplied, a nonparametric bootstrap resamples
    them and refits, so v / a / Ter carry percentile intervals. That matters
    more here than in most places in this codebase: these three numbers are
    passed on to the profiler prompt as the strongest available evidence, and
    a point estimate with no interval reads as far more certain than an RT
    inversion over a few dozen samples can possibly be.
    """
    if n < EZ_MIN_DECISIONS:
        return None
    fit = _ez_solve(mean_rt_s, var_rt_s2, p_correct, n, scale)
    if fit is None:
        return None
    v, a, ter, mdt, pc = fit

    out = {
        "drift_rate": round(v, 4),
        "boundary_separation": round(a, 4),
        "non_decision_time_s": round(ter, 4),
        "mean_decision_time_s": round(mdt, 4),
        "p_correct_corrected": round(pc, 4),
        "n": n,
        "interpretation": _ez_interpret(v, a, pc),
        "model": "EZ-diffusion (Wagenmakers et al. 2007)",
    }

    if latencies_s and len(latencies_s) >= EZ_MIN_DECISIONS:
        rng = random.Random(20260728)  # fixed seed: the interval is reproducible
        m = len(latencies_s)
        vs: list[float] = []
        as_: list[float] = []
        ters: list[float] = []
        for _ in range(_EZ_BOOTSTRAP):
            samp = [latencies_s[rng.randrange(m)] for _ in range(m)]
            bmu = sum(samp) / m
            bvar = sum((x - bmu) ** 2 for x in samp) / max(1, m - 1)
            bfit = _ez_solve(bmu, bvar, p_correct, n, scale)
            if bfit is None:
                continue
            vs.append(bfit[0])
            as_.append(bfit[1])
            ters.append(bfit[2])

        def pct(xs: list[float]) -> Optional[list[float]]:
            if len(xs) < _EZ_BOOTSTRAP // 4:  # too many degenerate refits to trust
                return None
            ys = sorted(xs)
            lo = ys[max(0, int(0.025 * len(ys)))]
            hi = ys[min(len(ys) - 1, int(0.975 * len(ys)))]
            return [round(lo, 4), round(hi, 4)]

        out["drift_rate_ci"] = pct(vs)
        out["boundary_separation_ci"] = pct(as_)
        out["non_decision_time_ci"] = pct(ters)
        out["bootstrap_resamples"] = _EZ_BOOTSTRAP

    return out


def _ez_interpret(v: float, a: float, pc: float) -> str:
    """Read the fit out honestly.

    Note the coupling, because it is easy to over-read these two numbers as
    independent traits: EZ gives a = s²·L/v with L = logit(Pc), so at a fixed
    observed accuracy the boundary separation is a deterministic function of the
    drift rate. Speed and caution here are two views of one estimated quantity,
    not two findings — at Pc = 0.75, for instance, "efficient" and "deliberate"
    cannot both hold, and reporting them as separate readings implied evidence
    that is not there.
    """
    speed = "efficient" if abs(v) > 0.25 else "effortful"
    caution = "deliberate" if a > 0.11 else "impulsive"
    return (
        f"{speed} evidence accumulation (v={v:.2f}) with "
        f"{caution} response thresholds (a={a:.2f}); "
        f"v and a are coupled through the observed accuracy (Pc={pc:.2f})"
    )


# ─────────────────────────────── HMM ─────────────────────────────────────


class _LCG:
    """Tiny deterministic PRNG so EM initialisation is reproducible."""

    def __init__(self, seed: int):
        self.state = seed & 0x7FFFFFFF or 1

    def next(self) -> float:
        self.state = (1103515245 * self.state + 12345) % (1 << 31)
        return self.state / float(1 << 31)


def _row_normalise(row: list[float]) -> list[float]:
    t = sum(row)
    return [x / t for x in row] if t > 0 else [1.0 / len(row)] * len(row)


def hmm_fit(
    observations: Sequence[int],
    n_states: int,
    n_symbols: int,
    max_iter: int = 60,
    tol: float = 1e-4,
    seed: int = 17,
    restarts: int = 3,
) -> Optional[dict]:
    """EM is only locally convergent — run `restarts` independent seeded
    initialisations and keep the best final log-likelihood."""
    best = None
    for r in range(max(1, restarts)):
        fit = _hmm_fit_single(observations, n_states, n_symbols, max_iter, tol, seed + 53 * r)
        if fit and (best is None or fit["log_likelihood"] > best["log_likelihood"]):
            best = fit
    if best is not None:
        best["em_restarts"] = max(1, restarts)
    return best


def _forward_scaled(
    obs: Sequence[int], A: list[list[float]], B: list[list[float]], pi: list[float]
) -> tuple[float, list[list[float]], list[float]]:
    """Scaled forward recursion. Returns (log-likelihood, alpha, scale factors).

    Factored out of the EM loop because the log-likelihood has to be computable
    for an *arbitrary* parameter triple, not only as a by-product of the sweep
    that produced it — see the max_iter-exhaustion path below.
    """
    K, T = len(pi), len(obs)
    alpha = [[0.0] * K for _ in range(T)]
    c = [0.0] * T
    for i in range(K):
        alpha[0][i] = pi[i] * B[i][obs[0]]
    c[0] = sum(alpha[0]) or _EPS
    alpha[0] = [x / c[0] for x in alpha[0]]
    for t in range(1, T):
        o = obs[t]
        for j in range(K):
            alpha[t][j] = B[j][o] * sum(alpha[t - 1][i] * A[i][j] for i in range(K))
        c[t] = sum(alpha[t]) or _EPS
        alpha[t] = [x / c[t] for x in alpha[t]]
    return sum(math.log(ct) for ct in c), alpha, c


def _hmm_fit_single(
    observations: Sequence[int],
    n_states: int,
    n_symbols: int,
    max_iter: int = 60,
    tol: float = 1e-4,
    seed: int = 17,
) -> Optional[dict]:
    """Baum-Welch EM for a discrete-emission HMM, with per-timestep scaling.

    Returns transition matrix A, emission matrix B, initial distribution pi,
    the log-likelihood trace across EM iterations, and the Viterbi state path.

    Two corrections over the naive version, both of which changed which model
    BIC selected:

    **Free parameters are counted against the OBSERVED alphabet.** The emission
    block was charged K·(M−1) with M = len(EVENT_SYMBOLS) = 7, unconditionally.
    But a session that only ever emitted {scroll, click, hesitate} gives every
    unobserved symbol a count of exactly the _EPS pseudo-count in the M-step —
    those cells are pinned at ~0, not estimated from data, and charging BIC for
    them is charging for parameters the fit never had. At T = 30 the inflation
    is decisive: K=4 was billed 39 parameters instead of 23, so
    n_params·ln(T) alone put K=4 at +136 BIC and K>2 was unreachable no matter
    what the likelihood said. On a planted 3-state / 3-symbol sequence at
    T=170 the old count picked K=2 (bic_by_k 331.4 / 339.6 / 401.8); the
    observed-alphabet count picks the true K=3 (290.3 / 278.0 / 319.6).

    **The fit guard is an identifiability check, not a length check.**
    `T < 2*K` let K=4 over a 7-symbol alphabet fit 39 free parameters to 8
    observations — 0.2 observations per parameter — and return a confident
    state decoding. The guard is now observations-per-parameter.

    **The returned log-likelihood belongs to the returned parameters.**
    `ll_trace.append(ll)` runs BEFORE the M-step, so on the convergence path
    (break) the last trace entry does describe the parameters we return, but on
    the max_iter-exhaustion path A/B/pi have had one more M-step applied than
    ll_trace[-1] reflects. BIC was then computed from a likelihood the returned
    model does not have: at max_iter=8 on a 3-state sequence the reported LL was
    −217.9651 while the returned parameters actually score −217.9274. Small in
    absolute terms, systematically anti-conservative in sign (EM only ever
    improves), and free to fix — one extra forward pass.
    """
    T = len(observations)
    K, M = n_states, n_symbols
    if K < 1 or T < 2 * K:
        return None
    # Only symbols the session actually emitted carry estimable emission mass.
    m_obs = len({o for o in observations})
    n_params = K * (K - 1) + K * max(0, m_obs - 1) + (K - 1)
    if T < _MIN_OBS_PER_PARAM * n_params:
        return None
    rng = _LCG(seed + K * 101)
    # perturbed-uniform init keeps EM off the saddle point at exact uniformity
    A = [_row_normalise([1.0 + 0.2 * rng.next() for _ in range(K)]) for _ in range(K)]
    B = [_row_normalise([1.0 + 0.5 * rng.next() for _ in range(M)]) for _ in range(K)]
    pi = _row_normalise([1.0 + 0.2 * rng.next() for _ in range(K)])

    ll_trace: list[float] = []
    converged = False
    for _ in range(max_iter):
        ll, alpha, c = _forward_scaled(observations, A, B, pi)
        ll_trace.append(ll)
        if len(ll_trace) > 1 and abs(ll_trace[-1] - ll_trace[-2]) < tol:
            converged = True
            break
        # backward with the same scaling constants
        beta = [[0.0] * K for _ in range(T)]
        beta[T - 1] = [1.0 / c[T - 1]] * K
        for t in range(T - 2, -1, -1):
            o1 = observations[t + 1]
            for i in range(K):
                beta[t][i] = (
                    sum(A[i][j] * B[j][o1] * beta[t + 1][j] for j in range(K)) / c[t]
                )
        # E-step accumulators
        gamma = [[alpha[t][i] * beta[t][i] * c[t] for i in range(K)] for t in range(T)]
        gamma = [_row_normalise(g) for g in gamma]
        xi_sum = [[0.0] * K for _ in range(K)]
        for t in range(T - 1):
            o1 = observations[t + 1]
            denom = sum(
                alpha[t][i] * A[i][j] * B[j][o1] * beta[t + 1][j]
                for i in range(K)
                for j in range(K)
            ) or _EPS
            for i in range(K):
                for j in range(K):
                    xi_sum[i][j] += (
                        alpha[t][i] * A[i][j] * B[j][o1] * beta[t + 1][j] / denom
                    )
        # M-step
        pi = _row_normalise([gamma[0][i] + _EPS for i in range(K)])
        A = [_row_normalise([xi_sum[i][j] + _EPS for j in range(K)]) for i in range(K)]
        Bn = [[_EPS] * M for _ in range(K)]
        for t in range(T):
            for i in range(K):
                Bn[i][observations[t]] += gamma[t][i]
        B = [_row_normalise(Bn[i]) for i in range(K)]

    # Exhaustion path: the loop body appended ll BEFORE its M-step, so A/B/pi
    # are now one M-step ahead of ll_trace[-1]. Score the parameters we are
    # actually about to return. (EM is monotone, so this entry can only be ≥ the
    # previous one and the trace stays non-decreasing.)
    if not converged:
        ll_trace.append(_forward_scaled(observations, A, B, pi)[0])

    path = _viterbi(observations, A, B, pi)
    bic = -2.0 * ll_trace[-1] + n_params * math.log(T)
    return {
        "n_states": K,
        "log_likelihood": round(ll_trace[-1], 4),
        "ll_trace": [round(x, 4) for x in ll_trace],
        "bic": round(bic, 4),
        "n_params": n_params,
        "n_symbols_observed": m_obs,
        "converged": converged,
        "transition": [[round(x, 4) for x in row] for row in A],
        "emission": [[round(x, 4) for x in row] for row in B],
        "initial": [round(x, 4) for x in pi],
        "state_path": path,
        "em_iterations": len(ll_trace),
    }


def _viterbi(
    obs: Sequence[int], A: list[list[float]], B: list[list[float]], pi: list[float]
) -> list[int]:
    K, T = len(pi), len(obs)
    lg = lambda x: math.log(x + _EPS)  # noqa: E731
    delta = [[lg(pi[i]) + lg(B[i][obs[0]]) for i in range(K)]]
    back: list[list[int]] = []
    for t in range(1, T):
        row, brow = [], []
        for j in range(K):
            best_i = max(range(K), key=lambda i: delta[-1][i] + lg(A[i][j]))
            row.append(delta[-1][best_i] + lg(A[best_i][j]) + lg(B[j][obs[t]]))
            brow.append(best_i)
        delta.append(row)
        back.append(brow)
    path = [max(range(K), key=lambda i: delta[-1][i])]
    for brow in reversed(back):
        path.append(brow[path[-1]])
    return path[::-1]


def hmm_select(
    observations: Sequence[int],
    n_symbols: int,
    k_candidates: Sequence[int] = (2, 3, 4),
) -> Optional[dict]:
    """Fit each candidate state count; keep the BIC winner. Honest model
    selection — complexity must pay for itself in likelihood.

    Candidates that fail _hmm_fit_single's observations-per-parameter guard are
    simply absent from `bic_by_k`: "we could not identify K=4 from this many
    events" is a different statement from "K=4 lost on BIC", and collapsing the
    two is how a session with 30 events used to produce a confident four-state
    decoding.
    """
    fits = [f for k in k_candidates if (f := hmm_fit(observations, k, n_symbols))]
    if not fits:
        return None
    best = min(fits, key=lambda f: f["bic"])
    best["bic_by_k"] = {str(f["n_states"]): f["bic"] for f in fits}
    best["n_params_by_k"] = {str(f["n_states"]): f["n_params"] for f in fits}
    best["k_candidates_identifiable"] = [f["n_states"] for f in fits]
    best["states"] = _label_states(best["emission"])
    return best


def _label_states(emission: list[list[float]]) -> list[dict]:
    """Name each latent state by its emission profile. Behavioral labels only."""
    naming = {
        "scroll": "scanning",
        "dwell": "focused reading",
        "click": "goal pursuit",
        "hesitate": "hesitant deliberation",
        "rage": "frustration",
        "backtrack": "re-orientation",
        "zone": "exploration",
    }
    out = []
    for row in emission:
        top = max(range(len(row)), key=lambda m: row[m])
        symbol = EVENT_SYMBOLS[top] if top < len(EVENT_SYMBOLS) else str(top)
        out.append(
            {
                "label": naming.get(symbol, symbol),
                "dominant_event": symbol,
                "dominance": round(row[top], 4),
            }
        )
    return out


# ───────────────────── Markov / information dynamics ─────────────────────


def markov_transitions(seq: Sequence[int], n_symbols: int) -> dict:
    """First-order transition matrix, stationary distribution (power iteration)
    and entropy rate H = -Σ_i π_i Σ_j P_ij log2 P_ij.

    Two things here are deliberate, and the previous version got both wrong:

    **The prior is Jeffreys (½), and it is confined to the observed alphabet.**
    A flat unit prior over all 7×7 = 49 cells puts 49 pseudo-counts against the
    ~30–100 real transitions a typical session provides — the prior wins. A
    perfectly deterministic alternating sequence scored 0.24 predictability,
    and that number is handed to the profiler as "the strongest evidence
    available". Restricting the prior to the symbols the person actually
    emitted, at Jeffreys strength, makes the same sequence score ≈ 1.

    **Normalisation is against the observed alphabet.** Dividing by log2(7)
    conflates two different things — "this person is predictable" and "this
    person only ever does two things". Relative to the repertoire they actually
    use is the quantity a behavioural profile wants. `n_symbols_observed` is
    returned so a consumer can tell the two apart.

    The returned matrix keeps the full n_symbols shape so the UI's event labels
    still line up; unobserved rows are simply all-zero.
    """
    observed = sorted({s for s in seq if 0 <= s < n_symbols})
    k = len(observed)
    if k == 0:
        return {
            "transition": [[0.0] * n_symbols for _ in range(n_symbols)],
            "stationary": [0.0] * n_symbols,
            "entropy_rate_bits": 0.0,
            "predictability": 0.0,
            "n_symbols_observed": 0,
        }

    index = {s: i for i, s in enumerate(observed)}
    # Jeffreys prior (½), only over the observed sub-alphabet
    counts = [[0.5] * k for _ in range(k)]
    for a, b in zip(seq, seq[1:]):
        if a in index and b in index:
            counts[index[a]][index[b]] += 1.0
    Pk = [_row_normalise(row) for row in counts]

    pi = [1.0 / k] * k
    for _ in range(200):
        nxt = [sum(pi[i] * Pk[i][j] for i in range(k)) for j in range(k)]
        if max(abs(a - b) for a, b in zip(pi, nxt)) < 1e-12:
            pi = nxt
            break
        pi = nxt

    h_rate = -sum(
        pi[i] * sum(Pk[i][j] * math.log2(Pk[i][j] + _EPS) for j in range(k))
        for i in range(k)
    )
    max_h = math.log2(k) if k > 1 else 0.0

    # embed back into the full-alphabet shape the UI renders
    P_full = [[0.0] * n_symbols for _ in range(n_symbols)]
    pi_full = [0.0] * n_symbols
    for a, i in index.items():
        pi_full[a] = pi[i]
        for b, j in index.items():
            P_full[a][b] = Pk[i][j]

    return {
        "transition": [[round(x, 4) for x in row] for row in P_full],
        "stationary": [round(x, 4) for x in pi_full],
        "entropy_rate_bits": round(h_rate, 4),
        # k == 1 ⇒ the person did exactly one kind of thing: perfectly predictable
        "predictability": round(1.0 - h_rate / max_h, 4) if max_h > 0 else 1.0,
        "n_symbols_observed": k,
    }


def lz76_complexity(seq: Sequence[int]) -> Optional[float]:
    """Lempel-Ziv 1976 phrase count, normalised by n / log_a(n) so 1.0 ≈ the
    complexity of an i.i.d. random sequence over the same alphabet."""
    n = len(seq)
    if n < 8:
        return None
    alphabet = max(len(set(seq)), 2)
    s = list(seq)
    phrases, i, k, l = 1, 0, 1, 1
    k_max = 1
    while True:
        if s[i + k - 1] == s[l + k - 1]:
            k += 1
            if l + k > n:
                phrases += 1
                break
        else:
            k_max = max(k, k_max)
            i += 1
            if i == l:
                phrases += 1
                l += k_max
                if l + 1 > n:
                    break
                i, k, k_max = 0, 1, 1
            else:
                k = 1
    norm = n / (math.log(n, alphabet) if alphabet > 1 else 1.0)
    return round(phrases / norm, 4)


def sample_entropy(xs: Sequence[float], m: int = 2, r_factor: float = 0.2) -> Optional[float]:
    """SampEn(m, r) = −ln(A/B): irregularity of a kinematic series. Higher =
    less self-similar motion. Richman & Moorman (2000) formulation.

    **A and B must be counted over the SAME template set.** The previous form
    used `range(n - mm)` inside the counter, so the length-m pass ranged over
    N−m template origins and the length-(m+1) pass over N−m−1. A/B is then a
    ratio of two differently-sized samples, not the conditional probability
    "two sequences that matched for m points still match at m+1" that SampEn is
    defined to be — the m+1 count is systematically short by one template, so
    the ratio is biased low and the entropy biased high. Richman & Moorman
    restrict both to N−m vectors precisely to remove this (it is the same
    bookkeeping choice that makes SampEn unbiased where ApEn is not).

    The bias is not academic on short series. A perfectly periodic sequence has
    SampEn ≡ 0 by construction — every m-match extends to an m+1-match, so
    A = B. Before this fix, `(i % 6) / 5` repeated over 30 points returned
    0.08 and a period-8 triangle wave returned 0.1054: spurious irregularity
    manufactured entirely by the off-by-one, on the most regular input that
    exists. Longer series dilute it but do not remove it (a 120-point sine went
    0.2905 → 0.2576).
    """
    n = len(xs)
    if n < m + 2:
        return None
    mu = sum(xs) / n
    sd = math.sqrt(sum((x - mu) ** 2 for x in xs) / n)
    if sd == 0:
        return 0.0
    r = r_factor * sd
    # N−m template origins, shared by both passes. The longest window read is
    # xs[(n-m-1) + m] = xs[n-1], so the m+1 pass stays in bounds.
    n_templates = n - m

    def count(mm: int) -> int:
        c = 0
        for i in range(n_templates):
            for j in range(i + 1, n_templates):
                if all(abs(xs[i + k] - xs[j + k]) <= r for k in range(mm)):
                    c += 1
        return c

    b, a = count(m), count(m + 1)
    if a == 0 or b == 0:
        return None
    # + 0.0 normalises the -0.0 that a perfectly regular series (A == B) yields.
    return round(-math.log(a / b), 4) + 0.0


# ─────────────────────────── spectral analysis ───────────────────────────


def lomb_scargle(
    times_s: Sequence[float], values: Sequence[float], freqs_hz: Sequence[float]
) -> list[float]:
    """Classic Lomb-Scargle periodogram — spectral power for *irregularly*
    sampled series, which is what real interaction telemetry is."""
    n = len(times_s)
    mu = sum(values) / n
    ys = [v - mu for v in values]
    powers = []
    for f in freqs_hz:
        w = 2 * math.pi * f
        s2 = sum(math.sin(2 * w * t) for t in times_s)
        c2 = sum(math.cos(2 * w * t) for t in times_s)
        tau = math.atan2(s2, c2) / (2 * w) if w > 0 else 0.0
        cc = ss = yc = ysn = 0.0
        for t, y in zip(times_s, ys):
            arg = w * (t - tau)
            cv, sv = math.cos(arg), math.sin(arg)
            yc += y * cv
            ysn += y * sv
            cc += cv * cv
            ss += sv * sv
        p = 0.5 * ((yc * yc / (cc + _EPS)) + (ysn * ysn / (ss + _EPS)))
        powers.append(p)
    return powers


# Behavioral (not neural) bands, in Hz: slow drift, reading rhythm, saccadic
# burst. Widths are deliberately unequal — which is exactly why the summaries
# must be densities, not raw shares.
_SPECTRAL_BANDS: tuple[tuple[str, float, float], ...] = (
    ("drift", 0.0, 0.3),
    ("rhythm", 0.3, 1.0),
    ("burst", 1.0, 3.01),
)
_SPECTRAL_BINS = 60
_SPECTRAL_F_CEILING = 3.0  # above the burst band nothing behavioral lives


def spectral_profile(times_ms: Sequence[float], values: Sequence[float]) -> Optional[dict]:
    """Band powers of scroll kinematics, as **power densities**, over a
    frequency grid the record can actually support.

    Bands are behavioral, not neural: slow drift (<0.3 Hz), reading rhythm
    (0.3-1 Hz), saccadic burst (1-3 Hz). Two things have to be right here, and
    getting either wrong pushes every session toward the same verdict:

    **Band summaries must be bandwidth-normalised.** On a fixed
    0.05-Hz / 60-bin grid the three bands get 5, 14 and 41 bins. Summing raw
    power inside each and dividing by the grand total therefore measures how
    *wide* a band is at least as much as how much energy is in it: white noise —
    flat by definition, no rhythm at any scale — reported drift 0.086, rhythm
    0.231, burst 0.683 (mean over 60 realisations) and read out as "saccadic
    burst dominant". Dividing each band's power by its bin count first, then
    normalising the three densities, makes them comparable; the same white noise
    now returns 0.339 / 0.330 / 0.331. The three still sum to 1, so the
    quantity remains a share — but a share of *density*, which is the thing a
    reader is already assuming it is.

    **The grid ignored the record.** 0.05-3.0 Hz was emitted regardless of how
    long the session was or how fast it was sampled. A 1.15-second scroll burst
    was assigned a drift band spanning 0.05-0.25 Hz — periods of 4 to 20
    seconds, 4× to 20× longer than the entire record — and duly reported
    drift 0.094: a fitted amplitude for an oscillation that never completed a
    single cycle. Lomb-Scargle will happily return a number there; it is
    unidentifiable, not small. The grid is now bounded below by 1/span (one
    complete cycle inside the record) and above by the Nyquist analog
    1/(2·median Δt) of the actual sampling, capped at the top of the burst
    band. Bands with no bins left inside those limits return **None** rather
    than 0.0 — "not resolvable from this record" is not the same claim as "no
    energy there", and 0.0 would be read as the second.
    """
    if len(values) < 16:
        return None
    t0 = times_ms[0]
    ts = [(t - t0) / 1000.0 for t in times_ms]
    span = ts[-1] - ts[0]
    if span <= 1.0:
        return None
    deltas = sorted(b - a for a, b in zip(ts, ts[1:]) if b > a)
    if not deltas:
        return None
    median_dt = deltas[len(deltas) // 2]
    f_min = 1.0 / span
    f_max = min(_SPECTRAL_F_CEILING, 1.0 / (2.0 * median_dt))
    # Need real dynamic range, not two adjacent bins, before any of this means
    # anything.
    if f_max <= 1.5 * f_min:
        return None

    df = (f_max - f_min) / (_SPECTRAL_BINS - 1)
    freqs = [f_min + i * df for i in range(_SPECTRAL_BINS)]
    p = lomb_scargle(ts, values, freqs)

    # Mean power per bin = power density (the grid is uniform, so bin count is
    # proportional to bandwidth and the two normalisations coincide).
    densities: dict[str, Optional[float]] = {}
    bin_counts: dict[str, int] = {}
    for name, lo, hi in _SPECTRAL_BANDS:
        in_band = [pw for f, pw in zip(freqs, p) if lo <= f < hi]
        bin_counts[name] = len(in_band)
        densities[name] = (sum(in_band) / len(in_band)) if in_band else None
    total = sum(d for d in densities.values() if d is not None) or _EPS

    peak_i = max(range(len(p)), key=lambda i: p[i])
    out: dict = {
        f"{name}_band": (round(d / total, 4) if d is not None else None)
        for name, d in densities.items()
    }
    out.update(
        {
            "peak_frequency_hz": round(freqs[peak_i], 3),
            "frequency_range_hz": [round(f_min, 4), round(f_max, 4)],
            "band_bin_counts": bin_counts,
            "median_sample_interval_s": round(median_dt, 4),
            "span_s": round(span, 3),
            "n_samples": len(values),
            "method": (
                "Lomb-Scargle periodogram (irregular sampling), bandwidth-normalised "
                "band densities over a 1/span … 1/(2·median Δt) grid"
            ),
        }
    )
    return out


# ─────────────────────────── foraging dynamics ───────────────────────────


def _powerlaw_tail(tail: Sequence[float], xmin: float) -> Optional[tuple[float, float]]:
    """MLE α and the KS distance for a continuous power law above `xmin`.

    Returns (alpha, ks_distance); None if the tail is degenerate.
    """
    n = len(tail)
    if n < 10 or xmin <= 0:
        return None
    slog = sum(math.log(x / xmin) for x in tail)
    if slog <= 0:
        return None
    alpha = 1.0 + n / slog
    # KS between the empirical CDF and the fitted CDF F(x) = 1 − (x/xmin)^(1−α)
    d = 0.0
    for i, x in enumerate(tail):  # tail must be sorted ascending
        f = 1.0 - (x / xmin) ** (1.0 - alpha)
        d = max(d, abs((i + 1) / n - f), abs(f - i / n))
    return alpha, d


def powerlaw_vs_exponential(xs: Sequence[float]) -> Optional[dict]:
    """Continuous power-law MLE (Clauset et al. 2009) against an exponential
    alternative, decided by a **Vuong likelihood-ratio test**.

    Heavy-tailed step/interval distributions are the Lévy-flight signature of
    exploratory foraging; exponential tails are Brownian-like local search.

    Two corrections over the naive version, both of which changed verdicts:

    **x_min is estimated, not assumed.** Setting `xmin = min(xs)` fits the power
    law to the entire sample — which is the exact failure mode Clauset's method
    exists to prevent, since the *body* of a heavy-tailed distribution is not
    power-law. On cascade sizes with a large atom at 1, those x = xmin points
    contribute log(1) = 0 to the sum and inflate α = 1 + n/Σlog wildly; a real
    heavy tail was reported as α ≈ 10, far outside any Lévy range. x_min is now
    chosen by minimising the KS distance over candidate values.

    **The comparison is a test, not a sign check.** Previously two non-nested
    models were separated by the sign of a per-sample mean, with no variance,
    no p-value, and no way to say "indistinguishable" — so genuinely ambiguous
    data coin-flipped between regimes. Vuong's normalised LR gives a z and a
    p-value, and an inconclusive result now says so.

    Also returns the standard error of α, which is (α−1)/√n in closed form and
    was simply never reported.
    """
    xs = sorted(x for x in xs if x > 0)
    n_all = len(xs)
    if n_all < 10:
        return None

    # ── estimate x_min by KS minimisation over candidate cut-points ──────
    #
    # The search is floored at 10% of the sample. Unconstrained KS minimisation
    # is right for *fitting* a power law but wrong for *comparing* two models:
    # on data that is not power-law it happily walks x_min out to the 96th
    # percentile, because a power law fits the extreme tail of almost anything.
    # That left 88 of 2000 clean exponential samples in the comparison, and the
    # Vuong test — correctly — could no longer separate the models. Keeping a
    # floor preserves the power to reach a verdict while still refusing to fit
    # the body of the distribution.
    min_tail = max(10, int(0.10 * n_all))
    candidates = sorted(set(xs))
    best: Optional[tuple[float, float, list[float]]] = None  # (ks, alpha, tail)
    xmin_best = xs[0]
    for c in candidates:
        if c <= 0:
            continue
        tail = [x for x in xs if x >= c]
        if len(tail) < min_tail:
            break  # candidates ascend, so every later tail is smaller still
        fit = _powerlaw_tail(tail, c)
        if fit is None:
            continue
        alpha_c, ks = fit
        if best is None or ks < best[0]:
            best = (ks, alpha_c, tail)
            xmin_best = c
    if best is None:
        return None
    ks_d, alpha, tail = best
    n = len(tail)
    xmin = xmin_best

    # ── exponential alternative on the same tail ─────────────────────────
    mean_tail = sum(tail) / n
    if mean_tail - xmin <= _EPS:
        return None
    lam = 1.0 / (mean_tail - xmin)

    # ── Vuong: pointwise log-likelihood difference ───────────────────────
    diffs = [
        (math.log(alpha - 1.0) - math.log(xmin) - alpha * math.log(x / xmin))
        - (math.log(lam) - lam * (x - xmin))
        for x in tail
    ]
    r_total = sum(diffs)
    mean_d = r_total / n
    var_d = sum((d - mean_d) ** 2 for d in diffs) / n
    sigma = math.sqrt(var_d)
    if sigma < _EPS:
        z, p_value = 0.0, 1.0
    else:
        z = r_total / (math.sqrt(n) * sigma)
        p_value = math.erfc(abs(z) / math.sqrt(2.0))

    inconclusive = p_value > 0.10
    levy_like = (not inconclusive) and r_total > 0 and alpha <= 3.0

    if inconclusive:
        regime = "indistinguishable"
        interpretation = (
            "the tail does not separate a power law from an exponential at this "
            f"sample size (p={p_value:.2f}) — treat the regime as unresolved"
        )
    elif levy_like:
        regime = "levy_like"
        interpretation = "heavy-tailed exploration — long relocations punctuate local scanning"
    else:
        regime = "brownian_like"
        interpretation = "exponential-tailed local search — incremental, nearby moves"

    return {
        "alpha": round(alpha, 4),
        # closed-form SE for the Hill-type MLE; at n=10 this is ≈0.6, which is
        # why a bare α to 4 decimals was always overstating what is known
        "alpha_se": round((alpha - 1.0) / math.sqrt(n), 4),
        "xmin": round(xmin, 4),
        "ks_distance": round(ks_d, 4),
        "loglik_ratio_per_sample": round(mean_d, 4),
        "vuong_z": round(z, 4),
        "p_value": round(p_value, 4),
        "regime": regime,
        "interpretation": interpretation,
        "n": n,
        "n_total": n_all,
    }


# ───────────────────────────── habituation ───────────────────────────────


def habituation_fit(
    times_s: Sequence[float], engagement: Sequence[float], em_iterations: int = 40
) -> Optional[dict]:
    """Exponential decay fit y = A·exp(-λt) on the log scale. λ>0 means the
    stimulus is losing grip; half-life says how fast.

    Two subtleties, both of which move the headline numbers:

    **Zero engagement is a measurement, not a missing value.** The tempting
    filter `if y > 0` deletes every sample where the person stopped moving —
    precisely the decayed tail this model exists to measure. That deletion is
    not random with respect to the fit: near the resolution limit only the
    positive-noise realisations survive, so the retained points sit above the
    true line at the right-hand end and the slope flattens. On a quantised
    y = 3·exp(−0.15t) series that form yields λ = 0.1406 against a true 0.15 —
    habituation under-reported — and yields *the identical* 0.1406 / A = 2.742
    for records of 32, 36, 41 and 50 samples, because everything past the point
    where the signal fell below the quantum is discarded. Twenty extra seconds
    of recorded disengagement would change nothing.

    So zeros are kept and treated as what they are: **left-censored at the
    resolution limit**, which here is the smallest positive value the collector
    reported. The fit is the standard EM for a censored normal regression
    (Tobit, 1958): censored points enter at the conventional LOD/2 substitution
    and are then replaced each pass by their conditional expectation under the
    current line, E[ln y | ln y < ln c] = μ − σ·φ(α)/Φ(α). This converges to
    the censored MLE and, unlike naive substitution, does not degrade as the
    censored fraction grows — λ came out 0.1430 at 12%, 22%, 32% and 44%
    censoring where LOD/2-only substitution decayed 0.1374 → 0.0870.

    **exp(intercept) estimates the median, not A.** With ln y = ln A − λt + ε
    and ε ~ N(0, σ²), E[y] = A·exp(−λt)·E[e^ε] = A·exp(−λt)·exp(σ²/2). The
    retransformed intercept is therefore biased low by exactly the factor the
    scatter contributes; Duan's smearing / half-σ² correction restores it.
    Uncorrected the same series gave Â = 1.96 against a true 3.0.

    `censored_fraction` is returned because a censored MLE from mostly-censored
    data is still a weak estimate, and the consumer should be able to see that.
    """
    ts = [float(t) for t in times_s]
    ys = [float(y) for y in engagement]
    n = min(len(ts), len(ys))
    ts, ys = ts[:n], ys[:n]
    if n < 5:
        return None

    positive = [y for y in ys if y > 0]
    # ≥3 uncensored points are needed to identify slope, intercept and σ.
    if len(positive) < 3:
        return None
    # Limit of detection = the finest positive value the instrument reported;
    # LOD/2 is the conventional substitution point and our EM starting value.
    lod = min(positive)
    log_c = math.log(lod / 2.0)
    censored = [y <= 0 for y in ys]
    n_censored = sum(censored)
    logs = [math.log(y) if y > 0 else log_c for y in ys]

    sx = sum(ts)
    sxx = sum(t * t for t in ts)
    denom = n * sxx - sx * sx
    if abs(denom) < _EPS:
        return None

    slope = intercept = 0.0
    for _ in range(max(1, em_iterations)):
        # M-step: OLS on the current (partly imputed) log series
        sy = sum(logs)
        sxy = sum(t * l for t, l in zip(ts, logs))
        slope = (n * sxy - sx * sy) / denom
        intercept = (sy - slope * sx) / n
        if n_censored == 0:
            break
        # σ from the uncensored residuals only — imputed points carry no
        # independent information about scatter and would shrink it artificially
        obs_res = [
            l - (intercept + slope * t)
            for t, l, c in zip(ts, logs, censored)
            if not c
        ]
        k = len(obs_res)
        sigma = math.sqrt(max(sum(r * r for r in obs_res) / max(1, k - 2), _EPS))
        # E-step: replace each censored point by E[ln y | ln y < ln c]
        moved = 0.0
        updated: list[float] = []
        for t, l, c in zip(ts, logs, censored):
            if not c:
                updated.append(l)
                continue
            mu = intercept + slope * t
            alpha = (log_c - mu) / sigma
            tail = _norm_cdf(alpha)
            e = mu - sigma * _norm_pdf(alpha) / tail if tail > _EPS else log_c
            e = min(e, log_c)  # a censored observation lies below the limit
            moved = max(moved, abs(e - l))
            updated.append(e)
        logs = updated
        if moved < 1e-10:
            break

    lam = -slope
    mean_log = sum(logs) / n
    ss_tot = sum((l - mean_log) ** 2 for l in logs) or _EPS
    ss_res = sum((l - (intercept + slope * t)) ** 2 for t, l in zip(ts, logs))
    sigma2 = ss_res / (n - 2) if n > 2 else 0.0
    return {
        "decay_rate_per_s": round(lam, 5),
        "half_life_s": round(math.log(2) / lam, 2) if lam > 1e-6 else None,
        # Duan smearing: E[y] = exp(intercept + σ²/2), not exp(intercept)
        "initial_engagement": round(math.exp(intercept + sigma2 / 2.0), 4),
        "initial_engagement_median": round(math.exp(intercept), 4),
        "r2": round(1 - ss_res / ss_tot, 4),
        "n": n,
        "censored_fraction": round(n_censored / n, 4),
        "detection_limit": round(lod, 6),
        "habituating": lam > 1e-4,
    }


# ─────────────────────── the full profile pipeline ───────────────────────


def iter_cognitive_profile(
    event_stream: Optional[Sequence[Sequence[float]]] = None,
    velocity_series: Optional[Sequence[Sequence[float]]] = None,
    decision_latencies_ms: Optional[Sequence[float]] = None,
    completion_ratio: Optional[float] = None,
) -> Iterator[dict]:
    """Run every model in sequence, yielding one frame per computation stage —
    this is what the SSE endpoint streams so the frontend can show the actual
    work. Final frame: {"stage": "complete", "profile": {...}}.

    event_stream: [[t_ms, symbol_id], ...]; velocity_series: [[t_ms, v], ...].
    """
    profile: dict = {"models_run": [], "models_skipped": []}

    # 1. drift-diffusion over decision latencies
    #
    # EZ-diffusion inverts three summary statistics — mean RT, RT variance and
    # ACCURACY — into (v, a, Ter). Accuracy is not optional decoration: v is a
    # monotone function of it, and the mapping is steep. On identical latencies,
    # assuming p_correct = 0.60 gives v = 0.054, 0.75 gives v = 0.140, and 0.90
    # gives v = 0.243. A fabricated accuracy therefore moves the headline
    # "processing efficiency" by a factor of ~2.6 while looking like a
    # measurement.
    #
    # Substituting a default accuracy when no completion signal exists is
    # therefore not a safe fallback: `summarise_profile` hands this straight to
    # the profiler prompt as "the strongest evidence available", so a guess
    # would be laundered into evidence. Refusing is the honest output — the
    # session lands in models_skipped with a reason attached.
    yield {"stage": "ddm", "status": "fitting", "detail": "EZ-diffusion on decision latencies"}
    ddm = None
    ddm_skip: Optional[str] = None
    if completion_ratio is None:
        ddm_skip = "no completion signal — EZ-diffusion needs an observed accuracy"
    elif not decision_latencies_ms or len(decision_latencies_ms) < EZ_MIN_DECISIONS:
        ddm_skip = (
            f"{len(decision_latencies_ms or [])} decision latencies "
            f"(< {EZ_MIN_DECISIONS} needed for a stable RT variance)"
        )
    else:
        lat_s = [x / 1000.0 for x in decision_latencies_ms]
        mu = sum(lat_s) / len(lat_s)
        var = sum((x - mu) ** 2 for x in lat_s) / max(1, len(lat_s) - 1)
        ddm = ez_diffusion(mu, var, completion_ratio, len(lat_s), latencies_s=lat_s)
        if ddm is None:
            ddm_skip = "EZ-diffusion degenerate at this accuracy (p at 0 or 1)"
    profile["ddm"] = ddm
    if ddm:
        profile["models_run"].append("ez_diffusion")
    else:
        profile["models_skipped"].append("ez_diffusion")
        profile["ddm_skipped_reason"] = ddm_skip
    yield {"stage": "ddm", "status": "done", "result": ddm, "detail": ddm_skip}

    symbols = [int(s) for _, s in (event_stream or []) if 0 <= int(s) < len(EVENT_SYMBOLS)]
    times = [float(t) for t, _ in (event_stream or [])]

    # 2. HMM latent-state decoding with BIC selection
    yield {"stage": "hmm", "status": "fitting", "detail": f"Baum-Welch EM over {len(symbols)} events, K ∈ {{2,3,4}}"}
    # The `>= 12` here is only a cheap pre-filter. The real gate is inside
    # _hmm_fit_single, which refuses any K it cannot identify from this many
    # events at this observed alphabet size — so short sessions now legitimately
    # land in models_skipped instead of returning a fabricated state decoding.
    hmm = hmm_select(symbols, len(EVENT_SYMBOLS)) if len(symbols) >= 12 else None
    if hmm:
        yield {
            "stage": "hmm",
            "status": "converged",
            "detail": (
                f"K={hmm['n_states']} wins BIC; LL {hmm['ll_trace'][0]:.1f} → "
                f"{hmm['log_likelihood']:.1f} in {hmm['em_iterations']} EM iterations"
            ),
            "ll_trace": hmm["ll_trace"],
            "bic_by_k": hmm["bic_by_k"],
        }
    profile["hmm"] = hmm
    (profile["models_run"] if hmm else profile["models_skipped"]).append("hmm")
    yield {"stage": "hmm", "status": "done", "result": hmm}

    # 3. Markov + information dynamics
    yield {"stage": "information", "status": "computing", "detail": "transition structure, entropy rate, LZ76, SampEn"}
    info = None
    if len(symbols) >= 8:
        info = markov_transitions(symbols, len(EVENT_SYMBOLS))
        info["lz76_complexity"] = lz76_complexity(symbols)
    if velocity_series and len(velocity_series) >= 12:
        se = sample_entropy([v for _, v in velocity_series][:200])
        if info is None:
            info = {}
        info["kinematic_sample_entropy"] = se
    profile["information"] = info
    (profile["models_run"] if info else profile["models_skipped"]).append("information_dynamics")
    yield {"stage": "information", "status": "done", "result": info}

    # 4. spectral profile of scroll kinematics
    # The grid is derived from the record (1/span … 1/(2·median Δt), capped at
    # the top of the burst band), so the detail line cannot quote fixed limits.
    yield {
        "stage": "spectral",
        "status": "computing",
        "detail": "Lomb-Scargle periodogram over a record-bounded frequency grid",
    }
    spec = None
    if velocity_series and len(velocity_series) >= 16:
        spec = spectral_profile([t for t, _ in velocity_series], [v for _, v in velocity_series])
    profile["spectral"] = spec
    (profile["models_run"] if spec else profile["models_skipped"]).append("spectral")
    yield {"stage": "spectral", "status": "done", "result": spec}

    # 5. foraging regime from inter-event intervals
    yield {"stage": "foraging", "status": "fitting", "detail": "power-law vs exponential MLE on inter-event intervals"}
    forage = None
    if len(times) >= 12:
        intervals = [b - a for a, b in zip(times, times[1:]) if b > a]
        forage = powerlaw_vs_exponential(intervals)
    profile["foraging"] = forage
    (profile["models_run"] if forage else profile["models_skipped"]).append("foraging")
    yield {"stage": "foraging", "status": "done", "result": forage}

    # 6. habituation from kinematic magnitude over time
    yield {"stage": "habituation", "status": "fitting", "detail": "exponential engagement-decay fit"}
    hab = None
    if velocity_series and len(velocity_series) >= 10:
        t0 = velocity_series[0][0]
        hab = habituation_fit(
            [(t - t0) / 1000.0 for t, _ in velocity_series],
            [abs(v) for _, v in velocity_series],
        )
    profile["habituation"] = hab
    (profile["models_run"] if hab else profile["models_skipped"]).append("habituation")
    yield {"stage": "habituation", "status": "done", "result": hab}

    profile["summary"] = summarise_profile(profile)
    yield {"stage": "complete", "profile": profile}


def full_cognitive_profile(**kwargs) -> dict:
    """Non-streaming convenience wrapper around iter_cognitive_profile."""
    final = None
    for frame in iter_cognitive_profile(**kwargs):
        final = frame
    return final["profile"] if final else {}


def summarise_profile(profile: dict) -> dict:
    """Compact, honest trait summary for downstream persona seeding."""
    out: dict = {"disclaimer": "behavioral model parameters, not clinical measures"}
    ddm = profile.get("ddm")
    if ddm:
        out["processing_efficiency"] = ddm["drift_rate"]
        out["decision_caution"] = ddm["boundary_separation"]
    info = profile.get("information")
    if info and "predictability" in info:
        out["behavioral_predictability"] = info["predictability"]
        if info.get("lz76_complexity") is not None:
            out["sequence_complexity"] = info["lz76_complexity"]
    hmm = profile.get("hmm")
    if hmm and hmm.get("state_path"):
        path = hmm["state_path"]
        # ACCUMULATE per label, never assign.
        #
        # _label_states names a state by its argmax emission symbol, so two
        # latent states both dominated by `scroll` are both "scanning". A dict
        # comprehension keyed on label silently overwrote the first, so the
        # occupancy vector no longer summed to 1 and `dominant_state` could name
        # a state that was not in fact dominant. Two states sharing a label are
        # behaviourally the same mode, so summing them is also the honest read.
        occupancy: dict[str, float] = {}
        for i, s in enumerate(hmm["states"]):
            occupancy[s["label"]] = occupancy.get(s["label"], 0.0) + path.count(i) / len(path)
        occupancy = {k: round(v, 3) for k, v in occupancy.items()}
        out["state_occupancy"] = occupancy
        if occupancy:
            # sort key breaks ties deterministically, so the same session never
            # reports a different dominant state between runs
            out["dominant_state"] = max(sorted(occupancy), key=lambda k: occupancy[k])
    forage = profile.get("foraging")
    if forage:
        out["exploration_regime"] = forage["regime"]
    hab = profile.get("habituation")
    if hab and hab.get("habituating"):
        out["attention_half_life_s"] = hab.get("half_life_s")
    return out
