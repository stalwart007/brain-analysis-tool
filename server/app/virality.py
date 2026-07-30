"""Virality Forecaster — Galton-Watson branching-process analysis of twin
share intents.

Each twin reports p_i = probability they'd share the content. With fan-out k
(contacts exposed per sharer), spread is a branching process with offspring
distribution Binomial(k, p). The math this gives us, exactly:

- R0 = k·p̄ — basic reproduction number; R0 > 1 is the epidemic threshold.
- Extinction probability q — smallest fixed point of the offspring PGF
  G(s) = (1-p+ps)^k (homogeneous "mixing" model) and of the twin-mixture PGF
  G(s) = (1/n)Σᵢ(1-pᵢ+pᵢs)^k (heterogeneous model; Jensen ⇒ q_hetero ≥ q_mix
  at equal mean — audience heterogeneity kills cascades).
- Expected cascade size per seed = 1/(1-R0) when subcritical.
- 2,000 seeded Monte-Carlo cascades give generation-by-generation quantile
  bands and the final-size distribution, whose tail is tested with a DISCRETE
  power-law fit (Hurwitz-zeta MLE per Clauset-Shalizi-Newman 2009) plus a
  semi-parametric bootstrap KS goodness-of-fit test — near-critical spread has
  power-law cascade sizes with exponent exactly 3/2.

Honest framing: this is a sensitivity analysis under an assumed fan-out on
synthetic twins — not a prediction of a specific social network.
"""

from __future__ import annotations

import asyncio
import bisect
import math
import random
from typing import AsyncIterator, Callable, Optional, Sequence

import openai

from .analytics import bootstrap_ci
from .config import LOAD_TO_TEMPERATURE, SWARM_CONCURRENCY, TWIN_MODEL
from .oai import Refusal, async_client, parse_completion, response_format_for
from .schemas import PersonaSeed, TwinShare, ViralityRequest
from .studies import _clamp01, _persona_system

N_SIMS = 2000
SEED = 11
MAX_GENERATIONS = 20
SIZE_CAP = 50_000  # per-cascade runaway guard for supercritical sims

# ─────────────────────────────── pure math ───────────────────────────────


def extinction_probability(ps: Sequence[float], k: int) -> float:
    """Smallest fixed point of the mixture PGF G(s)=(1/n)Σ(1-p+ps)^k.

    Solved by safeguarded Newton on f(s) = G(s) − s rather than by fixed-point
    iteration. The iteration is correct in the limit but converges at a rate of
    G'(q), and G'(q) → 1 as R0 → 1 — so near criticality it crawls, and 500
    steps is nowhere near enough. That matters because near-critical is exactly
    the regime this product is about: at R0 = 1.0008 the iterate gave a takeoff
    probability of 0.0064 against a true 0.0021, an error of 200%, while
    looking perfectly converged.

    Newton is quadratic and lands in single-digit iterations across the whole
    range. It is bracketed to [0, 1] and falls back to a bisection step if a
    Newton step would leave the interval, so it cannot diverge on the flat
    region near s = 1.
    """
    n = len(ps)
    if n == 0 or k <= 0:
        return 1.0

    def G(s: float) -> float:
        return sum((1 - p + p * s) ** k for p in ps) / n

    def dG(s: float) -> float:
        return sum(k * p * (1 - p + p * s) ** (k - 1) for p in ps) / n

    # Subcritical or critical ⇒ extinction is certain and s = 1 is the minimal
    # root. G'(1) is the mean offspring count R0.
    if dG(1.0) <= 1.0:
        return 1.0

    # Supercritical: the minimal root lies strictly in [0, 1). f(0) = G(0) > 0
    # and f(1) = 0 with f decreasing into it, so bracket just below 1.
    lo, hi = 0.0, 1.0
    s = 0.5
    for _ in range(100):
        f = G(s) - s
        if abs(f) < 1e-15:
            return s
        # maintain the bracket: f > 0 means the root is above s
        if f > 0:
            lo = s
        else:
            hi = s
        d = dG(s) - 1.0
        nxt = s - f / d if d != 0.0 else 0.5 * (lo + hi)
        if not (lo < nxt < hi):
            nxt = 0.5 * (lo + hi)
        if abs(nxt - s) < 1e-15:
            return nxt
        s = nxt
    return s


def _binom(rng: random.Random, n: int, p: float) -> int:
    """Binomial(n, p) draw. Uses the stdlib sampler when available (3.12+)."""
    if n <= 0:
        return 0
    if p <= 0.0:
        return 0
    if p >= 1.0:
        return n
    try:
        return rng.binomialvariate(n, p)  # type: ignore[attr-defined]
    except AttributeError:  # pragma: no cover — Python < 3.12
        return sum(1 for _ in range(n) if rng.random() < p)


def simulate_cascades(
    ps: Sequence[float], k: int, n_sims: int = N_SIMS, seed: int = SEED
) -> dict:
    """Seeded Galton-Watson ensemble with twin-heterogeneous share
    probabilities: every individual is a random twin.

    Returns per-generation quantiles of new/cumulative sharers, survival
    fraction, final sizes, and — critically — which runs were **censored** by
    hitting the size cap.

    Two things about the size cap have to be handled carefully, because getting
    either wrong flips the verdict:

    1. The cap check must not `break` out of the *parent* loop mid-generation.
       Doing so leaves the remaining parents in that generation unreproduced,
       which does not truncate the process, it strangles it: `active` takes an
       under-count and the cascade dies early. Every capped run then lands on a
       fake atom at exactly SIZE_CAP, which feeds the power-law fit and makes a
       genuinely explosive cascade report as "spread stays local and
       predictable" — the opposite of the truth.

    2. Capped runs must stay distinguishable from runs that finished naturally,
       or nothing downstream can tell that a size is a lower bound.

    So a run that reaches the cap completes its current generation, is flagged
    censored, and stops. `censored` is returned so the tail fit can exclude
    lower bounds instead of treating them as observations.

    Performance: births per generation are drawn by allocating the `active`
    individuals across twins multinomially and taking one Binomial(k·nᵢ, pᵢ)
    per twin, rather than k Bernoulli draws per individual. Identical
    distribution, but O(twins) instead of O(active·k) — the per-individual form
    costs ~17 s of blocking CPU on a single supercritical run, inside an async
    generator, stalling every concurrent stream.
    """
    rng = random.Random(seed)
    m = len(ps)
    active_by_gen = [[0] * n_sims for _ in range(MAX_GENERATIONS + 1)]
    cum_by_gen = [[0] * n_sims for _ in range(MAX_GENERATIONS + 1)]
    finals: list[int] = []
    censored: list[bool] = []

    capped_flags: list[bool] = []

    for sim in range(n_sims):
        active, total = 1, 1
        was_censored = False
        capped = False
        active_by_gen[0][sim] = 1
        cum_by_gen[0][sim] = 1
        for g in range(1, MAX_GENERATIONS + 1):
            # allocate this generation's parents across twins (multinomial,
            # uniform) by sequential binomial splitting, then one binomial per
            # twin for its offspring
            births = 0
            remaining = active
            for i in range(m):
                if remaining <= 0:
                    break
                share = remaining if i == m - 1 else _binom(rng, remaining, 1.0 / (m - i))
                remaining -= share
                births += _binom(rng, k * share, ps[i])

            active = births
            total += births
            if total >= SIZE_CAP:
                # complete the generation honestly, record the bound, stop
                total = SIZE_CAP
                was_censored = True
                capped = True
                active_by_gen[g][sim] = active
                cum_by_gen[g][sim] = total
                # A capped run is still ALIVE — we stopped simulating it, it did
                # not die out. Leaving active_by_gen at its 0 initialiser made
                # `alive_frac` read 0.000 for exactly the runs that exploded: at
                # R0 = 5.4 the survival curve showed the epidemic going extinct
                # at generation 8, precisely because every run blew past
                # SIZE_CAP there. Carry the live count forward instead.
                #
                # Clamped to the same bound as `total`, so the two series stay
                # coherent: the raw live count at cap time can exceed SIZE_CAP
                # (138,939 against a 50,000 cumulative in the R0 = 5.4 case),
                # and an active count above the cumulative is not a thing that
                # can happen. Both are lower bounds past this point, which is
                # what `censored_frac` and `capped_frac` are there to say.
                active = min(active, SIZE_CAP)
                active_by_gen[g][sim] = active
                for g2 in range(g + 1, MAX_GENERATIONS + 1):
                    cum_by_gen[g2][sim] = total
                    active_by_gen[g2][sim] = active
                break

            active_by_gen[g][sim] = active
            cum_by_gen[g][sim] = total
            if active == 0:
                for g2 in range(g + 1, MAX_GENERATIONS + 1):
                    cum_by_gen[g2][sim] = total
                break
        # THE OTHER CENSORING BOUND, and the one that actually bites. A run
        # still carrying live individuals at MAX_GENERATIONS has not finished —
        # its total is a lower bound exactly like a capped run's, and recording
        # it as an exact final size is the same class of error as (2) above.
        #
        # Not a rare edge case: near criticality (R0 ≈ 1.2) roughly 40% of runs
        # are still alive at generation 20. Treating those as uncensored pins
        # `censored_frac` at 0.0, so the "explosive" branch in `verdict_for`
        # can never fire, the size quantiles become silent under-estimates, and
        # the tail fit is handed truncated sizes as if they were complete
        # observations.
        if active > 0:
            was_censored = True
        finals.append(total)
        censored.append(was_censored)
        capped_flags.append(capped)

    def q(xs: list[int], frac: float) -> float:
        ys = sorted(xs)
        return float(ys[min(len(ys) - 1, int(frac * len(ys)))])

    generations = []
    for g in range(1, MAX_GENERATIONS + 1):
        act, cum = active_by_gen[g], cum_by_gen[g]
        generations.append(
            {
                "g": g,
                "active_q10": q(act, 0.10),
                "active_q50": q(act, 0.50),
                "active_q90": q(act, 0.90),
                "cum_q10": q(cum, 0.10),
                "cum_q50": q(cum, 0.50),
                "cum_q90": q(cum, 0.90),
                "alive_frac": round(sum(1 for a in act if a > 0) / n_sims, 4),
            }
        )
    return {
        "generations": generations,
        "final_sizes": finals,
        "censored": censored,
        "censored_frac": round(sum(censored) / n_sims, 4) if n_sims else 0.0,
        # The two bounds are different findings and must not be conflated.
        # `capped` means the cascade outgrew SIZE_CAP — genuine explosive
        # growth. `horizon` means it was still alive at MAX_GENERATIONS —
        # slow burn, which near criticality is the common case. Reporting
        # their union under a message that names only the cap produced the
        # sentence "37.6% of simulated cascades exceeded the 50,000-node
        # simulation bound" at R0 = 1.2, where the true count was ZERO.
        "capped_frac": round(sum(capped_flags) / n_sims, 4) if n_sims else 0.0,
        "horizon_frac": (
            round(sum(1 for c, p in zip(censored, capped_flags) if c and not p) / n_sims, 4)
            if n_sims
            else 0.0
        ),
    }


def final_size_bins(finals: Sequence[int]) -> list[list[float]]:
    """Log-spaced histogram of cascade sizes (the classic heavy-tail view)."""
    edges = [1.0]
    while edges[-1] < max(finals) + 1:
        edges.append(edges[-1] * 2)
    bins = []
    for lo, hi in zip(edges, edges[1:]):
        c = sum(1 for f in finals if lo <= f < hi)
        if c:
            bins.append([lo, hi, c])
    return bins


# ──────────────────── discrete power-law tail machinery ──────────────────
#
# Cascade sizes are positive INTEGERS with a large atom at 1 (~30% of runs die
# at the seed). The continuous Clauset fit in cognition.py is correct for its
# own domain — foraging intervals, real-valued — but fitted here it gave
# α = 1.648 on critical Galton-Watson cascades where theory gives exactly 3/2,
# and it asserted "power-law tail" with no goodness-of-fit test at all. The
# machinery below is the discrete recipe from Clauset, Shalizi & Newman 2009:
# p(x) = x^(−α)/ζ(α, xmin), MLE by 1-D maximisation, xmin by KS minimisation
# on the discrete CDF, and a semi-parametric bootstrap KS test that can REJECT
# the power law instead of assuming it.

_ALPHA_LO, _ALPHA_HI = 1.05, 6.0
_ZETA_TERMS = 20  # explicit terms before the Euler–Maclaurin tail
_GOF_SIMS = 100  # seeded bootstrap datasets for the goodness-of-fit p-value
_PL_TABLE_LEN = 4096  # cached inverse-CDF entries before the exact fallback
_MAX_DRAW = 1 << 62  # overflow guard on the sampler's far-tail bracket


def hurwitz_zeta(s: float, q: float) -> float:
    """Hurwitz zeta ζ(s, q) = Σ_{j≥0} (q+j)^(−s), stdlib-only.

    The first 20 terms are summed explicitly; the remainder is Euler–Maclaurin:
    integral term a^(1−s)/(s−1), half-term a^(−s)/2, and the B₂ and B₄
    derivative corrections, with a = q + 20. The first neglected term is
    O(s⁵·a^(−s−5))/30240, which at the hardest point this code uses
    (s = 1.05, q = 1) is ~5·10⁻¹¹ absolute on a value of 20.58.

    Verified against reference values: ζ(2,1) = π²/6 and ζ(3,1) = Apéry's
    constant to < 1e-13 relative, ζ(1.5,1) = 2.612375348685488 to < 1e-13,
    and ζ(2, 1/2) = π²/2 exactly via the (2^s−1)ζ(s) identity.
    """
    if s <= 1.0 or q <= 0.0:
        raise ValueError("hurwitz_zeta requires s > 1 and q > 0")
    total = 0.0
    for j in range(_ZETA_TERMS):
        total += (q + j) ** -s
    a = q + _ZETA_TERMS
    total += a ** (1.0 - s) / (s - 1.0)  # ∫_N^∞ (q+x)^(−s) dx
    total += 0.5 * a ** -s  # boundary half-term
    total += s * a ** (-s - 1.0) / 12.0  # −B₂/2!·f′(N)
    total -= s * (s + 1.0) * (s + 2.0) * a ** (-s - 3.0) / 720.0  # −B₄/4!·f‴(N)
    return total


def _discrete_alpha_mle(logsum: float, n: int, xmin: int, tol: float = 1e-7) -> float:
    """MLE α for p(x) = x^(−α)/ζ(α, xmin) given Σ ln x over the tail.

    The log-likelihood L(α) = −n·ln ζ(α, xmin) − α·Σ ln x is strictly concave
    (exponential family; ln ζ is the convex log-partition), so golden-section
    on [1.05, 6] finds the unique maximum without derivatives. The bounds are
    not restrictive in practice: cascade-size tails live near α = 1.5 and an
    estimate pinned at either end is itself a finding (no power-law tail).
    """

    def nll(a: float) -> float:
        return n * math.log(hurwitz_zeta(a, xmin)) + a * logsum

    invphi = (math.sqrt(5.0) - 1.0) / 2.0
    lo, hi = _ALPHA_LO, _ALPHA_HI
    c = hi - invphi * (hi - lo)
    d = lo + invphi * (hi - lo)
    fc, fd = nll(c), nll(d)
    while hi - lo > tol:
        if fc < fd:
            hi, d, fd = d, c, fc
            c = hi - invphi * (hi - lo)
            fc = nll(c)
        else:
            lo, c, fc = c, d, fd
            d = lo + invphi * (hi - lo)
            fd = nll(d)
    return 0.5 * (lo + hi)


def _discrete_alpha_se(alpha: float, n: int, xmin: int) -> Optional[float]:
    """Standard error from the observed Fisher information.

    I(α) = n·d²(ln ζ)/dα² — the −α·Σln x likelihood term is linear in α and
    drops out of the second derivative. Central difference with h = 1e-3: the
    zeta itself is good to ~1e-12 relative, so the difference-quotient noise
    (~4e-6) is far below d²ln ζ/dα² which is O(1) on this range.

    The continuous fit's closed form (α−1)/√n is NOT valid here — it assumes
    the continuous Pareto information (α−1)^(−2), which overstates the discrete
    information near α = 1.5 where the atoms at small x carry most of the mass.
    """
    h = 1e-3

    def lz(a: float) -> float:
        return math.log(hurwitz_zeta(a, xmin))

    d2 = (lz(alpha + h) - 2.0 * lz(alpha) + lz(alpha - h)) / (h * h)
    info = n * d2
    return 1.0 / math.sqrt(info) if info > 0 else None


def _discrete_ks(
    values: Sequence[int], counts: Sequence[int], n: int, alpha: float, xmin: int
) -> float:
    """KS distance between the empirical and fitted DISCRETE CDFs on the tail.

    Model CDF F(x) = 1 − ζ(α, x+1)/ζ(α, xmin) for integer x ≥ xmin. Between
    two observed values the empirical CDF is flat while F keeps rising, so the
    sup over integers is attained either at an observed value v or at the
    integer just before the next one (v_next − 1); both are checked. Evaluating
    only at observed points — the tempting shortcut, and what a continuous KS
    does — under-reads the distance on sparse integer tails.

    Both checks share ONE zeta evaluation per unique value, via the exact
    recurrence ζ(α, v+1) = ζ(α, v) − v^(−α): F(v−1) = 1 − ζ(α,v)/ζ(α,xmin) and
    F(v) = F(v−1) + v^(−α)/ζ(α,xmin). This is the hot loop of the whole test —
    it runs once per xmin candidate per bootstrap refit — and halving its zeta
    calls took the full test from 1,915 ms to 1,434 ms on the heaviest tail
    measured (n = 1,990 critical cascades reaching size 21,426).
    """
    z = hurwitz_zeta(alpha, xmin)
    d, cum, prev = 0.0, 0, xmin - 1
    for v, c in zip(values, counts):
        f_below = 1.0 - hurwitz_zeta(alpha, float(v)) / z  # F(v−1)
        if v - 1 > prev:  # flat stretch of the ECDF ending just before v
            d = max(d, abs(cum / n - f_below))
        cum += c
        d = max(d, abs(cum / n - (f_below + v**-alpha / z)))
        prev = v
    return d


def _fit_discrete_tail(xs: Sequence[int], max_candidates: int = 40) -> Optional[dict]:
    """Discrete power-law fit with xmin chosen by KS minimisation.

    Same guard thinking as the continuous fit in cognition.py: the KS scan is
    floored at a tail of max(10, 10% of the sample), because unconstrained KS
    minimisation happily walks xmin into the extreme tail of almost anything —
    a power law fits the far tail of most distributions, and a fit with no
    data left has no power to be rejected. Candidate xmins are the observed
    unique values (capped at `max_candidates`, evenly strided, so the GoF
    bootstrap's 100 refits stay inside a request-path budget).

    Returns the raw fit pieces (alpha, se, xmin, ks, tail/body splits) for the
    public wrapper and the bootstrap to share; None if the sample is too small
    or degenerate.
    """
    xs_sorted = sorted(int(x) for x in xs if x >= 1)
    n_all = len(xs_sorted)
    if n_all < 10:
        return None
    min_tail = max(10, int(0.10 * n_all))

    uniq: list[int] = []
    ucnt: list[int] = []
    for x in xs_sorted:
        if uniq and uniq[-1] == x:
            ucnt[-1] += 1
        else:
            uniq.append(x)
            ucnt.append(1)

    log_prefix = [0.0]
    for x in xs_sorted:
        log_prefix.append(log_prefix[-1] + math.log(x))

    eligible: list[int] = []
    for c in uniq:
        if n_all - bisect.bisect_left(xs_sorted, c) < min_tail:
            break  # candidates ascend, every later tail is smaller still
        eligible.append(c)
    if not eligible:
        return None
    if len(eligible) > max_candidates:
        step = (len(eligible) - 1) / (max_candidates - 1)
        eligible = sorted({eligible[round(i * step)] for i in range(max_candidates)})

    best: Optional[tuple[float, int]] = None  # (ks, xmin)
    for c in eligible:
        if xs_sorted[-1] == c:
            continue  # all tail values equal ⇒ degenerate likelihood
        i = bisect.bisect_left(xs_sorted, c)
        m = n_all - i
        logsum = log_prefix[-1] - log_prefix[i]
        alpha_c = _discrete_alpha_mle(logsum, m, c, tol=1e-4)
        j = bisect.bisect_left(uniq, c)
        ks = _discrete_ks(uniq[j:], ucnt[j:], m, alpha_c, c)
        if best is None or ks < best[0]:
            best = (ks, c)
    if best is None:
        return None

    xmin = best[1]
    i = bisect.bisect_left(xs_sorted, xmin)
    m = n_all - i
    logsum = log_prefix[-1] - log_prefix[i]
    alpha = _discrete_alpha_mle(logsum, m, xmin, tol=1e-7)  # refine at the winner
    j = bisect.bisect_left(uniq, xmin)
    ks = _discrete_ks(uniq[j:], ucnt[j:], m, alpha, xmin)
    return {
        "alpha": alpha,
        "alpha_se": _discrete_alpha_se(alpha, m, xmin),
        "xmin": xmin,
        "ks": ks,
        "tail": xs_sorted[i:],
        "body": xs_sorted[:i],
        "n": m,
        "n_total": n_all,
    }


def _powerlaw_sampler(alpha: float, xmin: int) -> Callable[[float], int]:
    """Inverse-CDF sampler for the fitted discrete power law.

    The CDF is cached as cumulative x^(−α)/ζ(α, xmin) sums for the first 4096
    support points, which covers all but ~1% of draws at α = 1.5. Beyond the
    table the draw is still EXACT: F(x) = 1 − ζ(α, x+1)/ζ(α, xmin) is
    evaluated on demand and inverted by doubling + bisection — no geometric or
    continuous approximation, so the bootstrap's synthetic tails carry the
    same far-tail mass as the fitted model.

    The doubling is capped at `_MAX_DRAW`. Without the cap, a draw with
    u > 1 − 1e-16 against a fit pinned at the α = 1.05 bound walks the bracket
    out past 1e320, where `(q+j) ** -s` raises OverflowError converting the
    int to a float. That is a ~1e-11 event per bootstrap run, which is exactly
    the kind of thing that surfaces once, in production, as a 500.
    """
    z = hurwitz_zeta(alpha, xmin)
    table: list[float] = []
    cum = 0.0
    x = xmin
    while len(table) < _PL_TABLE_LEN:
        cum += x ** -alpha / z
        table.append(cum)
        if cum >= 1.0 - 1e-9:
            break
        x += 1

    def cdf(v: int) -> float:
        return 1.0 - hurwitz_zeta(alpha, v + 1.0) / z

    def draw(u: float) -> int:
        if u <= table[-1]:
            return xmin + bisect.bisect_left(table, u)
        lo = xmin + len(table) - 1  # cdf(lo) = table[-1] < u
        hi = min(2 * lo, _MAX_DRAW)
        while hi < _MAX_DRAW and cdf(hi) < u:
            lo = hi
            hi = min(2 * hi, _MAX_DRAW)
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if cdf(mid) < u:
                lo = mid
            else:
                hi = mid
        return hi

    return draw


def _powerlaw_gof_p(fit: dict, gof_sims: int, seed: int) -> float:
    """Semi-parametric bootstrap KS goodness-of-fit p-value (Clauset §4.1).

    Each of `gof_sims` seeded synthetic datasets is built to mimic the real
    generative split: with probability n_tail/n a point is drawn from the
    FITTED discrete power law at/above xmin, otherwise it resamples the
    empirical body below xmin. Each synthetic dataset is then refit from
    scratch — xmin scan and all, so the null distribution of the KS statistic
    carries the same fitting flexibility that produced the observed one — and
    p is the fraction of synthetic KS distances ≥ the observed. Small p means
    the data are FARTHER from a power law than the power law is from itself:
    reject. A synthetic set too degenerate to refit counts toward p (it cannot
    witness a rejection).

    Refitting each synthetic — rather than scoring it against the *observed*
    α and xmin — is the whole point and the easy thing to get wrong. The
    observed KS is a minimum over the xmin scan, so comparing it to KS
    distances computed at fixed parameters would compare a minimised statistic
    against unminimised ones and reject nearly everything.

    Calibration, measured at B = 100, n = 1,000, over 100 seeded trials on data
    drawn from an exact discrete power law: 7.0% of p-values below 0.1 at
    α = 1.5 and 8.0% at α = 2.5 (nominal 10%), 4.0% and 3.0% below 0.05
    (nominal 5%), mean p 0.525 and 0.514 against 0.5 for a uniform. Slightly
    conservative at B = 100, which is the right direction for a test whose
    rejection is used to withhold a heavy-tail claim.
    """
    rng = random.Random(seed)
    body = fit["body"]
    n = fit["n_total"]
    p_tail = fit["n"] / n
    draw = _powerlaw_sampler(fit["alpha"], fit["xmin"])
    exceed = 0
    for _ in range(gof_sims):
        synth = [
            draw(rng.random())
            if (not body or rng.random() < p_tail)
            else body[rng.randrange(len(body))]
            for _ in range(n)
        ]
        refit = _fit_discrete_tail(synth)
        if refit is None or refit["ks"] >= fit["ks"]:
            exceed += 1
    return exceed / gof_sims


def cascade_tail_test(
    sizes: Sequence[int], seed: int = SEED, gof_sims: int = _GOF_SIMS
) -> Optional[dict]:
    """Discrete power-law analysis of cascade final sizes, with an honest
    goodness-of-fit verdict.

    Replaces the continuous `powerlaw_vs_exponential` at this call site. Two
    things were wrong with that, both measured on critical Galton-Watson
    cascades (R0 = 1.0 at k = 4, p = 0.25, 2,000 seeded sims), where
    branching-process theory gives the total-progeny exponent exactly 3/2:

    **The fit was on the wrong support.** Cascade sizes are integers with a
    32.7% atom at 1; the continuous Pareto MLE returned α = 1.5795 ± 0.0277
    where the discrete Hurwitz-zeta MLE on the same sample returns
    1.5462 ± 0.0203. Running the sample out to n = 20,000 to separate estimator
    bias from sampling noise, the discrete MLE at xmin = 2 gives
    α = 1.4978 ± 0.0043 — half a standard error from 3/2. On data drawn from an
    exact discrete power law it is unbiased at both ends of the range:
    1.5011 ± 0.0183 at α = 1.5 and 2.5059 ± 0.0517 at α = 2.5, n = 1,000. The
    continuous fit also spent the sample to get there — it kept 439 tail points
    against the discrete fit's 722, because a continuous law can only be made
    to fit these integers far out in the tail.

    **"Power-law tail" was asserted, never tested.** There was no
    goodness-of-fit test at all, so the only way to *not* be told a regime was
    for the fit to fail outright. The semi-parametric bootstrap KS test added
    here rejects: 72–80% of geometric and discretised-exponential samples at
    n = 1,000, and — the number that matters for a product that sells "this
    could go viral" — **zero of 150 such samples were labelled `levy_like` or
    `heavy_tailed`**. On data genuinely from the fitted model the p-value is
    calibrated-to-conservative: 7.0% and 8.0% fall below 0.1 against a nominal
    10% (α = 1.5 and 2.5, 100 trials each), with mean p 0.525 and 0.514.

    Three-part verdict, per Clauset-Shalizi-Newman:
    - fit: α (with observed-Fisher SE) and xmin (discrete-KS minimisation);
    - plausibility: `power_law_plausible` = bootstrap GoF p ≥ 0.1 — `p_value`
      IS this GoF p-value (the headline claim is "the tail is a power law");
    - alternative: Vuong LR against a geometric tail on the same xmin
      (`vuong_z`, `vuong_p_value`), so an exponentially bounded tail is named
      rather than merely "not power law".

    Regimes: `levy_like` (power law plausible, α ≤ 3), `heavy_tailed` (pure
    power law rejected but decisively heavier than geometric), `brownian_like`
    (geometric favored, or a plausible but steep α > 3 tail — finite variance
    either way), `not_power_law` (rejected and not detectably heavy — the tail
    shape is unclassified). Fully deterministic under `seed`.

    Runtime, measured, since this runs inside a request path: 477 ms at n = 886
    on production-shaped cascade sizes, 1,017 ms on the worst case the fit can
    face (n = 1,000 drawn from α = 1.5, xmin = 1 — a tail spanning six decades
    of unique values), 49 ms at α = 2.5. B = 100 bootstrap refits throughout.
    """
    fit = _fit_discrete_tail(sizes)
    if fit is None:
        return None
    alpha, xmin, tail, n = fit["alpha"], fit["xmin"], fit["tail"], fit["n"]
    n_all = fit["n_total"]

    # ── Vuong LR against a geometric tail on the same support ────────────
    mean_tail = sum(tail) / n
    if mean_tail - xmin <= 1e-12:
        return None
    theta = 1.0 / (mean_tail - xmin + 1.0)  # geometric MLE on {xmin, xmin+1, …}
    lnz = math.log(hurwitz_zeta(alpha, xmin))
    ln_theta, ln_1m = math.log(theta), math.log(1.0 - theta)
    diffs = [
        (-alpha * math.log(x) - lnz) - (ln_theta + (x - xmin) * ln_1m) for x in tail
    ]
    r_total = sum(diffs)
    mean_d = r_total / n
    var_d = sum((d - mean_d) ** 2 for d in diffs) / n
    sigma = math.sqrt(var_d)
    if sigma < 1e-12:
        vuong_z, vuong_p = 0.0, 1.0
    else:
        vuong_z = r_total / (math.sqrt(n) * sigma)
        vuong_p = math.erfc(abs(vuong_z) / math.sqrt(2.0))

    gof_p = _powerlaw_gof_p(fit, gof_sims, seed)
    plausible = gof_p >= 0.1
    favors_geometric = vuong_p <= 0.10 and r_total < 0
    beats_geometric = vuong_p <= 0.10 and r_total > 0

    if favors_geometric:
        regime = "brownian_like"
        interpretation = (
            "exponentially bounded cascade sizes — a geometric tail beats the "
            f"power law (Vuong p={vuong_p:.3f}); spread stays local and predictable"
        )
    elif plausible and alpha <= 3.0:
        regime = "levy_like"
        interpretation = (
            "heavy-tailed cascade sizes — near-critical spread, occasional huge "
            f"cascades (discrete power-law tail plausible, bootstrap KS p={gof_p:.2f})"
        )
    elif plausible:
        regime = "brownian_like"
        interpretation = (
            f"steep power-law tail (α={alpha:.2f} > 3) — finite-variance cascade "
            "sizes; spread stays local and predictable"
        )
    elif beats_geometric and alpha <= 3.0:
        # The GoF rejects the pure power law, yet the tail is decisively
        # heavier than geometric. Collapsing this into "not_power_law" throws
        # away the finding the reader needs: occasional huge cascades are real,
        # only the exact functional form is not established. This is the
        # measured verdict for genuine critical Galton-Watson sizes at
        # n = 2,000 (α = 1.5462 ± 0.0203, GoF p = 0.01, Vuong z = +11.5): the
        # exponent matches the 3/2 theory, and the rejection is the finite-size
        # curvature of the exact Borel-type law near xmin, which a sample this
        # large can see. α is descriptive here, not a fitted law.
        regime = "heavy_tailed"
        interpretation = (
            f"heavy-tailed cascade sizes (α≈{alpha:.2f}) — decisively heavier than "
            f"an exponential tail (Vuong p={vuong_p:.3f}), but a pure power law is "
            f"rejected as the exact form (bootstrap KS p={gof_p:.2f}); expect "
            "occasional huge cascades and read α as descriptive, not fitted"
        )
    else:
        # Measured on clean subcritical runs (R0 = 0.6 and 0.8, k = 4, 2,000
        # sims): GoF p = 0.00 with Vuong z = +1.21 and −0.03 — the sizes are
        # neither a pure power law nor a pure geometric, which is exactly right,
        # because subcritical Galton-Watson progeny is a power law with an
        # exponential cutoff. The old code called this `brownian_like` and
        # printed "exponentially bounded cascade sizes" — a functional form it
        # never tested. Naming the tail unclassified and pointing at the
        # simulated quantiles is the honest version of the same advice.
        regime = "not_power_law"
        interpretation = (
            f"a power-law tail is REJECTED (bootstrap KS p={gof_p:.2f}) and the "
            f"tail is not detectably heavier than exponential (Vuong p={vuong_p:.2f}) "
            "— no heavy-tail claim is supported here; read the simulated size "
            "quantiles rather than a tail exponent"
        )

    return {
        "alpha": round(alpha, 4),
        "alpha_se": (
            round(fit["alpha_se"], 4) if fit["alpha_se"] is not None else None
        ),
        "xmin": xmin,
        "ks_distance": round(fit["ks"], 4),
        "loglik_ratio_per_sample": round(mean_d, 4),
        "vuong_z": round(vuong_z, 4),
        "vuong_p_value": round(vuong_p, 4),
        # `p_value` is now the GOODNESS-OF-FIT p (was: the Vuong p in the
        # continuous fit). The headline claim downstream is "power-law tail",
        # so the headline p must be the test of that claim; the Vuong p keeps
        # its own key above.
        "p_value": round(gof_p, 4),
        "gof_p_value": round(gof_p, 4),
        "power_law_plausible": plausible,
        "gof_sims": gof_sims,
        "discrete": True,
        "regime": regime,
        "interpretation": interpretation,
        "n": n,
        "n_total": n_all,
        "method": (
            "discrete power-law MLE via Hurwitz zeta (Clauset-Shalizi-Newman "
            "2009), xmin by discrete-KS minimisation, semi-parametric bootstrap "
            "KS goodness-of-fit, Vuong LR vs geometric"
        ),
    }


def _pct_ci(sorted_reps: Sequence[float], nd: int = 4) -> Optional[list[float]]:
    """95% percentile interval from pre-sorted bootstrap replicates."""
    n = len(sorted_reps)
    if n < 20:
        return None
    lo = sorted_reps[int(0.025 * n)]
    hi = sorted_reps[min(n - 1, int(0.975 * n))]
    return [round(lo, nd), round(hi, nd)]


def _size_summary(sims: dict) -> dict:
    """Cascade-size quantiles straight from the simulation.

    These are the only size figures that survive the supercritical regime: the
    analytic mean 1/(1−R₀) genuinely diverges above criticality, and returning
    None there left the product's headline case with no size number at all.
    Quantiles are always defined, and censoring is reported alongside so a
    bounded reading is never mistaken for a converged one.
    """
    finals = sorted(sims["final_sizes"])
    n = len(finals)
    if n == 0:
        return {
            "median": None,
            "p90": None,
            "max": None,
            "censored_frac": 0.0,
            "capped_frac": 0.0,
            "horizon_frac": 0.0,
        }
    return {
        "median": finals[n // 2],
        "p90": finals[min(n - 1, int(0.90 * n))],
        "max": finals[-1],
        "censored_frac": sims["censored_frac"],
        # Split out so consumers stop saying "hit the bound" about runs that
        # merely had not finished. The old note said "lower bounds at the
        # simulation cap" for both, which is only true of the capped ones.
        "capped_frac": sims["capped_frac"],
        "horizon_frac": sims["horizon_frac"],
        "note": (
            "quantiles over simulated cascades; capped runs are lower bounds at "
            f"the {SIZE_CAP:,}-node cap, horizon runs are lower bounds at "
            f"generation {MAX_GENERATIONS}"
        ),
    }


def virality_result(ps: list[float], k: int, load: str) -> dict:
    """All branching-process outputs from the twin share intents. Pure."""
    n = len(ps)
    p_mean = sum(ps) / n
    r0 = k * p_mean
    share_ci = bootstrap_ci(ps)
    q_mix = extinction_probability([p_mean], k)
    q_het = extinction_probability(ps, k)
    # twin bootstrap on the heterogeneous extinction probability
    rng = random.Random(SEED)
    q_reps, size_reps = [], []
    for _ in range(200):
        rs = [ps[rng.randrange(n)] for _ in range(n)]
        q_reps.append(extinction_probability(rs, k))
        rb = k * sum(rs) / n
        if rb < 0.999:
            size_reps.append(1.0 / (1.0 - rb))
    q_reps.sort()
    size_reps.sort()
    sims = simulate_cascades(ps, k)

    # Fit the tail only to cascades that actually terminated. A censored run's
    # size is a lower bound, not an observation, and feeding a pile of them in
    # at exactly SIZE_CAP is what made explosive content read as "local".
    censored_frac = sims["censored_frac"]
    uncensored = [
        int(f)
        for f, was_censored in zip(sims["final_sizes"], sims["censored"])
        if f > 0 and not was_censored
    ]

    capped_frac = sims["capped_frac"]
    horizon_frac = sims["horizon_frac"]

    # "Explosive" is a claim about outgrowing the SIZE CAP, so gate it on the
    # cap alone. Gating on the union fired this branch — and printed "exceeded
    # the 50,000-node simulation bound" — at R0 = 1.2, where 36.7% of runs were
    # merely still alive at generation 20 (median size 463) and exactly ZERO
    # had reached the cap. That is a slow burn, not unbounded growth, and it is
    # the regime this product exists to analyse.
    if capped_frac >= 0.02:
        crit = {
            "regime": "explosive",
            "censored_frac": censored_frac,
            "capped_frac": capped_frac,
            "horizon_frac": horizon_frac,
            "n_uncensored": len(uncensored),
            "interpretation": (
                f"{capped_frac * 100:.1f}% of simulated cascades exceeded the "
                f"{SIZE_CAP:,}-node simulation bound — growth is unbounded at this "
                "fanout, so cascade size is limited by the audience, not by the content"
            ),
        }
    elif horizon_frac >= 0.02:
        # Still spreading when the simulation stopped: the sizes are lower
        # bounds, so a tail fit to the survivors would be selection on the
        # fast-extinguishing runs. Report the censoring instead of a shape.
        crit = {
            "regime": "unresolved",
            "censored_frac": censored_frac,
            "capped_frac": capped_frac,
            "horizon_frac": horizon_frac,
            "n_uncensored": len(uncensored),
            "interpretation": (
                f"{horizon_frac * 100:.1f}% of simulated cascades were still spreading "
                f"at generation {MAX_GENERATIONS}, so their sizes are lower bounds and "
                "the tail shape cannot be identified from this horizon — near-critical "
                "spread that neither dies out nor runs away within the simulated window"
            ),
        }
    else:
        # Fit only when the sizes are genuine observations. The tail test runs
        # a 100-refit bootstrap, so it is also the expensive path — computing
        # it first and throwing it away in the censored regimes (as the old
        # call order did with the continuous fit) would burn the request-path
        # budget on a number nobody sees.
        crit = cascade_tail_test(uncensored, seed=SEED)
        if crit:
            crit["censored_frac"] = censored_frac
            crit["n_uncensored"] = len(uncensored)
    return {
        "run_id": "",
        "twin_count": n,
        "cognitive_load": load,
        "fanout": k,
        "share_mean": round(p_mean, 4),
        "share_ci": share_ci,
        "r0": round(r0, 4),
        "r0_ci": [round(k * share_ci[0], 4), round(k * share_ci[1], 4)] if share_ci else None,
        "critical_fanout": round(1.0 / p_mean, 2) if p_mean > 1e-9 else None,
        "supercritical": r0 > 1.0,
        "extinction_q_mixing": round(q_mix, 4),
        "extinction_q_hetero": round(q_het, 4),
        # general percentile form — the old hard-coded [4] and [194] silently
        # became the wrong quantiles the moment the 200-rep loop changed
        "extinction_q_ci": _pct_ci(q_reps),
        "takeoff_probability": round(1.0 - q_het, 4),
        "expected_cascade_size": round(1.0 / (1.0 - r0), 2) if r0 < 1.0 else None,
        "cascade_size_ci": _pct_ci(size_reps, nd=2) if len(size_reps) >= 100 else None,
        # `cascade_size_ci` bootstraps 1/(1−R₀), which only exists below
        # criticality, so replicates that land supercritical are dropped — the
        # interval is therefore conditional on subcriticality and truncated
        # exactly where the upside is. Say so rather than passing it off as a
        # plain 95% CI, and always give the simulation's own quantiles, which
        # are valid in both regimes and are the only size figure available when
        # the analytic mean diverges.
        "cascade_size_ci_conditional_on_subcritical": len(size_reps) < 200,
        "simulated_size": _size_summary(sims),
        "generations": sims["generations"],
        "final_size_bins": final_size_bins(sims["final_sizes"]),
        "criticality": crit,
        "share_intents": [round(p, 4) for p in ps[:200]],
        "n_sims": N_SIMS,
        "seed": SEED,
        "method": (
            "Galton-Watson branching process · PGF fixed-point extinction · "
            "2,000 seeded cascades · twin-bootstrap 95% CI"
        ),
        "disclaimer": (
            "R0 and cascade forecasts are model-derived from synthetic-twin "
            "share intents under an assumed fan-out — a sensitivity analysis, "
            "not a prediction of a specific social network"
        ),
    }


# ───────────────────────── twin orchestration ────────────────────────────

_SHARE_PREAMBLE = """You are one synthetic audience member deciding whether to
share a piece of content with people you know. Report would_share — YOUR
probability (0..1) of actually sharing/forwarding/discussing it — and the one
real reason. Sharing is costly (your reputation rides on it): most content is
NOT shared, so calibrate honestly to your persona."""


async def _share_twin(
    persona: PersonaSeed, content: str, load: str, semaphore: asyncio.Semaphore
) -> Optional[TwinShare]:
    async with semaphore:
        try:
            completion = await async_client().chat.completions.create(
                model=TWIN_MODEL,
                messages=[
                    {"role": "system", "content": _SHARE_PREAMBLE},
                    _persona_system(persona, load),
                    {"role": "user", "content": f"The content:\n{content}"},
                ],
                temperature=LOAD_TO_TEMPERATURE[load],
                response_format=response_format_for(TwinShare),
            )
            return parse_completion(completion, TwinShare)
        except (openai.OpenAIError, Refusal, RuntimeError, ValueError):
            return None


async def stream_virality(
    personas: list[PersonaSeed], request: ViralityRequest
) -> AsyncIterator[dict]:
    roster = personas * request.twins_per_persona
    k = request.fanout
    yield {"type": "start", "total": len(roster), "fanout": k}
    semaphore = asyncio.Semaphore(SWARM_CONCURRENCY)
    tasks = [
        asyncio.create_task(_share_twin(p, request.content, request.cognitive_load, semaphore))
        for p in roster
    ]
    ps: list[float] = []
    for fut in asyncio.as_completed(tasks):
        resp = await fut
        if resp is None:
            yield {"type": "agent_failed"}
            continue
        share = _clamp01(resp.would_share)
        ps.append(share)
        yield {"type": "agent", "share": share, "reason": resp.reason[:120]}
    if len(ps) < 3:
        yield {"type": "error", "detail": "Too few twins responded to model spread."}
        return

    yield {"type": "stage", "stage": "branching", "detail": "solving PGF fixed point for extinction probability"}
    yield {"type": "stage", "stage": "simulating", "detail": "2,000 seeded Galton-Watson cascades"}
    result = virality_result(ps, k, request.cognitive_load)
    result["content_preview"] = request.content[:140]
    for gen in result["generations"]:
        yield {"type": "generation", **gen}
    yield {"type": "done", "result": result}
