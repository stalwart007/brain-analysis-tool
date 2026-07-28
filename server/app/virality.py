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
  bands and the final-size distribution (checked for heavy tails with the
  power-law machinery — near-critical spread has power-law cascade sizes).

Honest framing: this is a sensitivity analysis under an assumed fan-out on
synthetic twins — not a prediction of a specific social network.
"""

from __future__ import annotations

import asyncio
import random
from typing import AsyncIterator, Optional, Sequence

import openai

from .analytics import bootstrap_ci
from .cognition import powerlaw_vs_exponential
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
        float(f)
        for f, was_censored in zip(sims["final_sizes"], sims["censored"])
        if f > 0 and not was_censored
    ]
    crit = powerlaw_vs_exponential(uncensored)

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
    elif crit:  # re-domain the shared tail-test's interpretation for cascades
        crit["censored_frac"] = censored_frac
        crit["n_uncensored"] = len(uncensored)
        crit["interpretation"] = (
            "heavy-tailed cascade sizes — near-critical spread, occasional huge cascades"
            if crit["regime"] == "levy_like"
            else "exponentially bounded cascade sizes — spread stays local and predictable"
        )
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
