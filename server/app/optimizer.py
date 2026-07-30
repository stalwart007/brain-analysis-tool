"""Copy Optimizer — an evolutionary search over copy, scored by the twin swarm.

Every other study in this product SCORES copy that a human wrote. This one
WRITES it: a population of variants is evaluated against the swarm, the winners
are bred into the next generation by an LLM "mutator", and the loop repeats.
The pieces, and what each is actually doing:

- **Population / generations.** Generation 0 is the submitted seed plus P−1
  first-order mutants of it. Each later generation is bred from the previous
  generation's measured elites. Variants are never carried forward as duplicate
  arms; the champion is selected across the whole search history at the end
  (hall-of-fame elitism in the *reporting*), which is why arms have unequal
  trial counts and why the verdict is posterior-based rather than mean-based.

- **Thompson allocation inside a generation.** Splitting a generation's budget
  evenly across P variants spends most of it confirming that bad copy is bad.
  Instead each generation opens with a forced-exploration floor (every variant
  gets `min_evals_per_variant`, so no variant is judged on a single draw) and
  then allocates the remainder in waves by Thompson sampling over the arms'
  Beta posteriors — budget flows to the variants that look like winners while
  weak ones keep a nonzero probe rate. Same policy as the adaptive A/B in
  swarm.stream_compare, applied within a generation rather than across a test.

- **Breeding.** The mutator is the reasoning model, not the twin model, and it
  is given the parents' texts, their measured intent/act rates, and the phrases
  twins said landed hardest. It returns crossovers (fuse the winning mechanism
  of two parents) and mutations (change exactly one thing about one parent).
  Sampling temperature is the mutation rate.

- **Honest convergence.** An evolutionary loop always produces a "winner" —
  the maximum of a noisy sample is biased upward by construction, and with P×G
  variants the winner's curse is severe. So the population is ranked by an
  empirical-Bayes SHRUNK mean intent (a lucky 2-trial arm no longer outranks a
  solid 40-trial seed), the reported lift is measured select-then-evaluate on
  split halves (the readings that score the champion had no say in choosing
  it), and a win is only claimed when the champion's 95% Beta credible
  interval on the act rate clears the seed's AND the selection-adjusted lift's
  bootstrap interval excludes zero. The naive in-sample margin is still
  reported, labelled as such, so the size of the correction is visible — same
  treatment as sequence.py's cross-fitted ordering gain, and for the same
  statistical sin: the expensive mistake here is shipping a rewrite that was
  noise. Measured across 300 null studies and 300 with a real +0.100 arm:
  the reported lift went from +0.0870 (99.3% positive) to −0.0003 under the
  null while still recovering +0.0820 of a true +0.100, and the "improved"
  verdict fires on 0.3% of nulls with power rising from 11.7% to 15.7%.

The twin never sees the brief. An audience does not know what the advertiser
was trying to achieve, and telling them primes exactly the response the brief
asks for — that would make the whole optimisation self-fulfilling.

`optimizer_result` is pure: twin outputs + lineage in, result dict out, no
network. Everything statistical lives there and is unit-tested.
"""

from __future__ import annotations

import asyncio
import itertools
import random
from typing import AsyncIterator, Optional, Sequence

import openai

from .streaming import as_they_land
from .analytics import bayesian_ab, bootstrap_ci, thompson_allocate
from .config import (
    LOAD_TO_TEMPERATURE,
    PROFILER_MODEL,
    SWARM_CONCURRENCY,
    TWIN_MODEL,
)
from .oai import Refusal, async_client, parse_completion, response_format_for
from .schemas import (
    CopyMutant,
    CopyMutantBatch,
    CopyOptimizerRequest,
    PersonaSeed,
    TwinCopyScore,
)
from .studies import _clamp01, _persona_system

# Bootstrap seed for the per-variant intent intervals. Distinct from
# analytics.bootstrap_ci's default (7) for the reason documented in swarm.py:
# intervals drawn from the identical resample index sequence move in lockstep.
_INTENT_CI_SEED = 4649
# Posterior draws for the final ranked population. Matches the density
# swarm.stream_compare uses for its end-of-run posterior — stable p_best at the
# ~40 arms a 8×5 search can produce.
_FINAL_DRAWS = 8000
# Thompson seed. Fixed for reproducibility; the value itself carries no meaning.
_THOMPSON_SEED = 4243
# Posterior draws for the pairwise champion-vs-seed act-rate comparison.
# bayesian_ab keeps its Monte-Carlo draws internal, so P(champion > seed) is
# recomputed here from the same Beta(1+acts, 1+misses) posteriors, seeded.
_PAIRWISE_DRAWS = 8000
_PAIRWISE_SEED = 6367
# Selection-adjusted intent lift: repeated split-halves for the point estimate
# and within-arm bootstrap replicates — each re-running the whole
# select-then-evaluate procedure — for its interval. Split and replicate
# budgets mirror sequence.py's cross-fit (_SPLIT_REPS = 12, _LIFT_REPS = 200).
_ADJUST_SPLIT_REPS = 12
_ADJUST_BOOT_REPS = 200
_ADJUST_SEED = 9203
# Elites shown to the breeder. Two is the minimum for a crossover to be a
# crossover; past three the breeder starts averaging everything into mush.
_ELITE_COUNT = 3
# Sampling temperature for the breeder. This is the mutation rate: at the
# twin-scoring temperatures (0.45–1.15) the breeder regresses to the mean and
# the population converges on generation 1.
_BREEDER_TEMPERATURE = 0.95


# ─────────────────────────────── pure math ───────────────────────────────


def _intervals_overlap(a: Sequence[float], b: Sequence[float]) -> bool:
    """Two closed intervals overlap iff neither lies strictly above the other."""
    return a[0] <= b[1] and b[0] <= a[1]


def _shrunk_means(rows_by_arm: dict[str, list[float]]) -> tuple[dict[str, float], dict]:
    """Empirical-Bayes shrinkage of per-arm mean intents toward the grand mean.

    THE RANKING BUG THIS REPLACES. The population was sorted on the raw
    per-arm mean, so a challenger seen twice outranked a seed seen forty
    times whenever its two draws happened to land high — and with adaptive
    allocation most challengers ARE thinly sampled, so the top of the ranking
    was systematically the luckiest small arm, not the best copy. Measured on
    300 null studies (all nine arms at the same true intent, seed 40 trials,
    challengers 2–20 — the shape the Thompson allocation actually produces):
    the raw-mean ranking put a challenger above the seed in **98.7%** of runs
    and handed the top slot to an arm with ≤3 trials in **45.0%** (mean 7.0
    trials for the winner). After shrinkage: **93.3%** and **31.3%** (mean
    10.3). 93.3% is close to the 88.9% a fair 1-of-9 ranking would give, so
    what is left is mostly the arithmetic of having eight challengers, not
    bias toward thin ones.

    The fix is the standard one-way random-effects estimator: each arm's mean
    is pulled toward the trial-weighted grand mean with weight n_i/(n_i + τ̂),
    where τ̂ = σ̂²_within / τ̂²_between is the prior pseudo-count implied by a
    method-of-moments variance decomposition across arms (MSB/MSW with the
    unequal-n n₀ correction). Forty trials barely move; two trials are pulled
    most of the way home. Measured example (it is the regression test): a
    2-trial arm at 0.90 raw against a 40-trial seed at 0.85 raw, in a
    population whose other mutants are worse — grand mean 0.8068, within-arm
    variance 0.0294, between-arm variance 0.0009, so τ̂ = 31.98 prior trials.
    The seed goes 0.8500 → 0.8308 and the challenger 0.9000 → 0.8123: the
    seed ranks FIRST, where the raw mean ranked it second. Across 400 random
    mutant populations of that shape (realistic twin noise, sd 0.18–0.30) the
    shrunk ranking puts the seed above the lucky 2-trial arm in 86.8% of them.
    The raw-mean ranking puts the challenger first in all 400 — not a
    measurement but an identity, since 0.90 > 0.85 whatever the rest of the
    population does, which is the whole complaint about it.

    Edges, each deliberate:
      · one arm, or σ̂²_within measured 0 → no shrinkage (the means are exact
        as observed);
      · τ̂²_between ≤ 0 → the spread of arm means is no larger than sampling
        noise alone predicts, so full pooling: every arm gets the grand mean
        and the ranking falls entirely to the tie-breakers (p_best, trials);
      · every arm a single reading → within-arm noise is inestimable, and
        calling it zero would rank pure luck; full pooling again.
    """
    arms = {k: v for k, v in rows_by_arm.items() if v}
    raw = {k: sum(v) / len(v) for k, v in arms.items()}
    info: dict = {
        "grand_mean": None,
        "within_var": None,
        "between_var": None,
        "prior_strength": None,
        "pooled": False,
    }
    if not arms:
        return {}, info
    n_tot = sum(len(v) for v in arms.values())
    grand = sum(x for v in arms.values() for x in v) / n_tot
    info["grand_mean"] = round(grand, 4)
    if len(arms) == 1:
        info["prior_strength"] = 0.0
        return raw, info
    dfw = n_tot - len(arms)
    if dfw <= 0:
        info["pooled"] = True
        return {k: grand for k in arms}, info
    s2w = sum((x - raw[k]) ** 2 for k, v in arms.items() for x in v) / dfw
    info["within_var"] = round(s2w, 6)
    k_arms = len(arms)
    msb = sum(len(v) * (raw[k] - grand) ** 2 for k, v in arms.items()) / (k_arms - 1)
    n0 = (n_tot - sum(len(v) ** 2 for v in arms.values()) / n_tot) / (k_arms - 1)
    tau2_b = (msb - s2w) / n0 if n0 > 0 else 0.0
    info["between_var"] = round(max(tau2_b, 0.0), 6)
    if s2w <= 1e-12:
        info["prior_strength"] = 0.0
        return raw, info
    if tau2_b <= 1e-12:
        info["pooled"] = True
        return {k: grand for k in arms}, info
    prior = s2w / tau2_b
    info["prior_strength"] = round(prior, 3)
    return {
        k: (len(v) * raw[k] + prior * grand) / (len(v) + prior)
        for k, v in arms.items()
    }, info


def _p_beats(
    challenger: tuple[int, int],
    baseline: tuple[int, int],
    draws: int = _PAIRWISE_DRAWS,
    seed: int = _PAIRWISE_SEED,
) -> float:
    """P(challenger's true act rate > baseline's) under Beta(1+s, 1+f) posteriors.

    The same posteriors bayesian_ab builds; recomputed here because that
    function returns summaries, not draws. Seeded, so deterministic. This is a
    statement about the two arms' data — the multiplicity of having searched
    many arms is handled by the selection-adjusted lift, not here.
    """
    rng = random.Random(seed)
    ca, cn = challenger
    ba, bn = baseline
    wins = 0
    for _ in range(draws):
        c = rng.betavariate(1.0 + ca, 1.0 + (cn - ca))
        b = rng.betavariate(1.0 + ba, 1.0 + (bn - ba))
        if c > b:
            wins += 1
    return wins / draws


def _split_lift(
    rows_by_arm: dict[str, list[float]], seed_id: str, rng: random.Random
) -> Optional[float]:
    """One select-then-evaluate split of the intent readings.

    Each arm's readings are shuffled and cut in half. The champion challenger
    is chosen on one half (argmax of the shrunk means — the same rule the
    shipping ranking uses) and its margin over the seed is measured on the
    other half, which had no say in the choice. Averaged over both role
    assignments. Candidates must have at least one reading in BOTH halves —
    membership is decided by row counts, never by row values, so nothing
    leaks. Returns None when the split leaves nothing to select or score.
    """
    halves: dict[str, tuple[list[float], list[float]]] = {}
    for k, v in rows_by_arm.items():
        if not v:
            continue
        idx = list(range(len(v)))
        rng.shuffle(idx)
        cut = len(v) // 2
        halves[k] = ([v[i] for i in idx[:cut]], [v[i] for i in idx[cut:]])

    def one(select: int, evaluate: int) -> Optional[float]:
        sel = {k: h[select] for k, h in halves.items() if h[select]}
        ev = {k: h[evaluate] for k, h in halves.items() if h[evaluate]}
        if seed_id not in ev:
            return None
        candidates = [k for k in sel if k != seed_id and k in ev]
        if not candidates:
            return None
        shrunk, _ = _shrunk_means(sel)
        pick = max(candidates, key=lambda k: (shrunk[k], len(sel[k]), k))
        return (sum(ev[pick]) / len(ev[pick])) - (sum(ev[seed_id]) / len(ev[seed_id]))

    vals = [g for g in (one(0, 1), one(1, 0)) if g is not None]
    return sum(vals) / len(vals) if vals else None


def _selection_adjusted_lift(
    rows_by_arm: dict[str, list[float]], seed_id: str, seed: int = _ADJUST_SEED
) -> tuple[Optional[float], Optional[tuple[float, float]]]:
    """The champion's intent margin over the seed, with the winner's curse out.

    Sibling of sequence.py's cross-fitted `objective_gain`, and for the same
    reason: the champion is the argmax over the whole searched population
    scored on the full sample, so its in-sample margin is biased upward by
    construction — measured **+0.0870 under a true null, positive in 99.3%**
    of the runs that reported it (300 null studies, nine equal arms). Here:

      · POINT ESTIMATE — repeated split-halves. Selection runs on one half of
        each arm's readings, evaluation on the other, averaged over
        _ADJUST_SPLIT_REPS random splits (both role assignments each). The
        half that scores the winner had no say in picking it. Measured on the
        same null: **−0.0003** when reported (−0.0046 unconditionally) against
        the naive +0.0870 — the bias is gone, not merely shrunk. On a real
        effect (one 20-trial arm at +0.100) it recovers **+0.0820**, the
        attenuation being the price of selecting on half the readings.

      · INTERVAL — a within-arm bootstrap that re-runs the WHOLE
        select-then-evaluate procedure inside every replicate. Resampling
        around a fixed winner would reproduce the original error in a new
        place (see the sequence.py docstring); re-running the argmax puts the
        selection noise into the interval where it belongs.

    Returns (None, None) when no split leaves both a selectable challenger
    and a scoreable seed — e.g. every challenger has a single reading. The
    caller must treat that as "cannot support a claim", not as zero.
    """
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(_ADJUST_SPLIT_REPS):
        g = _split_lift(rows_by_arm, seed_id, rng)
        if g is not None:
            estimates.append(g)
    if not estimates:
        return None, None
    point = sum(estimates) / len(estimates)

    boot_rng = random.Random(seed + 1)
    replicates: list[float] = []
    for _ in range(_ADJUST_BOOT_REPS):
        resample = {
            k: [v[boot_rng.randrange(len(v))] for _ in range(len(v))]
            for k, v in rows_by_arm.items()
            if v
        }
        reps = [
            g
            for g in (
                _split_lift(resample, seed_id, boot_rng)
                for _ in range(_ADJUST_SPLIT_REPS)
            )
            if g is not None
        ]
        if reps:
            replicates.append(sum(reps) / len(reps))
    if len(replicates) < 20:
        return round(point, 4), None
    replicates.sort()
    lo = replicates[max(0, int(0.025 * len(replicates)))]
    hi = replicates[min(len(replicates) - 1, int(0.975 * len(replicates)) - 1)]
    return round(point, 4), (round(lo, 4), round(hi, 4))


def optimizer_result(
    lineage: list[dict],
    scores: dict[str, list[TwinCopyScore]],
    seed_id: str,
    brief: str,
    cognitive_load: str,
    generations_planned: int,
    failed_evaluations: int = 0,
) -> dict:
    """Rank the searched population and say — honestly — whether it beat the seed.

    `lineage` is one record per variant ever created:
        {id, generation, text, parents: [id...], operation, rationale}
    `scores` maps variant id → the twin outputs it actually received. A variant
    can legitimately have ZERO scores (Thompson starved it, or every twin call
    on it failed); those are reported with trials=0 and null fitness rather
    than 0.0, because an unmeasured variant is not a bad variant and must not
    rank above one.

    Two fitness metrics are carried the whole way, deliberately:
      · mean intent — continuous, sensitive; ranked on its SHRUNK posterior
        mean, lifted via split-half select-then-evaluate
      · act rate — binary, the decision metric, Beta-Binomial posterior
    A win must clear both. Optimisation searches for the maximum of a noisy
    objective, so the winner is upward-biased by construction. Before this
    was corrected, `intent_lift_vs_seed` measured +0.0870 under a true null
    and was positive in 99.3% of the runs that reported it — and it was
    reported as a headline even when convergence said "not_supported". Now the
    gate and the headline run on the selection-adjusted quantities, and the
    naive margin is carried alongside them, labelled, so the size of the
    correction stays visible.
    """
    if not any(scores.get(v["id"]) for v in lineage):
        raise RuntimeError(
            "Copy optimisation produced no valid twin scores (all twins failed)."
        )

    stats: dict[str, dict] = {}
    intent_rows: dict[str, list[float]] = {}
    for v in lineage:
        vid = v["id"]
        rows = scores.get(vid, [])
        intents = [_clamp01(r.intent) for r in rows]
        if intents:
            intent_rows[vid] = intents
        trials = len(rows)
        acts = sum(1 for r in rows if r.would_act)
        # bootstrap_ci returns None for samples it deems too small to resample
        # meaningfully (analytics owns that gate); keep the None rather than
        # inventing a degenerate interval. Nothing decisional hangs on this
        # interval any more — the gate runs on the selection-adjusted lift.
        ci = bootstrap_ci(intents, seed=_INTENT_CI_SEED) if trials >= 2 else None
        stats[vid] = {
            "id": vid,
            "generation": v["generation"],
            "text": v["text"],
            "preview": v["text"][:200],
            "parents": list(v.get("parents") or []),
            "operation": v.get("operation", "mutation"),
            "rationale": v.get("rationale", ""),
            "trials": trials,
            "acts": acts,
            "act_rate": round(acts / trials, 4) if trials else None,
            "mean_intent": round(sum(intents) / trials, 4) if trials else None,
            "intent_ci": list(ci) if ci else None,
            "phrases": [r.strongest_phrase.strip() for r in rows[:5] if r.strongest_phrase.strip()],
        }

    evaluated = [s for s in stats.values() if s["trials"] > 0]

    # Joint Beta-Binomial posterior over every variant that was actually
    # evaluated. Unevaluated arms are excluded: Beta(1,1) is uniform, so
    # including them would hand real p_best mass to copy nobody ever read.
    bayes = bayesian_ab(
        {s["id"]: (s["acts"], s["trials"]) for s in evaluated}, draws=_FINAL_DRAWS
    )
    posterior_by_id = {a["name"]: a for a in bayes["arms"]}
    for s in evaluated:
        s["posterior"] = posterior_by_id.get(s["id"])

    # ── per-generation fitness trace ────────────────────────────────────
    max_gen = max(s["generation"] for s in stats.values())
    generations: list[dict] = []
    for g in range(max_gen + 1):
        members = [s for s in evaluated if s["generation"] == g]
        born = [s for s in stats.values() if s["generation"] == g]
        if not members:
            generations.append(
                {
                    "index": g,
                    "population": len(born),
                    "evaluated": 0,
                    "evaluations": 0,
                    "best_fitness": None,
                    "mean_fitness": None,
                    "best_variant_id": None,
                    "best_preview": None,
                }
            )
            continue
        fits = [s["mean_intent"] for s in members]
        best = max(members, key=lambda s: s["mean_intent"])
        generations.append(
            {
                "index": g,
                "population": len(born),
                "evaluated": len(members),
                "evaluations": sum(s["trials"] for s in members),
                "best_fitness": round(max(fits), 4),
                "mean_fitness": round(sum(fits) / len(fits), 4),
                "best_variant_id": best["id"],
                "best_preview": best["preview"],
            }
        )

    # ── ranking + convergence verdict ───────────────────────────────────
    # Rank on the empirical-Bayes SHRUNK mean intent (see _shrunk_means), not
    # the raw mean: raw-mean sorting put the luckiest thinly-sampled arm on
    # top by construction. Tie-breakers unchanged — posterior p_best, then
    # trials — so under full pooling (a null-looking population) the ranking
    # degrades to the evidence-weighted order rather than to dict order. The
    # raw mean stays in every row; only the sort key changed.
    shrunk, shrink_info = _shrunk_means(intent_rows)
    for s in evaluated:
        s["shrunk_intent"] = round(shrunk[s["id"]], 4)
    ranking = sorted(
        evaluated,
        key=lambda s: (
            shrunk[s["id"]],
            (s.get("posterior") or {}).get("p_best", 0.0),
            s["trials"],
        ),
        reverse=True,
    )
    best = ranking[0]
    seed_stat = stats.get(seed_id)

    beat_seed = False
    posterior_separated: Optional[bool] = None
    intent_separated: Optional[bool] = None
    intent_lift: Optional[float] = None
    intent_lift_pct: Optional[float] = None
    adjusted_lift: Optional[float] = None
    adjusted_ci: Optional[tuple[float, float]] = None
    adjusted_lift_pct: Optional[float] = None
    p_beats_seed: Optional[float] = None

    if seed_stat is None or not seed_stat["trials"]:
        convergence = "no_baseline"
        verdict = (
            "The seed variant was never successfully scored, so there is no "
            "baseline to beat. Every number below is descriptive only."
        )
    elif best["id"] == seed_id:
        convergence = "seed_wins"
        verdict = (
            "The seed is still the top-ranked variant — the search produced "
            "nothing measurably better. Keep the copy you have."
        )
    else:
        # The naive in-sample margin. Kept — as sequence.py keeps
        # naive_objective_gain — so the size of the correction is visible,
        # but it is never the gate and never the headline.
        intent_lift = round(best["mean_intent"] - seed_stat["mean_intent"], 4)
        intent_lift_pct = (
            round(intent_lift / seed_stat["mean_intent"] * 100, 1)
            if seed_stat["mean_intent"] > 0
            else None
        )
        adjusted_lift, adjusted_ci = _selection_adjusted_lift(intent_rows, seed_id)
        adjusted_lift_pct = (
            round(adjusted_lift / seed_stat["mean_intent"] * 100, 1)
            if adjusted_lift is not None and seed_stat["mean_intent"] > 0
            else None
        )
        p_beats_seed = round(
            _p_beats(
                (best["acts"], best["trials"]),
                (seed_stat["acts"], seed_stat["trials"]),
            ),
            4,
        )
        pb, ps = best.get("posterior"), seed_stat.get("posterior")
        posterior_separated = (
            not _intervals_overlap((pb["ci_low"], pb["ci_high"]), (ps["ci_low"], ps["ci_high"]))
            if pb and ps
            else False
        )
        # Descriptive only since the selection-adjusted lift took over the
        # gate: overlap of two naive intent intervals neither accounts for
        # selection nor for the arms' shared grand mean.
        intent_separated = (
            not _intervals_overlap(best["intent_ci"], seed_stat["intent_ci"])
            if best["intent_ci"] and seed_stat["intent_ci"]
            else False
        )
        # The gate runs on the honest quantities and nothing else: the
        # act-rate posteriors must separate AND the selection-adjusted intent
        # lift must be positive with an interval that excludes zero. A thin
        # sample that cannot be split (adjusted_lift None) cannot win.
        beat_seed = bool(
            posterior_separated
            and adjusted_lift is not None
            and adjusted_lift > 0
            and adjusted_ci is not None
            and adjusted_ci[0] > 0
        )
        if beat_seed:
            convergence = "improved"
            verdict = (
                f"{best['id']} beats the seed: selection-adjusted intent lift "
                f"{adjusted_lift:+.3f} (95% CI {adjusted_ci[0]:+.3f} to "
                f"{adjusted_ci[1]:+.3f}), P(champion act rate beats seed) = "
                f"{p_beats_seed:.2f}, with non-overlapping 95% act-rate "
                f"posteriors. The naive in-sample margin was {intent_lift:+.3f}; "
                "the gap between the two is the selection bias the search added."
            )
        else:
            convergence = "not_supported"
            if adjusted_lift is None:
                reason = (
                    "the challenger arms are too thinly sampled to separate "
                    "selection from evaluation, so no honest lift exists"
                )
            elif adjusted_lift <= 0:
                reason = (
                    f"once selection is accounted for the lift is "
                    f"{adjusted_lift:+.3f} — the in-sample margin of "
                    f"{intent_lift:+.3f} was selection bias, not copy"
                )
            elif not posterior_separated:
                reason = (
                    f"the act-rate posteriors still overlap (P(champion beats "
                    f"seed) = {p_beats_seed:.2f})"
                )
            else:
                reason = (
                    f"the selection-adjusted lift of {adjusted_lift:+.3f} has "
                    "an interval that still includes zero"
                )
            verdict = (
                f"{best['id']} ranks first, but the optimisation did NOT beat the "
                f"starting point: {reason}. The maximum of a noisy search is "
                "biased upward — treat this as a candidate to A/B test, not a result."
            )

    edges = [
        {"parent": p, "child": s["id"], "operation": s["operation"]}
        for s in stats.values()
        for p in s["parents"]
        if p in stats
    ]

    total_evals = sum(s["trials"] for s in stats.values())
    return {
        "run_id": "",
        # Survivor count, as everywhere else in this codebase: failed twin calls
        # are excluded, not counted as zeros.
        "twin_count": total_evals,
        "failed_evaluations": failed_evaluations,
        "cognitive_load": cognitive_load,
        "brief_preview": brief[:200],
        "seed_variant_id": seed_id,
        "seed_preview": (seed_stat or {}).get("preview", ""),
        "generations_planned": generations_planned,
        "generations": generations,
        "population": ranking,
        "unevaluated": [s["id"] for s in stats.values() if s["trials"] == 0],
        "best_variant_id": best["id"],
        "best_text": best["text"],
        "beat_seed": beat_seed,
        "convergence": convergence,
        "verdict": verdict,
        # The naive in-sample margin, under both its historical key and its
        # honest name. Selected-then-measured on the same sample, so it is
        # upward-biased by construction; kept only so the size of the
        # correction is visible. Never render it as the lift.
        "intent_lift_vs_seed": intent_lift,
        "intent_lift_vs_seed_pct": intent_lift_pct,
        "naive_intent_lift": intent_lift,
        # The headline lift: select-then-evaluate on split halves, with a
        # bootstrap that re-runs the selection inside every replicate.
        "selection_adjusted_lift": adjusted_lift,
        "selection_adjusted_lift_ci": list(adjusted_ci) if adjusted_ci else None,
        "selection_adjusted_lift_pct": adjusted_lift_pct,
        "selection_bias": (
            round(intent_lift - adjusted_lift, 4)
            if intent_lift is not None and adjusted_lift is not None
            else None
        ),
        # Pairwise P(champion act rate > seed act rate) from the Beta
        # posteriors — the act-rate side of the honest comparison.
        "p_best_beats_seed": p_beats_seed,
        "shrinkage": shrink_info,
        "lift_inference": (
            "ranked by empirical-Bayes shrunk mean intent (weight n/(n+τ̂), τ̂ "
            "from a method-of-moments variance decomposition across arms); the "
            "honest lift is select-then-evaluate — the champion challenger is "
            "re-chosen on one half of each arm's readings and scored on the "
            "other, averaged over repeated splits, with a within-arm bootstrap "
            "that re-runs the whole procedure per replicate. naive_intent_lift "
            "is the in-sample margin, kept only so the correction is visible."
        ),
        "posterior_intervals_separated": posterior_separated,
        "intent_intervals_separated": intent_separated,
        "bayesian": bayes,
        "lineage_edges": edges,
        "method": (
            "evolutionary search · Thompson-sampled budget within each generation · "
            "empirical-Bayes shrunk ranking · Beta-Binomial posteriors over act rate · "
            "selection-adjusted lift via split-half select-then-evaluate bootstrap"
        ),
        "disclaimer": (
            "Fitness is synthetic-twin intent, not observed behaviour. A variant "
            "that wins here is a hypothesis worth testing on real traffic. The "
            "search selects the maximum of a noisy objective; the "
            "selection-adjusted lift removes that bias from the reported margin, "
            "and the naive in-sample margin is kept alongside it so the size of "
            "the correction is visible."
        ),
    }


# ───────────────────────── twin orchestration ────────────────────────────

_SCORE_PREAMBLE = """You are one synthetic member of a target audience reading
ONE piece of marketing copy. You do not know who wrote it or what they were
trying to achieve — react to what is actually on the page. Report `intent`:
YOUR probability (0..1) of taking the action the copy asks for. `would_act` is
true only if you would genuinely do it right now, which means intent above 0.5.
Quote the phrase that landed hardest on you (good or bad) in strongest_phrase.
Most copy does not move you; say so when it doesn't."""


async def _score_twin(
    persona: PersonaSeed,
    text: str,
    load: str,
    semaphore: asyncio.Semaphore,
) -> Optional[TwinCopyScore]:
    async with semaphore:
        try:
            completion = await async_client().chat.completions.create(
                model=TWIN_MODEL,
                messages=[
                    {"role": "system", "content": _SCORE_PREAMBLE},
                    _persona_system(persona, load),
                    {"role": "user", "content": f"The copy:\n{text}"},
                ],
                temperature=LOAD_TO_TEMPERATURE[load],
                response_format=response_format_for(TwinCopyScore),
            )
            return parse_completion(completion, TwinCopyScore)
        except (openai.OpenAIError, Refusal, RuntimeError, ValueError):
            # degrade to a smaller sample rather than failing the generation
            return None


_MUTATOR_SYSTEM = """You are a direct-response copy chief running an
evolutionary optimisation. You get a creative brief and the current population
of variants with their MEASURED audience scores, their lineage, and the phrases
the audience said landed hardest. Breed the next generation.

Rules:
- Produce exactly the number of variants requested, no more.
- At least one must be a CROSSOVER when two or more parents are available: take
  the mechanism that is working in one parent and fuse it with the mechanism
  working in another. List both parent ids.
- The others are MUTATIONS of a single high-scoring parent: change exactly ONE
  thing — the hook, the specificity, the ask, the length, or the emotional
  register — and say in one line which one you changed. List one parent id.
- Stay inside the brief's constraints. Never invent facts, numbers, claims,
  guarantees, or endorsements that are not already in the brief or a parent.
- Never repeat a parent verbatim, and keep the population genuinely different
  from each other. A converged population stops searching."""


def _parent_block(parents: list[dict]) -> str:
    lines = []
    for p in parents:
        fitness = (
            f"mean intent {p['mean_intent']:.3f}, act rate {p['act_rate']:.2f} over {p['trials']} readers"
            if p.get("trials")
            else "not yet scored"
        )
        phrases = "; ".join(p.get("phrases") or []) or "—"
        lines.append(
            f"[{p['id']}] ({fitness})\nCOPY: {p['text']}\nPHRASES THAT LANDED: {phrases}"
        )
    return "\n\n".join(lines)


async def _breed(
    brief: str, parents: list[dict], n_children: int, generation: int
) -> list[CopyMutant]:
    """One breeder call. Returns [] on any failure — the caller then keeps the
    generation it already has rather than losing the whole run to a bad JSON."""
    if n_children <= 0 or not parents:
        return []
    user = (
        f"CREATIVE BRIEF:\n{brief}\n\n"
        f"CURRENT POPULATION (generation {generation}):\n{_parent_block(parents)}\n\n"
        f"Breed generation {generation + 1}: exactly {n_children} variants."
    )
    try:
        completion = await async_client().chat.completions.create(
            model=PROFILER_MODEL,  # breeding is a reasoning task, not a twin task
            messages=[
                {"role": "system", "content": _MUTATOR_SYSTEM},
                {"role": "user", "content": user},
            ],
            temperature=_BREEDER_TEMPERATURE,
            response_format=response_format_for(CopyMutantBatch),
        )
        batch = parse_completion(completion, CopyMutantBatch)
    except (openai.OpenAIError, Refusal, RuntimeError, ValueError):
        return []
    return [c for c in batch.variants if c.text.strip()][:n_children]


def _adopt(
    child: CopyMutant,
    generation: int,
    index: int,
    valid_ids: set[str],
    fallback_parent: str,
) -> dict:
    """Turn a breeder's child into a lineage record, sanitised.

    The breeder can hallucinate a parent id, and it can label a child a
    crossover while naming one parent. Both are silently corrected here: the
    lineage is the audit trail of what the search actually did, so it must not
    record a crossover that never crossed anything.
    """
    parents = [p for p in child.parent_ids if p in valid_ids] or [fallback_parent]
    operation = "crossover" if (child.operation == "crossover" and len(parents) >= 2) else "mutation"
    return {
        "id": f"g{generation}v{index}",
        "generation": generation,
        "text": child.text.strip(),
        "parents": parents,
        "operation": operation,
        "rationale": child.rationale.strip(),
    }


async def stream_copy_optimizer(
    personas: list[PersonaSeed], request: CopyOptimizerRequest
) -> AsyncIterator[dict]:
    """The evolutionary loop as SSE.

    Frames: start · generation · wave · agent · agent_failed · variant_scored ·
    mutation · done. `wave` carries the Thompson allocation so the budget can
    be watched moving toward the leaders — that reallocation is the point of
    the bandit and is invisible otherwise.
    """
    semaphore = asyncio.Semaphore(SWARM_CONCURRENCY)
    persona_cycle = itertools.cycle(personas)
    budget = len(personas) * request.twins_per_persona

    lineage: list[dict] = []
    scores: dict[str, list[TwinCopyScore]] = {}
    failed = 0

    seed_id = "g0v0"
    seed_record = {
        "id": seed_id,
        "generation": 0,
        "text": request.seed_variant,
        "parents": [],
        "operation": "seed",
        "rationale": "the copy you have today — the baseline every mutant must beat",
    }
    lineage.append(seed_record)
    scores[seed_id] = []

    yield {
        "type": "start",
        "total": budget * request.generations,
        "budget_per_generation": budget,
        "generations": request.generations,
        "population_size": request.population_size,
        "seed_variant_id": seed_id,
        "seed_preview": request.seed_variant[:200],
        "brief_preview": request.brief[:200],
    }

    def stat(vid: str) -> dict:
        rows = scores.get(vid, [])
        trials = len(rows)
        intents = [_clamp01(r.intent) for r in rows]
        return {
            "trials": trials,
            "acts": sum(1 for r in rows if r.would_act),
            "mean_intent": round(sum(intents) / trials, 4) if trials else None,
            "act_rate": round(sum(1 for r in rows if r.would_act) / trials, 4) if trials else None,
        }

    async def tagged(vid: str, text: str):
        return vid, await _score_twin(next(persona_cycle), text, request.cognitive_load, semaphore)

    # ── generation 0: the seed plus P-1 first-order mutants of it ───────
    population = [seed_record]
    pending_children = await _breed(
        request.brief,
        [{"id": seed_id, "text": request.seed_variant, "trials": 0, "phrases": []}],
        request.population_size - 1,
        generation=0,
    )
    seen_texts = {request.seed_variant.strip().lower()}
    for child in pending_children:
        # Exact duplicates would split the budget across identical arms and
        # dilute both posteriors, so they never enter the population.
        if child.text.strip().lower() in seen_texts:
            continue
        record = _adopt(child, 0, len(population), {seed_id}, seed_id)
        seen_texts.add(record["text"].lower())
        lineage.append(record)
        scores[record["id"]] = []
        population.append(record)
        yield {
            "type": "mutation",
            "generation": 0,
            "child": record["id"],
            "parents": record["parents"],
            "operation": record["operation"],
            "rationale": record["rationale"],
            "preview": record["text"][:200],
        }

    for g in range(request.generations):
        yield {
            "type": "generation",
            "index": g,
            "budget": budget,
            "population": [
                {
                    "id": v["id"],
                    "operation": v["operation"],
                    "parents": v["parents"],
                    "preview": v["text"][:200],
                }
                for v in population
            ],
        }

        # Forced-exploration floor first: pure Thompson from a cold start can
        # ride one lucky draw and never probe a variant again.
        remaining = budget
        floor_alloc: dict[str, int] = {}
        for v in population:
            take = min(request.min_evals_per_variant, remaining)
            if take > 0:
                floor_alloc[v["id"]] = take
                remaining -= take

        gen_start_trials = {v["id"]: len(scores[v["id"]]) for v in population}

        async def run_batch(allocation: dict[str, int]):
            nonlocal failed
            texts = {v["id"]: v["text"] for v in population}
            batch = as_they_land(
                tagged(vid, texts[vid])
                for vid, count in allocation.items()
                for _ in range(count)
            )
            async for vid, score in batch:
                if score is None:
                    failed += 1
                    yield {"type": "agent_failed", "generation": g, "variant": vid}
                    continue
                scores[vid].append(score)
                yield {
                    "type": "agent",
                    "generation": g,
                    "variant": vid,
                    "intent": round(_clamp01(score.intent), 3),
                    "acts": score.would_act,
                    "phrase": score.strongest_phrase.strip()[:120],
                }

        if floor_alloc:
            yield {"type": "wave", "generation": g, "index": 0, "allocation": floor_alloc, "policy": "exploration floor"}
            async for evt in run_batch(floor_alloc):
                yield evt

        wave_idx = 1
        while remaining > 0 and population:
            # Wave size mirrors the concurrency limit: allocations are only
            # worth recomputing once the previous wave's results are in.
            size = min(remaining, max(len(population), SWARM_CONCURRENCY))
            arms = {
                v["id"]: (
                    sum(1 for r in scores[v["id"]][gen_start_trials[v["id"]]:] if r.would_act),
                    len(scores[v["id"]]) - gen_start_trials[v["id"]],
                )
                for v in population
            }
            allocation = {k: c for k, c in thompson_allocate(
                arms, size, seed=_THOMPSON_SEED + g * 97 + wave_idx
            ).items() if c > 0}
            remaining -= size
            yield {"type": "wave", "generation": g, "index": wave_idx, "allocation": allocation, "policy": "thompson"}
            async for evt in run_batch(allocation):
                yield evt
            wave_idx += 1

        for v in population:
            yield {"type": "variant_scored", "generation": g, "variant": v["id"], **stat(v["id"])}

        if g == request.generations - 1:
            break

        # ── breed the next generation from this one's measured elites ───
        ranked = sorted(
            (v for v in population if scores[v["id"]]),
            key=lambda v: stat(v["id"])["mean_intent"],
            reverse=True,
        )
        if not ranked:
            # Every twin in this generation failed. Breeding from unmeasured
            # copy is guesswork, so keep the population and try again.
            yield {
                "type": "mutation",
                "generation": g,
                "status": "skipped",
                "detail": "no variant in this generation was successfully scored",
            }
            continue

        elites = [
            {
                "id": v["id"],
                "text": v["text"],
                "phrases": [
                    r.strongest_phrase.strip()
                    for r in scores[v["id"]][:5]
                    if r.strongest_phrase.strip()
                ],
                **stat(v["id"]),
            }
            for v in ranked[:_ELITE_COUNT]
        ]
        children = await _breed(request.brief, elites, request.population_size, generation=g)
        if not children:
            yield {
                "type": "mutation",
                "generation": g,
                "status": "failed",
                "detail": "breeder call failed; carrying the current population into the next generation",
            }
            continue

        valid_ids = {v["id"] for v in population}
        next_population: list[dict] = []
        for child in children:
            if child.text.strip().lower() in seen_texts:
                continue
            record = _adopt(child, g + 1, len(next_population), valid_ids, elites[0]["id"])
            seen_texts.add(record["text"].lower())
            lineage.append(record)
            scores[record["id"]] = []
            next_population.append(record)
            yield {
                "type": "mutation",
                "generation": g + 1,
                "child": record["id"],
                "parents": record["parents"],
                "operation": record["operation"],
                "rationale": record["rationale"],
                "preview": record["text"][:200],
            }
        if next_population:
            population = next_population

    result = optimizer_result(
        lineage,
        scores,
        seed_id,
        request.brief,
        request.cognitive_load,
        request.generations,
        failed_evaluations=failed,
    )
    yield {"type": "done", "result": result}
