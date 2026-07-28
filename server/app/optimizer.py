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
  variants the winner's curse is severe. So a win is only claimed when the best
  variant's 95% Beta credible interval on the act rate AND its 95% bootstrap
  interval on mean intent BOTH clear the seed's. Non-overlapping 95% intervals
  is a *stricter* test than a 5% two-sample test, which is the point: the
  expensive mistake here is shipping a rewrite that was noise.

The twin never sees the brief. An audience does not know what the advertiser
was trying to achieve, and telling them primes exactly the response the brief
asks for — that would make the whole optimisation self-fulfilling.

`optimizer_result` is pure: twin outputs + lineage in, result dict out, no
network. Everything statistical lives there and is unit-tested.
"""

from __future__ import annotations

import asyncio
import itertools
from typing import AsyncIterator, Optional, Sequence

import openai

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
      · mean intent — continuous, sensitive, bootstrap CI
      · act rate — binary, the decision metric, Beta-Binomial posterior
    A win must clear both. Optimisation searches for the maximum of a noisy
    objective, so the winner is upward-biased by construction; requiring two
    independent intervals to separate is what stops that bias becoming a claim.
    """
    if not any(scores.get(v["id"]) for v in lineage):
        raise RuntimeError(
            "Copy optimisation produced no valid twin scores (all twins failed)."
        )

    stats: dict[str, dict] = {}
    for v in lineage:
        vid = v["id"]
        rows = scores.get(vid, [])
        intents = [_clamp01(r.intent) for r in rows]
        trials = len(rows)
        acts = sum(1 for r in rows if r.would_act)
        # bootstrap_ci returns None below n=2; keep that as None rather than
        # inventing a degenerate interval around a single observation.
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
    # Rank on mean intent, break ties on posterior p_best then on trials, so a
    # variant measured 20 times never sits below one measured twice at the same
    # mean purely by dict order.
    ranking = sorted(
        evaluated,
        key=lambda s: (
            s["mean_intent"],
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
        intent_lift = round(best["mean_intent"] - seed_stat["mean_intent"], 4)
        intent_lift_pct = (
            round(intent_lift / seed_stat["mean_intent"] * 100, 1)
            if seed_stat["mean_intent"] > 0
            else None
        )
        pb, ps = best.get("posterior"), seed_stat.get("posterior")
        posterior_separated = (
            not _intervals_overlap((pb["ci_low"], pb["ci_high"]), (ps["ci_low"], ps["ci_high"]))
            if pb and ps
            else False
        )
        intent_separated = (
            not _intervals_overlap(best["intent_ci"], seed_stat["intent_ci"])
            if best["intent_ci"] and seed_stat["intent_ci"]
            else False
        )
        beat_seed = bool(posterior_separated and intent_separated and intent_lift > 0)
        if beat_seed:
            convergence = "improved"
            verdict = (
                f"{best['id']} beats the seed: +{intent_lift:.3f} mean intent "
                f"({intent_lift_pct:+.1f}%) with non-overlapping 95% intervals on "
                "both the act rate and mean intent."
            )
        else:
            convergence = "not_supported"
            reason = (
                "the credible intervals still overlap"
                if intent_lift > 0
                else "the ranked leader does not even out-score the seed on the mean"
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
        "intent_lift_vs_seed": intent_lift,
        "intent_lift_vs_seed_pct": intent_lift_pct,
        "posterior_intervals_separated": posterior_separated,
        "intent_intervals_separated": intent_separated,
        "bayesian": bayes,
        "lineage_edges": edges,
        "method": (
            "evolutionary search · Thompson-sampled budget within each generation · "
            "Beta-Binomial posteriors over act rate · percentile bootstrap on mean intent"
        ),
        "disclaimer": (
            "Fitness is synthetic-twin intent, not observed behaviour. A variant "
            "that wins here is a hypothesis worth testing on real traffic; the "
            "search selects the maximum of a noisy objective, so its measured "
            "advantage is optimistically biased even when the intervals separate."
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
            tasks = [
                asyncio.create_task(tagged(vid, texts[vid]))
                for vid, count in allocation.items()
                for _ in range(count)
            ]
            for coro in asyncio.as_completed(tasks):
                vid, score = await coro
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
