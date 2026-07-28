"""Orchestrator: fan a scenario out across N synthetic twins and aggregate (OpenAI backend).

Architecture (Phase 1 mini-swarm; the Phase 3 Temporal-backed version keeps
these exact request shapes):
- Twins never talk to each other — pure fan-out/fan-in, no state collisions.
- Concurrency is bounded by a semaphore; failures degrade to a smaller sample
  instead of failing the run.
- OpenAI auto-caches the identical simulation preamble + scenario prefix across
  twins; only the per-twin persona/state differs.
- Cognitive state ("3 AM user") is modeled with three levers:
    1. persona-state conditioning injected into the prompt,
    2. sampling temperature raised under load (LOAD_TO_TEMPERATURE),
    3. a context throttle (rolling memory window) that binds in multi-step
       scenarios; recorded per run so Phase 2 walkthroughs inherit it.
"""

import asyncio
import itertools
import json
from collections import Counter
from typing import Callable, Literal, Optional

import openai

from .analytics import (
    bayesian_ab,
    bimodality_coefficient,
    bootstrap_ci,
    consensus_label,
    kaplan_meier,
    kmeans_auto,
    shannon_entropy_normalised,
    thompson_allocate,
)
from .config import LOAD_TO_TEMPERATURE, SWARM_CONCURRENCY, TWIN_MODEL
from .oai import Refusal, async_client, parse_completion, response_format_for
from .schemas import (
    CompareResult,
    PersonaSeed,
    ScenarioVariant,
    SwarmAggregate,
    TwinReaction,
    TwinStepReaction,
    VariantRank,
    WalkAggregate,
    WalkStepStat,
)

# Frozen — identical across all runs (outermost cache prefix).
SIMULATION_PREAMBLE = """You are one synthetic persona inside a market-research
simulation swarm. A stimulus follows (an ad, content, or a described product
flow). React exactly as your persona would — including boredom, impatience,
misreading, and abandoning. Do not be helpful or agreeable; be human. Your
reaction is aggregated with thousands of others to evaluate the stimulus, not
you."""

_LOAD_CONDITIONING = {
    "low": "You are alert and unhurried.",
    "medium": (
        "You are mildly tired and multitasking. Your patience for dense or "
        "confusing content is reduced; you skim more than you read."
    ),
    "high": (
        "It is very late. You are exhausted and distracted, running on impulse "
        "and habit. Long text feels like a wall; friction feels personal; you "
        "make snap judgments and rarely revisit them."
    ),
}

# Context throttle (Challenge 3B): rolling memory window for multi-step
# scenarios. Single-turn runs record it as run metadata; the Phase 2 multi-step
# walker truncates its history to this many steps.
_LOAD_TO_CONTEXT_STEPS = {"low": 20, "medium": 8, "high": 2}


async def _run_twin(
    persona: PersonaSeed,
    scenario: str,
    cognitive_load: Literal["low", "medium", "high"],
    semaphore: asyncio.Semaphore,
) -> TwinReaction | None:
    messages = [
        {"role": "system", "content": SIMULATION_PREAMBLE},
        {"role": "system", "content": f"STIMULUS UNDER TEST:\n{scenario}"},
        {
            "role": "system",
            "content": persona.system_prompt
            + "\n\nCurrent state: "
            + _LOAD_CONDITIONING[cognitive_load],
        },
        {
            "role": "user",
            "content": (
                "Experience the stimulus as this persona and report your "
                "reaction. Write inner_monologue first, in character, then "
                "decide the rest."
            ),
        },
    ]
    async with semaphore:
        try:
            completion = await async_client().chat.completions.create(
                model=TWIN_MODEL,
                messages=messages,
                temperature=LOAD_TO_TEMPERATURE[cognitive_load],
                response_format=response_format_for(TwinReaction),
            )
            return parse_completion(completion, TwinReaction)
        except (openai.OpenAIError, Refusal, RuntimeError, ValueError):
            # degrade to a smaller sample rather than failing the whole run
            return None


async def run_swarm(
    personas: list[PersonaSeed],
    scenario: str,
    twins_per_persona: int,
    cognitive_load: Literal["low", "medium", "high"],
) -> SwarmAggregate:
    semaphore = asyncio.Semaphore(SWARM_CONCURRENCY)
    tasks = [
        _run_twin(p, scenario, cognitive_load, semaphore)
        for p in personas
        for _ in range(twins_per_persona)
    ]
    results = await asyncio.gather(*tasks)
    reactions = [r for r in results if r is not None]
    return aggregate_reactions(reactions, scenario, cognitive_load)


async def stream_swarm(
    personas: list[PersonaSeed],
    scenario: str,
    twins_per_persona: int,
    cognitive_load: Literal["low", "medium", "high"],
):
    """Async generator yielding one event per twin as it completes, then a final
    'done' with the aggregate. This is what makes agents spawn *live* in the UI
    rather than all at once at the end."""
    semaphore = asyncio.Semaphore(SWARM_CONCURRENCY)
    total = len(personas) * twins_per_persona
    yield {"type": "start", "total": total}

    tasks = [
        asyncio.create_task(_run_twin(p, scenario, cognitive_load, semaphore))
        for p in personas
        for _ in range(twins_per_persona)
    ]
    reactions: list[TwinReaction] = []
    for coro in asyncio.as_completed(tasks):
        r = await coro
        if r is None:
            yield {"type": "agent_failed"}
            continue
        reactions.append(r)
        yield {
            "type": "agent",
            "action": r.action,
            "engagement": round(r.engagement, 3),
            "intent": round(r.intent_score, 3),
        }

    aggregate = aggregate_reactions(reactions, scenario, cognitive_load)
    yield {"type": "done", "result": aggregate.model_dump()}


# The closed action vocabulary, mirroring TwinReaction.action's Literal. Kept
# here as an explicit tuple because entropy normalisation needs the size of the
# *possible* outcome set, which a Counter over observed reactions cannot supply.
ACTION_VOCABULARY: tuple[str, ...] = ("convert", "continue", "hesitate", "abandon")


def aggregate_reactions(
    reactions: list[TwinReaction], scenario: str, cognitive_load: str
) -> SwarmAggregate:
    """Pure aggregation over twin reactions — shared by the live fan-out and the
    Batch API runner, and unit-testable without any LLM call."""
    if not reactions:
        raise RuntimeError("Swarm run produced no valid reactions (all twins failed).")

    actions = Counter(r.action for r in reactions)
    dropoffs = Counter(
        r.dropoff_point.strip()
        for r in reactions
        if r.dropoff_point and r.dropoff_point.strip().lower() not in {"null", "none", "n/a", "na", ""}
    )
    frictions = Counter(r.friction_notes.strip() for r in reactions)

    # Statistical layer: a bare mean cannot distinguish "everyone is lukewarm"
    # from "half love it, half hate it" — these do.
    intents = [r.intent_score for r in reactions]
    engagements = [r.engagement for r in reactions]
    # Normalise entropy against the full action vocabulary, not the actions
    # that happened to occur. A Counter only holds observed keys, so passing
    # its values scored "half convert / half abandon" identically to "evenly
    # spread across all four" — both 1.0. The first is a decisive split, the
    # second is no signal at all.
    entropy = shannon_entropy_normalised(
        [actions.get(a, 0) for a in ACTION_VOCABULARY], k=len(ACTION_VOCABULARY)
    )
    bc = bimodality_coefficient(intents)
    intent_ci = bootstrap_ci(intents)
    # Distinct seeds: bootstrap_ci defaults to seed=7, so both intervals were
    # drawn from the identical resample index sequence and moved in lockstep.
    engagement_ci = bootstrap_ci(engagements, seed=8191)

    # Segment discovery: k-means++ over the (intent, engagement) plane, with k
    # chosen by silhouette and a practical-spread gate. None ⇒ genuinely
    # homogeneous audience, which is itself a finding.
    segments = None
    clustering = kmeans_auto([(r.intent_score, r.engagement) for r in reactions])
    if clustering is not None:
        labels, k, sil = clustering
        segments = []
        for c in sorted(set(labels)):
            members = [reactions[i] for i in range(len(reactions)) if labels[i] == c]
            seg_intent = sum(m.intent_score for m in members) / len(members)
            seg_eng = sum(m.engagement for m in members) / len(members)
            seg_actions = Counter(m.action for m in members)
            label = (
                "enthusiasts"
                if seg_intent >= 0.66
                else "skeptics" if seg_intent <= 0.33 else "on the fence"
            )
            segments.append(
                {
                    "label": label,
                    "size": len(members),
                    "share": round(len(members) / len(reactions), 3),
                    "mean_intent": round(seg_intent, 3),
                    "mean_engagement": round(seg_eng, 3),
                    "dominant_action": seg_actions.most_common(1)[0][0],
                }
            )
        segments.sort(key=lambda s: s["size"], reverse=True)
        segments = {"k": k, "silhouette": round(sil, 3), "groups": segments}

    return SwarmAggregate(
        run_id="",  # assigned by storage on insert
        scenario_preview=scenario[:200],
        twin_count=len(reactions),
        cognitive_load=cognitive_load,
        mean_engagement=round(sum(engagements) / len(reactions), 3),
        mean_intent=round(sum(intents) / len(reactions), 3),
        action_distribution=dict(actions),
        top_dropoff_points=[d for d, _ in dropoffs.most_common(5)],
        top_frictions=[f for f, _ in frictions.most_common(5)],
        reactions=[r.model_dump() for r in reactions],
        intent_ci=list(intent_ci) if intent_ci else None,
        engagement_ci=list(engagement_ci) if engagement_ci else None,
        polarization=round(entropy, 4),
        bimodality=round(bc, 4) if bc is not None else None,
        consensus=consensus_label(entropy, bc),
        segments=segments,
    )


# ------------------------------------------------------------------ A/B compare


def _rank_variants(by_name: dict[str, SwarmAggregate], control: str) -> list[VariantRank]:
    """Rank arms by mean intent, with lift measured against `control`."""
    control_intent = by_name[control].mean_intent
    return sorted(
        (
            VariantRank(
                name=name,
                mean_intent=agg.mean_intent,
                mean_engagement=agg.mean_engagement,
                lift_vs_control_pct=(
                    None
                    if name == control or control_intent == 0
                    else round((agg.mean_intent - control_intent) / control_intent * 100, 1)
                ),
            )
            for name, agg in by_name.items()
        ),
        key=lambda r: r.mean_intent,
        reverse=True,
    )


def _annotate_compare(
    bayes: dict,
    ranking: list[VariantRank],
    control: str,
    intended_control: Optional[str] = None,
) -> dict:
    """Attach the experiment's integrity flags to the Bayesian block.

    Two failure modes were previously invisible in the payload:

    **`winner` and `bayesian.winner` optimise different things.** The top-level
    `winner` is argmax mean_intent — a mean of continuous 0-1 self-reported
    purchase intent. `bayesian.winner` is argmax P(best) on the Beta-Binomial
    posterior over `convert` COUNTS. These are different estimands over
    different data, and they routinely disagree: an arm can win on average
    warmth while another converts more of its twins, which is the classic
    "liked it more / bought it less" split. Both numbers were emitted, side by
    side, with nothing saying they answer different questions and nothing
    flagging when they point at different arms — so whichever one the reader's
    eye landed on became "the winner". Both are now named for the metric they
    optimise and the disagreement is an explicit boolean.

    **The control can be silently re-designated.** `_compare_result` builds its
    arm map from variants that produced at least one usable twin, then takes the
    first surviving key as the control. If every twin in the first variant
    failed, the second variant becomes the baseline and every
    `lift_vs_control_pct` in the payload is measured against a different
    stimulus than the caller asked for — with nothing in the response saying so.
    The lifts stay (they are correct *relative to the surviving baseline*), but
    the substitution is now stated.

    Both live on the `bayesian` dict because CompareResult's field set is fixed
    in schemas.py, which this module does not own. See the module notes.
    """
    by_intent = ranking[0].name if ranking else None
    by_conversion = bayes.get("winner")
    disagree = bool(by_intent and by_conversion and by_intent != by_conversion)
    substituted = intended_control is not None and intended_control != control

    bayes = dict(bayes)
    bayes["winner_by_mean_intent"] = by_intent
    bayes["winner_by_conversion"] = by_conversion
    bayes["winners_disagree"] = disagree
    bayes["winner_disagreement_note"] = (
        (
            f"'{by_intent}' wins on mean intent while '{by_conversion}' wins on "
            "conversion probability — these optimise different metrics over "
            "different data. Pick the one that matches your decision; do not "
            "read either as 'the' winner."
        )
        if disagree
        else None
    )
    bayes["control_arm"] = control
    bayes["control_substituted"] = substituted
    bayes["control_substitution_note"] = (
        (
            f"Every twin in '{intended_control}' failed, so '{control}' is the "
            "baseline all lift_vs_control_pct values are measured against — not "
            f"'{intended_control}' as requested."
        )
        if substituted
        else None
    )
    return bayes


async def run_compare(
    personas: list[PersonaSeed],
    variants: list[ScenarioVariant],
    twins_per_persona: int,
    cognitive_load: Literal["low", "medium", "high"],
) -> CompareResult:
    """Zero-shot A/B: every variant faces the same persona set at the same
    cognitive state, so the only difference between arms is the stimulus."""
    aggregates = await asyncio.gather(
        *(
            run_swarm(personas, v.scenario, twins_per_persona, cognitive_load)
            for v in variants
        )
    )
    by_name = {v.name: agg for v, agg in zip(variants, aggregates)}

    control = variants[0].name
    ranking = _rank_variants(by_name, control)

    # Bayesian verdict over conversion counts. A naive "B beats A by 71%" on a
    # handful of twins is frequently noise; the posterior tells us whether the
    # difference survives the uncertainty, and what shipping the leader risks.
    arms = {
        name: (agg.action_distribution.get("convert", 0), agg.twin_count)
        for name, agg in by_name.items()
    }
    bayes = _annotate_compare(bayesian_ab(arms), ranking, control, control)

    return CompareResult(
        run_id="",
        cognitive_load=cognitive_load,
        twin_count_per_variant=twins_per_persona * len(personas),
        # The schema pins `winner` to one string, so it stays the mean-intent
        # leader for backward compatibility. `bayesian.winner_by_*` and
        # `bayesian.winners_disagree` carry the honest, unreduced answer.
        winner=ranking[0].name,
        ranking=ranking,
        variants=by_name,
        bayesian=bayes,
    )


def _compare_result(
    per_variant: dict[str, list[TwinReaction]],
    scenarios: dict[str, str],
    cognitive_load: str,
    twin_count_per_variant: int,
) -> CompareResult:
    """Assemble the final CompareResult from accumulated reactions."""
    by_name = {
        name: aggregate_reactions(rs, scenarios[name], cognitive_load)
        for name, rs in per_variant.items()
        if rs
    }
    if not by_name:
        raise RuntimeError("Compare produced no valid reactions in any variant.")

    # `per_variant` is seeded in declaration order, so its first key is the arm
    # the caller nominated as the control. `by_name` drops arms whose twins all
    # failed, so its first key can be a *different* arm — and every
    # lift_vs_control_pct below would then be measured against a baseline
    # nobody asked for. Keep both and let _annotate_compare say which happened.
    intended_control = next(iter(per_variant))
    control = next(iter(by_name))
    ranking = _rank_variants(by_name, control)
    bayes = _annotate_compare(
        bayesian_ab(
            {
                name: (agg.action_distribution.get("convert", 0), agg.twin_count)
                for name, agg in by_name.items()
            }
        ),
        ranking,
        control,
        intended_control,
    )
    return CompareResult(
        run_id="",
        cognitive_load=cognitive_load,
        twin_count_per_variant=twin_count_per_variant,
        winner=ranking[0].name,
        ranking=ranking,
        variants=by_name,
        bayesian=bayes,
    )


def _per_arm_persona_cycles(
    names: list[str], personas: list[PersonaSeed]
) -> dict[str, "itertools.cycle[PersonaSeed]"]:
    """One INDEPENDENT round-robin persona cycle per arm.

    run_compare's docstring promises "every variant faces the same persona set
    at the same cognitive state, so the only difference between arms is the
    stimulus". A single `itertools.cycle(personas)` shared across arms breaks
    that promise the moment allocation stops being equal, which is the entire
    point of adaptive mode: agents are dispatched from one global rotation, so
    which persona an arm gets depends on how many agents every *other* arm
    happened to consume first. With 3 personas and Thompson sending 9 twins to
    the leader and 3 to the laggard, the two arms see different persona mixes —
    and the measured "lift" is then part stimulus effect and part persona
    composition, with no way to separate them after the fact. That is a
    confound in an experiment whose only claim is that there isn't one.

    Per-arm cycles make each arm draw personas in the same order from the same
    list, so any arm's first c draws are the same maximally-balanced prefix
    (⌈c/P⌉ or ⌊c/P⌋ of each persona) regardless of what the other arms did.
    """
    return {n: itertools.cycle(personas) for n in names}


async def stream_compare(
    personas: list[PersonaSeed],
    variants: list[ScenarioVariant],
    twins_per_persona: int,
    cognitive_load: Literal["low", "medium", "high"],
    adaptive: bool = False,
    min_per_arm: int = 8,
    seed: int = 13,
):
    """Streaming A/B with optional **batched Thompson-sampling allocation**.

    Fixed mode: every variant gets the same budget, agents stream as they land.

    Adaptive mode implements a sequential Bayesian experiment:
      1. Wave 0 seeds every arm equally (so no arm starts unexplored).
      2. Each later wave draws θ ~ Beta posterior per arm and sends each agent
         to the argmax (Thompson sampling) — budget flows toward the likely
         winner while weak arms keep a nonzero probe rate.
      3. After every wave the joint posterior is re-computed; when the leader's
         expected loss drops inside the ROPE (and every arm has ≥ min_per_arm
         samples), the run STOPS EARLY — a principled optimal-stopping rule
         that spends no tokens confirming what is already decided.

    Emits: start, agent (per twin, tagged with its variant), posterior (after
    each wave / periodically in fixed mode), early_stop (adaptive only), done.
    """
    semaphore = asyncio.Semaphore(SWARM_CONCURRENCY)
    scenarios = {v.name: v.scenario for v in variants}
    names = [v.name for v in variants]
    per_variant: dict[str, list[TwinReaction]] = {n: [] for n in names}
    failed = 0
    total_budget = twins_per_persona * len(personas) * len(names)
    persona_cycles = _per_arm_persona_cycles(names, personas)

    yield {
        "type": "start",
        "total": total_budget,
        "variants": names,
        "adaptive": adaptive,
    }

    def arms_now() -> dict[str, tuple[int, int]]:
        return {
            n: (
                sum(1 for r in per_variant[n] if r.action == "convert"),
                len(per_variant[n]),
            )
            for n in names
        }

    async def tagged(vname: str):
        r = await _run_twin(
            next(persona_cycles[vname]), scenarios[vname], cognitive_load, semaphore
        )
        return vname, r

    async def run_batch(allocation: dict[str, int]):
        nonlocal failed
        tasks = [
            asyncio.create_task(tagged(vname))
            for vname, count in allocation.items()
            for _ in range(count)
        ]
        for coro in asyncio.as_completed(tasks):
            vname, r = await coro
            if r is None:
                failed += 1
                yield {"type": "agent_failed", "variant": vname}
                continue
            per_variant[vname].append(r)
            yield {
                "type": "agent",
                "variant": vname,
                "action": r.action,
                "engagement": round(r.engagement, 3),
                "intent": round(r.intent_score, 3),
            }

    if not adaptive:
        allocation = {n: twins_per_persona * len(personas) for n in names}
        emitted = 0
        posterior_every = max(2, total_budget // 8)
        async for evt in run_batch(allocation):
            yield evt
            emitted += 1
            if evt["type"] == "agent" and emitted % posterior_every == 0:
                yield {"type": "posterior", "bayesian": bayesian_ab(arms_now(), draws=4000)}
    else:
        wave = max(len(names), SWARM_CONCURRENCY)
        spent = 0
        wave_idx = 0
        while spent < total_budget:
            remaining = total_budget - spent
            if wave_idx == 0:
                # seed wave: equal split so every arm has a posterior to sample
                per_arm = max(1, min(remaining // len(names), wave // len(names)))
                allocation = {n: per_arm for n in names}
            else:
                # Forced-exploration floor: pure Thompson starves losing arms,
                # which would leave them below min_per_arm forever and make the
                # early-stop precondition unreachable. So first top up any arm
                # still under the floor, then Thompson-allocate the remainder.
                slots = min(wave, remaining)
                allocation = {}
                for n in names:
                    deficit = max(0, min_per_arm - len(per_variant[n]))
                    take = min(deficit, slots)
                    if take > 0:
                        allocation[n] = take
                        slots -= take
                if slots > 0:
                    sampled = thompson_allocate(arms_now(), slots, seed=seed + wave_idx)
                    for n, c in sampled.items():
                        if c > 0:
                            allocation[n] = allocation.get(n, 0) + c
            spent += sum(allocation.values())
            yield {"type": "wave", "index": wave_idx, "allocation": allocation}
            async for evt in run_batch(allocation):
                yield evt

            bayes = bayesian_ab(arms_now(), draws=8000)
            yield {"type": "posterior", "bayesian": bayes}
            enough = all(len(per_variant[n]) >= min_per_arm for n in names)
            if enough and bayes["conclusive"]:
                yield {
                    "type": "early_stop",
                    "spent": spent,
                    "budget": total_budget,
                    "saved": total_budget - spent,
                    "reason": (
                        f"expected loss {bayes['expected_loss']:.4f} < ROPE "
                        f"{bayes['rope']} with every arm ≥ {min_per_arm} samples"
                    ),
                }
                break
            wave_idx += 1

    result = _compare_result(
        per_variant, scenarios, cognitive_load, twins_per_persona * len(personas)
    )
    yield {"type": "done", "result": result.model_dump()}


async def stream_walkthrough(
    personas: list[PersonaSeed],
    steps: list[str],
    twins_per_persona: int,
    cognitive_load: Literal["low", "medium", "high"],
):
    """Streaming flow walkthrough with PER-STEP granularity: every gate decision
    of every twin is pushed onto a queue by the walking coroutine and relayed
    live, so the funnel visual shows agents advancing and dropping at the exact
    step it happens."""
    semaphore = asyncio.Semaphore(SWARM_CONCURRENCY)
    queue: asyncio.Queue[dict] = asyncio.Queue()
    total = len(personas) * twins_per_persona
    yield {"type": "start", "total": total, "steps": steps}

    async def walk_one(idx: int, persona: PersonaSeed):
        def on_step(ev: dict) -> None:
            queue.put_nowait({"type": "step", "twin": idx, "persona": persona.label, **ev})

        # THE `finally` IS THE LOAD-BEARING PART. This loop's termination
        # condition is a countdown of `twin_done` events, so a twin that dies
        # without enqueueing one leaves `pending` permanently above zero and the
        # `await queue.get()` below blocks forever — the HTTP connection hangs
        # open with the client having seen only the `start` frame, and every
        # sibling task is orphaned. Unlike every other streaming endpoint here,
        # which lets exceptions propagate out through `as_completed` into the
        # handler in main.py, this one does its own bookkeeping, so it has to
        # guarantee its own accounting.
        walk = None
        try:
            walk = await _walk_twin(persona, steps, cognitive_load, semaphore, on_step=on_step)
        finally:
            queue.put_nowait(
                {
                    "type": "twin_done",
                    "twin": idx,
                    "outcome": walk["outcome"] if walk else "failed",
                }
            )
        return walk

    tasks = [
        asyncio.create_task(walk_one(i, p))
        for i, p in enumerate(p for p in personas for _ in range(twins_per_persona))
    ]

    try:
        pending = len(tasks)
        while pending > 0 or not queue.empty():
            evt = await queue.get()
            if evt["type"] == "twin_done":
                pending -= 1
            yield evt

        # `gather(return_exceptions=True)` rather than `t.result()`: calling
        # .result() re-raises any task exception here, at the very end, after
        # the client has already been told every twin is done. Failed walks are
        # reported as outcome="failed" above, so they are simply dropped from
        # the aggregate.
        settled = await asyncio.gather(*tasks, return_exceptions=True)
        walks = [w for w in settled if isinstance(w, dict)]
    finally:
        # A client that navigates away closes the generator, which throws
        # GeneratorExit in here. Without this, the twin tasks keep running to
        # completion and keep billing tokens that nothing will ever read.
        for t in tasks:
            if not t.done():
                t.cancel()

    aggregate = aggregate_walks(
        walks, steps, cognitive_load, _LOAD_TO_CONTEXT_STEPS[cognitive_load]
    )
    yield {"type": "done", "result": aggregate.model_dump()}


# ------------------------------------------------------------------ flow walkthrough

WALK_PREAMBLE = """You are one synthetic persona inside a market-research
simulation, moving through a product flow ONE STEP AT A TIME. You only know the
steps you have already seen — never assume what comes next. React exactly as
your persona would: skim, misread, get impatient, abandon when it isn't worth
the effort, convert when genuinely persuaded. Do not be helpful or agreeable;
be human."""


def _walk_user_message(step: str, step_index: int, memory: list[dict]) -> str:
    memory_block = (
        "Your memory of the flow so far (older steps may have faded):\n"
        + json.dumps(memory, indent=0)
        if memory
        else "This is the first thing you see."
    )
    return (
        f"{memory_block}\n\n"
        f"You now arrive at step {step_index + 1}:\n{step}\n\n"
        "React in character. Write inner_monologue first, then decide: continue, "
        "hesitate (advance, but struggling), abandon, or convert."
    )


async def _walk_twin(
    persona: PersonaSeed,
    steps: list[str],
    cognitive_load: Literal["low", "medium", "high"],
    semaphore: asyncio.Semaphore,
    on_step: Optional[Callable[[dict], None]] = None,
) -> dict | None:
    """One twin walks the flow. The context throttle is the whole point here:
    the twin's memory of previous steps is truncated to a rolling window sized
    by cognitive load — a tired persona at step 6 barely remembers step 1."""
    window = _LOAD_TO_CONTEXT_STEPS[cognitive_load]
    temperature = LOAD_TO_TEMPERATURE[cognitive_load]
    memory: list[dict] = []
    records: list[dict] = []
    outcome = "completed"

    for index, step in enumerate(steps):
        messages = [
            {"role": "system", "content": WALK_PREAMBLE},
            {
                "role": "system",
                "content": persona.system_prompt
                + "\n\nCurrent state: "
                + _LOAD_CONDITIONING[cognitive_load],
            },
            {"role": "user", "content": _walk_user_message(step, index, memory[-window:])},
        ]
        async with semaphore:
            try:
                completion = await async_client().chat.completions.create(
                    model=TWIN_MODEL,
                    messages=messages,
                    temperature=temperature,
                    response_format=response_format_for(TwinStepReaction),
                )
                reaction = parse_completion(completion, TwinStepReaction)
            except (openai.OpenAIError, Refusal, RuntimeError, ValueError):
                return None

        records.append({"step_index": index, "step": step, **reaction.model_dump()})
        if on_step is not None:
            on_step(
                {
                    "step_index": index,
                    "action": reaction.action,
                    "engagement": round(reaction.engagement, 3),
                }
            )
        memory.append(
            {"step": step, "action": reaction.action, "felt": reaction.inner_monologue}
        )
        if reaction.action == "abandon":
            outcome = "abandoned"
            break
        if reaction.action == "convert":
            outcome = "converted"
            break

    return {"persona_label": persona.label, "outcome": outcome, "records": records}


def aggregate_walks(
    walks: list[dict],
    steps: list[str],
    cognitive_load: str,
    context_window_steps: int,
) -> WalkAggregate:
    """Pure aggregation — separated from the LLM loop so it's unit-testable."""
    if not walks:
        raise RuntimeError("Walkthrough produced no valid twin traces.")

    stats: list[WalkStepStat] = []
    for index, step in enumerate(steps):
        at_step = [
            r for w in walks for r in w["records"] if r["step_index"] == index
        ]
        stats.append(
            WalkStepStat(
                step_index=index,
                step=step,
                entered=len(at_step),
                abandoned=sum(1 for r in at_step if r["action"] == "abandon"),
                converted=sum(1 for r in at_step if r["action"] == "convert"),
                # None, not 0.0. A step nobody reached has no mean engagement,
                # and 0.0 is the *worst possible score* — so an unreached step
                # rendered identically to a step every twin hated, with the
                # funnel chart drawing a hard floor at the exact point the data
                # ran out. WalkStepStat.mean_engagement is Optional[float].
                mean_engagement=(
                    round(sum(r["engagement"] for r in at_step) / len(at_step), 3)
                    if at_step
                    else None
                ),
            )
        )

    total = len(walks)
    abandoned = sum(1 for w in walks if w["outcome"] == "abandoned")
    converted = sum(1 for w in walks if w["outcome"] == "converted")

    # Kaplan-Meier: hazard normalises drop-off by how many agents actually
    # reached each step, so the true killer step is identified rather than
    # simply the step the most agents happened to pass through.
    survival = kaplan_meier(
        entered=[s.entered for s in stats], dropped=[s.abandoned for s in stats]
    )

    return WalkAggregate(
        run_id="",
        twin_count=total,
        cognitive_load=cognitive_load,
        context_window_steps=context_window_steps,
        completion_rate=round((total - abandoned) / total, 3),
        conversion_rate=round(converted / total, 3),
        steps=stats,
        walks=walks[:20],
        survival=survival,
    )


async def run_walkthrough(
    personas: list[PersonaSeed],
    steps: list[str],
    twins_per_persona: int,
    cognitive_load: Literal["low", "medium", "high"],
) -> WalkAggregate:
    semaphore = asyncio.Semaphore(SWARM_CONCURRENCY)
    tasks = [
        _walk_twin(p, steps, cognitive_load, semaphore)
        for p in personas
        for _ in range(twins_per_persona)
    ]
    results = await asyncio.gather(*tasks)
    walks = [w for w in results if w is not None]
    return aggregate_walks(
        walks, steps, cognitive_load, _LOAD_TO_CONTEXT_STEPS[cognitive_load]
    )
