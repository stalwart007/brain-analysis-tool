"""Advanced study types that reuse the persona swarm.

- Price Sensitivity: each twin rates purchase intent at every candidate price;
  we aggregate into a demand curve and a revenue curve and pick the
  revenue-maximizing price.
- Objection Radar: each twin surfaces its single biggest objection + whether
  it's a dealbreaker + sentiment; we aggregate into ranked objection themes.

The LLM fan-out mirrors swarm.py (bounded concurrency, graceful degradation);
the aggregation functions are pure and unit-tested without any API call.
"""

import asyncio
from collections import Counter
from typing import Literal

import openai

from .streaming import as_they_land
from .analytics import (
    auto_cluster,
    pca_2d,
    bootstrap_ci,
    medoid,
    price_elasticity,
)
from .config import EMBEDDING_MODEL, LOAD_TO_TEMPERATURE, SWARM_CONCURRENCY, TWIN_MODEL
from .oai import Refusal, async_client, parse_completion, response_format_for
from .schemas import (
    ObjectionResult,
    ObjectionTheme,
    PersonaSeed,
    PricePoint,
    PriceSensitivityResult,
    TwinObjection,
    TwinPriceCurve,
)
from .swarm import _LOAD_CONDITIONING

Load = Literal["low", "medium", "high"]


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _persona_system(persona: PersonaSeed, load: Load) -> dict:
    return {
        "role": "system",
        "content": persona.system_prompt + "\n\nCurrent state: " + _LOAD_CONDITIONING[load],
    }


# ------------------------------------------------------------------ price sensitivity

_PRICE_PREAMBLE = """You are one synthetic consumer persona in a pricing study.
You will see a product and a list of candidate prices. For EACH price, in the
exact order given, report how likely you personally are to buy at that price:
0.0 = would never buy, 1.0 = would definitely buy. Be human and honest — your
own budget, priorities, and mood matter. Higher prices should generally lower
your intent, but a price that feels 'too cheap to be good' can lower it too."""


async def _price_twin(
    persona: PersonaSeed,
    product: str,
    prices: list[float],
    load: Load,
    semaphore: asyncio.Semaphore,
) -> list[float] | None:
    messages = [
        {"role": "system", "content": _PRICE_PREAMBLE},
        _persona_system(persona, load),
        {
            "role": "user",
            "content": f"Product:\n{product}\n\nPrices to rate, in this order: {prices}",
        },
    ]
    async with semaphore:
        try:
            completion = await async_client().chat.completions.create(
                model=TWIN_MODEL,
                messages=messages,
                temperature=LOAD_TO_TEMPERATURE[load],
                response_format=response_format_for(TwinPriceCurve),
            )
            curve = parse_completion(completion, TwinPriceCurve)
        except (openai.OpenAIError, Refusal, RuntimeError, ValueError):
            return None
    # drop twins that didn't answer for every price
    if len(curve.intents) != len(prices):
        return None
    return [_clamp01(x) for x in curve.intents]


def aggregate_price_curves(
    curves: list[list[float]], prices: list[float], load: str, product: str
) -> PriceSensitivityResult:
    if not curves:
        raise RuntimeError("Price study produced no valid twin curves.")

    demand = [
        round(sum(c[i] for c in curves) / len(curves), 3) for i in range(len(prices))
    ]
    revenue_raw = [prices[i] * demand[i] for i in range(len(prices))]
    peak = max(revenue_raw) or 1.0
    points = [
        PricePoint(
            price=prices[i],
            demand=demand[i],
            revenue_index=round(revenue_raw[i] / peak, 3),
        )
        for i in range(len(prices))
    ]
    best_i = max(range(len(prices)), key=lambda i: revenue_raw[i])
    return PriceSensitivityResult(
        run_id="",
        twin_count=len(curves),
        cognitive_load=load,
        product_preview=product[:200],
        points=points,
        recommended_price=prices[best_i],
        econometrics=price_elasticity(prices, demand),
    )


async def run_price_sensitivity(
    personas: list[PersonaSeed],
    product: str,
    prices: list[float],
    twins_per_persona: int,
    cognitive_load: Load,
) -> PriceSensitivityResult:
    semaphore = asyncio.Semaphore(SWARM_CONCURRENCY)
    tasks = [
        _price_twin(p, product, prices, cognitive_load, semaphore)
        for p in personas
        for _ in range(twins_per_persona)
    ]
    results = await asyncio.gather(*tasks)
    curves = [c for c in results if c is not None]
    return aggregate_price_curves(curves, prices, cognitive_load, product)


async def stream_price_sensitivity(
    personas: list[PersonaSeed],
    product: str,
    prices: list[float],
    twins_per_persona: int,
    cognitive_load: Load,
):
    """Streaming price study: each buyer-agent's full per-price intent curve is
    emitted the moment it lands, so the demand field builds live column by column."""
    semaphore = asyncio.Semaphore(SWARM_CONCURRENCY)
    total = len(personas) * twins_per_persona
    yield {"type": "start", "total": total, "prices": prices}

    dispatch = as_they_land(
        _price_twin(p, product, prices, cognitive_load, semaphore)
        for p in personas
        for _ in range(twins_per_persona)
    )
    curves: list[list[float]] = []
    async for curve in dispatch:
        if curve is None:
            yield {"type": "agent_failed"}
            continue
        curves.append(curve)
        # running mean demand per price so the UI can draw the live curve
        running = [
            round(sum(c[i] for c in curves) / len(curves), 3) for i in range(len(prices))
        ]
        yield {"type": "agent", "intents": curve, "running_demand": running}

    result = aggregate_price_curves(curves, prices, cognitive_load, product)
    yield {"type": "done", "result": result.model_dump()}


# ------------------------------------------------------------------ objection radar

_OBJECTION_PREAMBLE = """You are one synthetic persona reacting to a pitch,
claim, or piece of marketing copy. React as a real, slightly skeptical human
would. Report your overall sentiment, the single biggest objection or doubt you
have (or 'none' if fully convinced), whether that objection alone would stop you
from acting, and how likely you'd be to recommend it to a friend."""


async def _objection_twin(
    persona: PersonaSeed,
    pitch: str,
    load: Load,
    semaphore: asyncio.Semaphore,
) -> TwinObjection | None:
    messages = [
        {"role": "system", "content": _OBJECTION_PREAMBLE},
        _persona_system(persona, load),
        {"role": "user", "content": f"The pitch:\n{pitch}"},
    ]
    async with semaphore:
        try:
            completion = await async_client().chat.completions.create(
                model=TWIN_MODEL,
                messages=messages,
                temperature=LOAD_TO_TEMPERATURE[load],
                response_format=response_format_for(TwinObjection),
            )
            return parse_completion(completion, TwinObjection)
        except (openai.OpenAIError, Refusal, RuntimeError, ValueError):
            return None


def _substantive(objs: list[TwinObjection]) -> list[TwinObjection]:
    """Drop the "no objection" responses — they are not a theme."""
    return [
        o
        for o in objs
        if o.biggest_objection.strip().lower() not in ("", "none", "n/a", "no objection")
    ]


def aggregate_objections(
    objs: list[TwinObjection],
    load: str,
    pitch: str,
    embeddings: list[list[float]] | None = None,
) -> ObjectionResult:
    """Aggregate twin objections into themes.

    When `embeddings` are supplied (one per substantive objection, same order),
    themes are formed by average-linkage agglomerative clustering over cosine
    similarity, and each theme is labelled by its medoid — so "it's too pricey"
    and "the price is too high" become ONE theme of size 2 rather than two
    themes of size 1. Without embeddings we fall back to case-insensitive exact
    grouping, which is what the old implementation did and why near-duplicate
    objections never merged.
    """
    if not objs:
        raise RuntimeError("Objection scan produced no valid twin responses.")

    sentiment = Counter(o.sentiment for o in objs)
    dealbreakers = sum(1 for o in objs if o.is_dealbreaker)
    recommends = [_clamp01(o.would_recommend) for o in objs]
    substantive = _substantive(objs)

    themes: list[ObjectionTheme] = []
    method = "exact"

    silhouette: float | None = None
    chosen_threshold: float | None = None
    theme_map: list[dict] | None = None
    if embeddings and len(embeddings) == len(substantive) and substantive:
        method = "semantic"
        # merge threshold chosen by maximising silhouette — model selection by
        # internal validity, not a hand-tuned constant.
        clusters, chosen_threshold, silhouette = auto_cluster(embeddings)
        # Kernel-PCA the embedding cloud to 2-D so each theme has *semantic*
        # coordinates: nearby dots on the map are genuinely similar complaints.
        coords = pca_2d(embeddings)
        theme_map = []
        for cluster in clusters:
            rep = medoid(cluster, embeddings)
            texts = [substantive[i].biggest_objection.strip() for i in cluster]
            label = substantive[rep].biggest_objection.strip()
            themes.append(
                ObjectionTheme(
                    objection=label,
                    count=len(cluster),
                    dealbreaker_count=sum(1 for i in cluster if substantive[i].is_dealbreaker),
                    variants=[t for t in texts if t != label][:5],
                )
            )
            if coords is not None:
                # theme position = centroid of its members' PCA projections
                cx = sum(coords[i][0] for i in cluster) / len(cluster)
                cy = sum(coords[i][1] for i in cluster) / len(cluster)
                theme_map.append(
                    {
                        "x": round(cx, 4),
                        "y": round(cy, 4),
                        "count": len(cluster),
                        "dealbreaker": any(substantive[i].is_dealbreaker for i in cluster),
                        "label": label[:80],
                    }
                )
        if not theme_map:
            theme_map = None
    else:
        grouped: dict[str, dict] = {}
        for o in substantive:
            key = o.biggest_objection.strip().lower()
            g = grouped.setdefault(key, {"label": o.biggest_objection.strip(), "count": 0, "db": 0})
            g["count"] += 1
            if o.is_dealbreaker:
                g["db"] += 1
        themes = [
            ObjectionTheme(objection=g["label"], count=g["count"], dealbreaker_count=g["db"])
            for g in grouped.values()
        ]

    themes.sort(key=lambda t: (t.count, t.dealbreaker_count), reverse=True)
    ci = bootstrap_ci(recommends)

    return ObjectionResult(
        run_id="",
        twin_count=len(objs),
        cognitive_load=load,
        pitch_preview=pitch[:200],
        sentiment_distribution=dict(sentiment),
        dealbreaker_rate=round(dealbreakers / len(objs), 3),
        mean_recommend=round(sum(recommends) / len(objs), 3),
        top_objections=themes[:6],
        clustering_method=method,
        recommend_ci=list(ci) if ci else None,
        cluster_silhouette=round(silhouette, 4) if silhouette is not None else None,
        cluster_threshold=chosen_threshold,
        theme_map=theme_map,
    )


async def embed_texts(texts: list[str]) -> list[list[float]] | None:
    """Embed objection texts for semantic clustering. Returns None on any
    failure so the caller degrades to exact grouping rather than losing the run."""
    if not texts:
        return None
    try:
        response = await async_client().embeddings.create(model=EMBEDDING_MODEL, input=texts)
        return [item.embedding for item in response.data]
    except (openai.OpenAIError, Exception):
        return None


async def run_objection_scan(
    personas: list[PersonaSeed],
    pitch: str,
    twins_per_persona: int,
    cognitive_load: Load,
) -> ObjectionResult:
    semaphore = asyncio.Semaphore(SWARM_CONCURRENCY)
    tasks = [
        _objection_twin(p, pitch, cognitive_load, semaphore)
        for p in personas
        for _ in range(twins_per_persona)
    ]
    results = await asyncio.gather(*tasks)
    objs = [o for o in results if o is not None]
    if not objs:
        raise RuntimeError("Objection scan produced no valid twin responses.")
    # Embed the substantive objections so semantically identical complaints
    # collapse into a single theme instead of fragmenting the ranking.
    texts = [o.biggest_objection.strip() for o in _substantive(objs)]
    embeddings = await embed_texts(texts)
    return aggregate_objections(objs, cognitive_load, pitch, embeddings=embeddings)


async def stream_objection_scan(
    personas: list[PersonaSeed],
    pitch: str,
    twins_per_persona: int,
    cognitive_load: Load,
):
    """Streaming objection scan: each twin's verdict is emitted as it lands (so
    the sonar paints blips live), then a 'clustering' phase event marks the
    embedding + silhouette-optimised merge before the final themed result."""
    semaphore = asyncio.Semaphore(SWARM_CONCURRENCY)
    total = len(personas) * twins_per_persona
    yield {"type": "start", "total": total}

    dispatch = as_they_land(
        _objection_twin(p, pitch, cognitive_load, semaphore)
        for p in personas
        for _ in range(twins_per_persona)
    )
    objs: list[TwinObjection] = []
    async for o in dispatch:
        if o is None:
            yield {"type": "agent_failed"}
            continue
        objs.append(o)
        yield {
            "type": "agent",
            "sentiment": o.sentiment,
            "dealbreaker": o.is_dealbreaker,
            "objection": o.biggest_objection.strip()[:120],
            "recommend": round(_clamp01(o.would_recommend), 3),
        }

    if not objs:
        raise RuntimeError("Objection scan produced no valid twin responses.")
    yield {"type": "clustering", "note": "embedding objections + silhouette-optimised merge"}
    texts = [o.biggest_objection.strip() for o in _substantive(objs)]
    embeddings = await embed_texts(texts)
    result = aggregate_objections(objs, cognitive_load, pitch, embeddings=embeddings)
    yield {"type": "done", "result": result.model_dump()}
