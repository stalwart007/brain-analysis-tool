"""CogniSwarm API + dev pages.

Public (consent-gated, no key):
  POST /v1/ingest                     telemetry ingestion (SDK beacon target)

Analysis surface (X-API-Key when COGNISWARM_API_KEYS is set — the headless API):
  GET  /v1/sessions                   list ingested session segments
  POST /v1/sessions/{id}/profile      Layer 4: features -> BehavioralSignal -> PersonaSeed
  POST /v1/swarm/run                  fan one scenario across the twin swarm
  POST /v1/swarm/compare              zero-shot A/B: rank N variants against the same swarm
  POST /v1/swarm/walk                 multi-step flow walkthrough (context-throttled twins)
  GET  /v1/swarm/runs                 past runs (all kinds)
  POST /v1/runs/{id}/actuals          record real-world outcomes for a run
  GET  /v1/validation/report          calibration report (MAE / bias, predicted vs actual)

Dev pages: GET / (legacy mini-dashboard), GET /demo (instrumented demo site)
"""

import asyncio
import hmac
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import json

import openai
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import ValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from . import jobs, storage
from .config import REPO_ROOT
from .persona import seed_persona
from .cognition import full_cognitive_profile, iter_cognitive_profile
from .profiler import profile_features
from .schemas import (
    ActualsPayload,
    CalibrationReport,
    CompareRequest,
    FeaturePayload,
    IngestEnvelope,
    JobCreate,
    PanelMemberCreate,
    PersonaSeed,
    SwarmRunRequest,
    WalkRequest,
)
from .neuro import stream_content_study
from .optimizer import stream_copy_optimizer
from .schemas import (
    ContentStudyRequest,
    CopyOptimizerRequest,
    ObjectionRequest,
    PriceSensitivityRequest,
    SequenceRequest,
    ViralityRequest,
)
from .sequence import stream_sequence
from .virality import stream_virality
from .studies import (
    run_objection_scan,
    run_price_sensitivity,
    stream_objection_scan,
    stream_price_sensitivity,
)
from .swarm import (
    run_compare,
    run_swarm,
    run_walkthrough,
    stream_compare,
    stream_swarm,
    stream_walkthrough,
)
from .validation import calibration_report


def require_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    """Headless-API auth.

    FAILS CLOSED. Returning early when COGNISWARM_API_KEYS is unset would mean
    a typo in the variable name, a dropped secret, or a container started
    without it silently publishes the entire analysis surface — sessions, panel
    members, run history with every twin's inner monologue, and every endpoint
    that spends OpenAI credit — with no authentication and no signal that
    anything is wrong. An auth check whose failure mode is "allow everything"
    is not an auth check.

    Local development opts out explicitly instead, via
    COGNISWARM_ALLOW_ANONYMOUS=1, so the open state is always something someone
    chose rather than something that happened.
    """
    keys = {k.strip() for k in os.environ.get("COGNISWARM_API_KEYS", "").split(",") if k.strip()}
    if keys:
        # Configured keys always win: the anonymous escape hatch below can never
        # weaken a server that has been given keys.
        #
        # constant-time compare — `in` on a set of strings short-circuits on the
        # first differing byte, which leaks key length and prefix under timing.
        if not (x_api_key and any(hmac.compare_digest(x_api_key, k) for k in keys)):
            raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key.")
        return

    if os.environ.get("COGNISWARM_ALLOW_ANONYMOUS", "").strip().lower() in {"1", "true", "yes"}:
        return

    raise HTTPException(
        status_code=503,
        detail=(
            "Server is not configured for API access: set COGNISWARM_API_KEYS, "
            "or COGNISWARM_ALLOW_ANONYMOUS=1 for local development."
        ),
    )


@asynccontextmanager
async def lifespan(_: FastAPI):
    jobs.start_worker()
    yield
    jobs.stop_worker()


app = FastAPI(title="CogniSwarm", version="0.2.0", lifespan=lifespan)

# Browsers must be told which origins may talk to this server. The wildcard was
# marked "dev only" in a comment but had no way to be anything else, so it
# shipped as written — and paired with the fail-open auth above it meant any
# page a developer visited could read the whole telemetry corpus off localhost.
_ALLOWED_ORIGINS = [
    o.strip() for o in os.environ.get("COGNISWARM_ALLOWED_ORIGINS", "").split(",") if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

storage.init_db()

_APP_DIR = Path(__file__).resolve().parent
_DEMO_DIR = REPO_ROOT / "examples" / "demo-site"
_COLLECTOR_DIST = REPO_ROOT / "packages" / "collector" / "dist"

if _COLLECTOR_DIST.exists():
    app.mount("/static/collector", StaticFiles(directory=_COLLECTOR_DIST), name="collector")


# ------------------------------------------------------------------ ingestion


@app.post("/v1/ingest", status_code=202)
def ingest(envelope: IngestEnvelope) -> dict:
    # Hard consent gate, server-side too: never trust the client alone.
    if envelope.consent is not True:
        raise HTTPException(status_code=403, detail="Telemetry without consent is rejected.")
    if envelope.features.event_count == 0:
        return {"status": "ignored", "reason": "empty segment"}

    panel_member_id: Optional[str] = None
    if envelope.panel_token:
        member = storage.get_member_by_token(envelope.panel_token)
        # Panel telemetry requires an active, consented, non-revoked membership.
        if member is None or member["revoked_at"] or not member["consented_at"]:
            raise HTTPException(status_code=403, detail="Invalid or inactive panel membership.")
        panel_member_id = member["id"]

    session_id = storage.insert_session(
        envelope.site_id,
        envelope.page_path,
        envelope.features.model_dump(),
        panel_member_id=panel_member_id,
    )
    return {"status": "accepted", "session_id": session_id}


@app.get("/v1/sessions", dependencies=[Depends(require_api_key)])
def sessions() -> list[dict]:
    return storage.list_sessions()


# ------------------------------------------------------------------ profiling


def _cognition_inputs(features: FeaturePayload) -> dict:
    """Map stored telemetry onto the cognition engine's inputs.

    `completion_ratio` is deliberately **None**.

    The old proxy was `(click_count - rage_click_bursts * 3) / click_count`,
    which is not an accuracy in any sense the drift-diffusion model recognises,
    and it was numerically degenerate besides: because the burst counter
    reported one burst per sliding *window*, the subtracted term exceeded the
    click count whenever any burst existed at all. The proxy therefore took
    exactly two values — 1.0 with no rage burst, 0.0 with one — and `v` is
    antisymmetric about p = 0.5, so a single triple-click flipped the reported
    drift rate from +0.279 to -0.279 on identical latencies while the
    interpretation string still read "efficient evidence accumulation".

    `ez_diffusion` was already written to refuse without an observed accuracy
    (see the note above the `ddm_skip` branch in cognition.py); this caller was
    manufacturing the very input that refusal exists to demand. Passing None
    restores it: the session lands in `models_skipped` with a reason attached,
    which is visible and honest rather than confidently wrong.

    Restoring the DDM needs a real success/abandon signal — a declared goal
    zone, or a completion event class from the collector — not a rearrangement
    of click counts.
    """
    return {
        "event_stream": features.event_stream or None,
        "velocity_series": features.velocity_series or None,
        "decision_latencies_ms": features.decision_latencies_ms or None,
        "completion_ratio": None,
    }


@app.post("/v1/sessions/{session_id}/profile", dependencies=[Depends(require_api_key)])
def profile(session_id: str) -> dict:
    session = storage.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Unknown session")

    features = FeaturePayload(**session["features"])
    cognition = full_cognitive_profile(**_cognition_inputs(features))
    storage.set_cognition(session_id, cognition)
    try:
        signal = profile_features(features, cognition.get("summary"))
    except (openai.OpenAIError, RuntimeError, ValueError) as exc:
        raise _llm_errors(exc) from exc
    persona = seed_persona(signal)
    storage.set_signal(session_id, signal.model_dump())
    storage.set_persona(session_id, persona.model_dump())
    return {
        "signal": signal.model_dump(),
        "persona": persona.model_dump(),
        "cognition": cognition,
    }


@app.post(
    "/v1/sessions/{session_id}/profile/stream",
    dependencies=[Depends(require_api_key)],
)
async def profile_stream(session_id: str) -> StreamingResponse:
    """The full profiling pipeline as SSE — every model's computation is
    streamed as it happens so the frontend can show the real work: EM traces,
    BIC comparisons, fitted parameters, then the LLM synthesis stages."""
    session = storage.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Unknown session")
    features = FeaturePayload(**session["features"])

    async def gen():
        try:
            frames = iter_cognitive_profile(**_cognition_inputs(features))
            cognition: dict = {}
            sentinel = object()
            while True:
                frame = await asyncio.to_thread(next, frames, sentinel)
                if frame is sentinel:
                    break
                if frame["stage"] == "complete":
                    cognition = frame["profile"]
                yield f"data: {json.dumps({'type': 'cognition', **frame})}\n\n"
            storage.set_cognition(session_id, cognition)

            yield f"data: {json.dumps({'type': 'llm', 'stage': 'signal', 'status': 'running', 'detail': 'behavioral analyst synthesising segment signals'})}\n\n"
            signal = await asyncio.to_thread(
                profile_features, features, cognition.get("summary")
            )
            storage.set_signal(session_id, signal.model_dump())
            yield f"data: {json.dumps({'type': 'llm', 'stage': 'signal', 'status': 'done', 'result': signal.model_dump()})}\n\n"

            yield f"data: {json.dumps({'type': 'llm', 'stage': 'persona', 'status': 'running', 'detail': 'seeding synthetic persona from signals + model parameters'})}\n\n"
            persona = await asyncio.to_thread(seed_persona, signal)
            storage.set_persona(session_id, persona.model_dump())
            yield f"data: {json.dumps({'type': 'done', 'result': {'signal': signal.model_dump(), 'persona': persona.model_dump(), 'cognition': cognition}})}\n\n"
        except (openai.OpenAIError, RuntimeError, ValueError) as exc:
            detail = _llm_errors(exc).detail
            yield f"data: {json.dumps({'type': 'error', 'detail': detail})}\n\n"

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ------------------------------------------------------------------ swarm


def _load_personas(session_ids: list[str]) -> list[PersonaSeed]:
    candidates = (
        [storage.get_session(sid) for sid in session_ids]
        if session_ids
        else storage.list_sessions()
    )
    personas = [
        PersonaSeed(**s["persona"])
        for s in candidates
        if s is not None and s.get("persona")
    ]
    if not personas:
        raise HTTPException(
            status_code=400,
            detail="No profiled personas available. Profile at least one session first.",
        )
    return personas


def _llm_errors(exc: Exception) -> HTTPException:
    # pydantic's ValidationError subclasses ValueError, NOT RuntimeError, so
    # the handlers here used to miss it entirely: a schema violation surfaced
    # as a bare 500 on the JSON routes, and on the SSE routes it raised after
    # the headers were already sent, truncating the body with no terminal
    # frame — which the client cannot distinguish from a successful empty run.
    #
    # It is reachable by construction: `oai._clean` strips `minimum`/`maximum`/
    # `pattern` from the wire schema (strict mode rejects them), so the model
    # may legally return `confidence: 1.4`, and `parse_completion` then
    # re-applies the real Pydantic bounds and raises. Every per-twin handler in
    # swarm/studies/neuro already caught ValueError; these did not.
    if isinstance(exc, ValidationError):
        return HTTPException(
            status_code=502,
            detail=f"Model returned values outside the response schema: {exc}",
        )
    # AuthenticationError ⊂ APIError ⊂ OpenAIError — check most specific first.
    if isinstance(exc, openai.AuthenticationError):
        return HTTPException(
            status_code=503,
            detail="OpenAI credentials invalid — check OPENAI_API_KEY and restart.",
        )
    if isinstance(exc, openai.APIError):
        return HTTPException(status_code=502, detail=f"OpenAI API error: {exc}")
    if isinstance(exc, openai.OpenAIError):  # e.g. no API key at client construction
        return HTTPException(
            status_code=503,
            detail="OpenAI credentials not configured — set OPENAI_API_KEY and restart.",
        )
    if isinstance(exc, RuntimeError):
        return HTTPException(status_code=502, detail=str(exc))
    if isinstance(exc, ValueError):
        return HTTPException(status_code=502, detail=f"Invalid model output: {exc}")
    raise exc


@app.post("/v1/swarm/run", dependencies=[Depends(require_api_key)])
async def swarm_run(request: SwarmRunRequest) -> dict:
    personas = _load_personas(request.session_ids)
    try:
        aggregate = await run_swarm(
            personas=personas,
            scenario=request.scenario,
            twins_per_persona=request.twins_per_persona,
            cognitive_load=request.cognitive_load,
        )
    except (openai.OpenAIError, RuntimeError, ValueError) as exc:
        raise _llm_errors(exc) from exc
    run_id = storage.insert_swarm_run(request.model_dump(), aggregate.model_dump())
    result = aggregate.model_dump()
    result["run_id"] = run_id
    return result


def _sse(generator, request_model, kind: str) -> StreamingResponse:
    """Wrap a simulation's async event generator as an SSE response, persisting
    the run when the terminal 'done' event passes through."""

    async def gen():
        try:
            async for evt in generator:
                if evt.get("type") == "done":
                    run_id = storage.insert_swarm_run(
                        request_model.model_dump(), evt["result"], kind=kind
                    )
                    evt["result"]["run_id"] = run_id
                yield f"data: {json.dumps(evt)}\n\n"
        except (openai.OpenAIError, RuntimeError, ValueError) as exc:
            detail = _llm_errors(exc).detail
            yield f"data: {json.dumps({'type': 'error', 'detail': detail})}\n\n"

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/v1/swarm/stream", dependencies=[Depends(require_api_key)])
async def swarm_stream(request: SwarmRunRequest) -> StreamingResponse:
    personas = _load_personas(request.session_ids)
    return _sse(
        stream_swarm(
            personas, request.scenario, request.twins_per_persona, request.cognitive_load
        ),
        request,
        "swarm",
    )


@app.post("/v1/swarm/compare/stream", dependencies=[Depends(require_api_key)])
async def compare_stream(request: CompareRequest) -> StreamingResponse:
    names = [v.name for v in request.variants]
    if len(set(names)) != len(names):
        raise HTTPException(status_code=400, detail="Variant names must be unique.")
    personas = _load_personas(request.session_ids)
    return _sse(
        stream_compare(
            personas,
            request.variants,
            request.twins_per_persona,
            request.cognitive_load,
            adaptive=request.adaptive,
        ),
        request,
        "compare",
    )


@app.post("/v1/swarm/walk/stream", dependencies=[Depends(require_api_key)])
async def walk_stream(request: WalkRequest) -> StreamingResponse:
    personas = _load_personas(request.session_ids)
    return _sse(
        stream_walkthrough(
            personas, request.steps, request.twins_per_persona, request.cognitive_load
        ),
        request,
        "walk",
    )


@app.post("/v1/studies/price/stream", dependencies=[Depends(require_api_key)])
async def price_stream(request: PriceSensitivityRequest) -> StreamingResponse:
    personas = _load_personas(request.session_ids)
    return _sse(
        stream_price_sensitivity(
            personas,
            request.product,
            request.prices,
            request.twins_per_persona,
            request.cognitive_load,
        ),
        request,
        "price",
    )


@app.post("/v1/studies/objection/stream", dependencies=[Depends(require_api_key)])
async def objection_stream(request: ObjectionRequest) -> StreamingResponse:
    personas = _load_personas(request.session_ids)
    return _sse(
        stream_objection_scan(
            personas, request.pitch, request.twins_per_persona, request.cognitive_load
        ),
        request,
        "objection",
    )


@app.post("/v1/studies/virality/stream", dependencies=[Depends(require_api_key)])
async def virality_stream(request: ViralityRequest) -> StreamingResponse:
    """Virality forecast: Galton-Watson branching process over twin share
    intents — R0, extinction probability, seeded cascade quantile bands."""
    personas = _load_personas(request.session_ids)
    return _sse(stream_virality(personas, request), request, "virality")


@app.post("/v1/studies/content/stream", dependencies=[Depends(require_api_key)])
async def content_stream(request: ContentStudyRequest) -> StreamingResponse:
    """Neuro-impact study: beat-by-beat audience response with ISC,
    change-point, peak-end memory, and functional-system mapping."""
    personas = _load_personas(request.session_ids)
    return _sse(stream_content_study(personas, request), request, "content")


@app.post("/v1/studies/optimize/stream", dependencies=[Depends(require_api_key)])
async def optimize_stream(request: CopyOptimizerRequest) -> StreamingResponse:
    """Copy optimiser: an evolutionary loop that WRITES better copy — twin-scored
    populations, Thompson-allocated budget within each generation, LLM crossover
    and mutation between them, and a win claimed only when the champion's
    credible interval clears the seed's."""
    personas = _load_personas(request.session_ids)
    return _sse(stream_copy_optimizer(personas, request), request, "optimize")


@app.post("/v1/studies/sequence/stream", dependencies=[Depends(require_api_key)])
async def sequence_stream(request: SequenceRequest) -> StreamingResponse:
    """Message-sequence optimiser: estimates a position-adjusted precedence
    matrix from a sampled subset of the N! orderings, then solves the linear
    ordering problem heuristically for the ordering that ends with the most
    intent — reporting primacy/recency separately from message strength."""
    personas = _load_personas(request.session_ids)
    return _sse(stream_sequence(personas, request), request, "sequence")


@app.post("/v1/swarm/compare", dependencies=[Depends(require_api_key)])
async def swarm_compare(request: CompareRequest) -> dict:
    names = [v.name for v in request.variants]
    if len(set(names)) != len(names):
        raise HTTPException(status_code=400, detail="Variant names must be unique.")
    personas = _load_personas(request.session_ids)
    try:
        compared = await run_compare(
            personas=personas,
            variants=request.variants,
            twins_per_persona=request.twins_per_persona,
            cognitive_load=request.cognitive_load,
        )
    except (openai.OpenAIError, RuntimeError, ValueError) as exc:
        raise _llm_errors(exc) from exc
    run_id = storage.insert_swarm_run(request.model_dump(), compared.model_dump(), kind="compare")
    result = compared.model_dump()
    result["run_id"] = run_id
    return result


@app.post("/v1/swarm/walk", dependencies=[Depends(require_api_key)])
async def swarm_walk(request: WalkRequest) -> dict:
    personas = _load_personas(request.session_ids)
    try:
        aggregate = await run_walkthrough(
            personas=personas,
            steps=request.steps,
            twins_per_persona=request.twins_per_persona,
            cognitive_load=request.cognitive_load,
        )
    except (openai.OpenAIError, RuntimeError, ValueError) as exc:
        raise _llm_errors(exc) from exc
    run_id = storage.insert_swarm_run(request.model_dump(), aggregate.model_dump(), kind="walk")
    result = aggregate.model_dump()
    result["run_id"] = run_id
    return result


@app.get("/v1/swarm/runs", dependencies=[Depends(require_api_key)])
def swarm_runs(kind: Optional[str] = None) -> list[dict]:
    return storage.list_swarm_runs(kind=kind)


# ------------------------------------------------------------------ studies (advanced use cases)


@app.post("/v1/studies/price", dependencies=[Depends(require_api_key)])
async def study_price(request: PriceSensitivityRequest) -> dict:
    personas = _load_personas(request.session_ids)
    try:
        result = await run_price_sensitivity(
            personas=personas,
            product=request.product,
            prices=request.prices,
            twins_per_persona=request.twins_per_persona,
            cognitive_load=request.cognitive_load,
        )
    except (openai.OpenAIError, RuntimeError, ValueError) as exc:
        raise _llm_errors(exc) from exc
    run_id = storage.insert_swarm_run(request.model_dump(), result.model_dump(), kind="price")
    out = result.model_dump()
    out["run_id"] = run_id
    return out


@app.post("/v1/studies/objection", dependencies=[Depends(require_api_key)])
async def study_objection(request: ObjectionRequest) -> dict:
    personas = _load_personas(request.session_ids)
    try:
        result = await run_objection_scan(
            personas=personas,
            pitch=request.pitch,
            twins_per_persona=request.twins_per_persona,
            cognitive_load=request.cognitive_load,
        )
    except (openai.OpenAIError, RuntimeError, ValueError) as exc:
        raise _llm_errors(exc) from exc
    run_id = storage.insert_swarm_run(request.model_dump(), result.model_dump(), kind="objection")
    out = result.model_dump()
    out["run_id"] = run_id
    return out


# ------------------------------------------------------------------ validation


@app.post("/v1/runs/{run_id}/actuals", dependencies=[Depends(require_api_key)])
def record_actuals(run_id: str, payload: ActualsPayload) -> dict:
    if payload.engagement is None and payload.intent is None:
        raise HTTPException(status_code=400, detail="Provide at least one metric.")
    if not storage.set_actuals(run_id, payload.model_dump(exclude_none=True)):
        raise HTTPException(status_code=404, detail="Unknown run")
    return {"status": "recorded", "run_id": run_id}


@app.get("/v1/validation/report", dependencies=[Depends(require_api_key)])
def validation_report() -> CalibrationReport:
    return calibration_report(storage.list_swarm_runs(limit=500, kind="swarm"))


# ------------------------------------------------------------------ jobs (Phase 3)


@app.post("/v1/jobs", status_code=202, dependencies=[Depends(require_api_key)])
def create_job(request: JobCreate) -> dict:
    # Validate the payload against the target request schema up front, so a bad
    # job fails at submission rather than silently in the worker.
    schema = {
        "swarm": SwarmRunRequest,
        "batch_swarm": SwarmRunRequest,
        "compare": CompareRequest,
        "walk": WalkRequest,
    }[request.kind]
    try:
        payload = schema(**request.payload).model_dump()
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors(include_url=False)) from exc
    job_id = jobs.enqueue(request.kind, payload)
    return {"job_id": job_id, "status": "queued"}


@app.get("/v1/jobs", dependencies=[Depends(require_api_key)])
def list_jobs() -> list[dict]:
    return storage.list_jobs()


@app.get("/v1/jobs/{job_id}", dependencies=[Depends(require_api_key)])
def get_job(job_id: str) -> dict:
    job = storage.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job")
    return job


# ------------------------------------------------------------------ panel (Phase 4)


@app.post("/v1/panel/members", dependencies=[Depends(require_api_key)])
def create_panel_member(request: PanelMemberCreate) -> dict:
    member = storage.insert_panel_member(request.label)
    member["disclosure_url"] = f"/panel/{member['token']}"
    return member


@app.get("/v1/panel/members", dependencies=[Depends(require_api_key)])
def panel_members() -> list[dict]:
    # tokens are capability URLs — do not leak them wholesale in listings
    return [
        {k: v for k, v in m.items() if k != "token"} | {"disclosure_url": f"/panel/{m['token']}"}
        for m in storage.list_panel_members()
    ]


@app.post("/v1/panel/{token}/consent")
def panel_consent(token: str) -> dict:
    member = storage.get_member_by_token(token)
    if member is None or member["revoked_at"]:
        raise HTTPException(status_code=404, detail="Invalid or revoked invitation.")
    storage.set_member_consent(member["id"])
    return {"status": "enrolled", "member_id": member["id"]}


@app.post("/v1/panel/{token}/revoke")
def panel_revoke(token: str) -> dict:
    member = storage.get_member_by_token(token)
    if member is None:
        raise HTTPException(status_code=404, detail="Invalid invitation.")
    deleted = storage.revoke_member(member["id"])
    return {"status": "revoked", "deleted_sessions": deleted}


@app.get("/panel/{token}", include_in_schema=False)
def panel_disclosure(token: str) -> HTMLResponse:
    member = storage.get_member_by_token(token)
    if member is None:
        raise HTTPException(status_code=404, detail="Invalid invitation.")
    html = (_APP_DIR / "panel.html").read_text().replace("__TOKEN__", token)
    return HTMLResponse(html)


# ------------------------------------------------------------------ dev pages


@app.get("/", include_in_schema=False)
def dashboard() -> FileResponse:
    return FileResponse(_APP_DIR / "dashboard.html")


@app.get("/demo", include_in_schema=False)
def demo() -> FileResponse:
    page = _DEMO_DIR / "index.html"
    if not page.exists():
        raise HTTPException(status_code=404, detail="Demo site not found")
    return FileResponse(page)
