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
import logging
import os
import re
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Optional

import json

import openai
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import ValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from . import jobs, storage
from .config import MAX_TWINS_PER_RUN, REPO_ROOT
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

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Caller:
    """Who is asking, and what they are allowed to see.

    `site_id is None` means UNSCOPED — every tenant's data. That is the right
    answer for a single-tenant deployment and for an admin key, and the wrong
    answer for a customer key, so it is represented explicitly rather than as a
    magic string: a run owned by nobody must stay distinguishable from a run
    owned by a tenant that happens to be called "default".
    """

    site_id: Optional[str] = None

    @property
    def scoped(self) -> bool:
        return self.site_id is not None

    def owns(self, site_id: Optional[str]) -> bool:
        return self.site_id is None or self.site_id == site_id


def _key_table() -> dict[str, Optional[str]]:
    """Parse COGNISWARM_API_KEYS into {key: site_id or None}.

    Two accepted forms, so adding tenancy does not break an existing
    deployment on the night it ships:

        COGNISWARM_API_KEYS=k1:acme,k2:globex   # scoped to one tenant each
        COGNISWARM_API_KEYS=k3                  # unscoped (admin/single-tenant)

    A site id cannot contain ':' (it comes from IngestEnvelope.site_id, which
    the SDK sets), so splitting on the first colon is unambiguous.
    """
    table: dict[str, Optional[str]] = {}
    for entry in os.environ.get("COGNISWARM_API_KEYS", "").split(","):
        entry = entry.strip()
        if not entry:
            continue
        key, _, site = entry.partition(":")
        key = key.strip()
        if key:
            table[key] = site.strip() or None
    return table


def require_api_key(x_api_key: Optional[str] = Header(default=None)) -> Caller:
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
    keys = _key_table()
    if keys:
        # Configured keys always win: the anonymous escape hatch below can never
        # weaken a server that has been given keys.
        #
        # constant-time compare — `in` on a dict of strings short-circuits on
        # the first differing byte, which leaks key length and prefix under
        # timing. Every candidate is compared so the work does not depend on
        # which key matched either.
        matched: Optional[str] = None
        for candidate, site in keys.items():
            if x_api_key and hmac.compare_digest(x_api_key, candidate):
                matched = candidate
        if matched is None:
            raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key.")
        return Caller(site_id=keys[matched])

    if os.environ.get("COGNISWARM_ALLOW_ANONYMOUS", "").strip().lower() in {"1", "true", "yes"}:
        # Anonymous is unscoped by construction — there is no key to carry a
        # tenant. Correct for local development and single-tenant servers; it
        # is why the anonymous hatch must never be set on a shared deployment.
        return Caller(site_id=None)

    raise HTTPException(
        status_code=503,
        detail=(
            "Server is not configured for API access: set COGNISWARM_API_KEYS, "
            "or COGNISWARM_ALLOW_ANONYMOUS=1 for local development."
        ),
    )


#: Every authenticated endpoint takes this instead of listing the dependency
#: in `dependencies=[...]`. Taking it as a PARAMETER rather than a side-effect
#: is the point: the tenant scope has to be visible in the signature, or the
#: next endpoint someone adds will authenticate correctly and then query every
#: tenant's data — which is exactly how `site_id` came to be written and never
#: read. FastAPI caches the dependency per request, so this costs nothing.
CallerDep = Annotated[Caller, Depends(require_api_key)]


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
#
# This list is what makes the SDK work on a customer's site, and an empty one
# is why it currently does not. `transport.ts` sends the beacon as a Blob typed
# application/json, which is not a CORS-safelisted content type, so every
# cross-origin POST is preceded by a preflight — and with no configured origins
# that preflight is refused. The failure is completely silent from both ends:
# `navigator.sendBeacon` returns true on *queueing*, so the SDK reports success,
# and the request never reaches a route so the server logs nothing. A correctly
# integrated site drops 100% of its telemetry and nobody finds out.
#
# Set COGNISWARM_ALLOWED_ORIGINS to the customer origins that may post
# telemetry, comma separated, scheme included:
#   COGNISWARM_ALLOWED_ORIGINS=https://acme.com,https://www.acme.com
#
# Origins are matched exactly — scheme, host and port. https://acme.com does
# not cover https://www.acme.com, and neither covers a staging subdomain.
_ALLOWED_ORIGINS = [
    o.strip() for o in os.environ.get("COGNISWARM_ALLOWED_ORIGINS", "").split(",") if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    # Ingest is the only cross-origin surface: the dashboard reaches the
    # analysis API server-side through its own proxy and is never subject to
    # CORS. So the browser only ever needs POST (plus the OPTIONS preflight,
    # which CORSMiddleware answers itself) and only the one header the beacon
    # actually sets. Narrowing these means a stolen origin entry cannot be used
    # to drive the rest of the API from someone's browser.
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)

storage.init_db()

_APP_DIR = Path(__file__).resolve().parent
_DEMO_DIR = REPO_ROOT / "examples" / "demo-site"
_COLLECTOR_DIST = REPO_ROOT / "packages" / "collector" / "dist"

if _COLLECTOR_DIST.exists():
    app.mount("/static/collector", StaticFiles(directory=_COLLECTOR_DIST), name="collector")


# ------------------------------------------------------------------ liveness


@app.get("/healthz", include_in_schema=False)
def healthz() -> dict:
    """Platform health check. Deliberately outside `require_api_key`.

    An orchestrator probes this before the app is reachable by anything else,
    so gating it behind auth means the container is killed as unhealthy while
    working perfectly. It touches the database because a backend that cannot
    reach its volume is not healthy in any useful sense — that is the failure
    this needs to catch, since the mount is the part most likely to be wrong.

    Returns no configuration, no version, no paths: it is a public endpoint and
    the only thing a prober is entitled to know is up or not up.
    """
    try:
        storage.ping()
    except Exception:
        raise HTTPException(status_code=503, detail="Datastore unavailable.")
    return {"status": "ok"}


# ------------------------------------------------------------------ ingestion


@app.post("/v1/ingest", status_code=202)
def ingest(envelope: IngestEnvelope, request: Request) -> dict:
    # Rate limit before anything else: this is the only endpoint that accepts
    # writes without a key, and every accepted segment becomes a row.
    #
    # In the deployed topology the backend sits behind the dashboard's
    # /api/ingest passthrough, so the socket peer is the proxy for every
    # request — X-Forwarded-For is what distinguishes callers, and it is
    # spoofable. That is accepted: this is a backstop against a script, and a
    # real distributed flood is an edge-layer problem. The per-site component
    # of the key still holds regardless, since site_id is in the payload.
    client = request.headers.get("x-forwarded-for", "").split(",")[0].strip() or (
        request.client.host if request.client else "unknown"
    )
    if not _ingest_limiter.allow(envelope.site_id, client):
        raise HTTPException(
            status_code=429,
            detail="Too many telemetry segments; slow down.",
            headers={"Retry-After": str(int(_ingest_limiter.window_s))},
        )

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
        template_path(envelope.page_path),
        envelope.features.model_dump(),
        panel_member_id=panel_member_id,
    )
    return {"status": "accepted", "session_id": session_id}


@app.get("/v1/sessions")
def sessions(caller: CallerDep) -> list[dict]:
    return storage.list_sessions(site_id=caller.site_id)


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


@app.post("/v1/sessions/{session_id}/profile")
def profile(caller: CallerDep, session_id: str) -> dict:
    session = storage.get_session(session_id, site_id=caller.site_id)
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


@app.post("/v1/sessions/{session_id}/profile/stream")
async def profile_stream(session_id: str, caller: CallerDep) -> StreamingResponse:
    """The full profiling pipeline as SSE — every model's computation is
    streamed as it happens so the frontend can show the real work: EM traces,
    BIC comparisons, fitted parameters, then the LLM synthesis stages."""
    session = storage.get_session(session_id, site_id=caller.site_id)
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


class _IngestLimiter:
    """Fixed-window rate limit for the one unauthenticated write endpoint.

    In-process and best-effort, deliberately. The service already runs as a
    single instance (jobs.py cannot be replicated), so a shared store would buy
    nothing today, and the honest framing is that this is a cheap backstop
    against a script — not a defence against a distributed flood, which belongs
    at the edge. It bounds the damage from the amplification paths that remain:
    every accepted segment is a row, and rows are what the persona window and
    the storage volume are made of.

    Keyed per (site, client) so one noisy integration cannot starve another
    tenant's ingestion.
    """

    def __init__(self, limit: int, window_s: float) -> None:
        self.limit = limit
        self.window_s = window_s
        self._hits: dict[tuple[str, str], list[float]] = {}

    def allow(self, site_id: str, client: str) -> bool:
        if self.limit <= 0:
            return True
        now = time.monotonic()
        key = (site_id, client)
        recent = [t for t in self._hits.get(key, ()) if now - t < self.window_s]
        # Bounded memory: a key with nothing recent is dropped rather than kept
        # forever, so a rotating source cannot grow this dict without limit.
        if not recent and key in self._hits:
            del self._hits[key]
        if len(recent) >= self.limit:
            self._hits[key] = recent
            return False
        recent.append(now)
        self._hits[key] = recent
        return True


_ingest_limiter = _IngestLimiter(
    limit=int(os.environ.get("COGNISWARM_INGEST_RATE_LIMIT", "120")),
    window_s=float(os.environ.get("COGNISWARM_INGEST_RATE_WINDOW_S", "60")),
)


_ID_SEGMENT = re.compile(
    r"^(?:"
    r"\d+"                                          # 8817
    r"|[0-9a-f]{8,}"                                # hex ids, uuids without dashes
    r"|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"  # uuid
    r"|[A-Za-z0-9_-]{20,}"                          # long opaque tokens
    r")$",
    re.IGNORECASE,
)


def template_path(path: str) -> str:
    """Replace identifier-shaped path segments with `:id` before storage.

    The SDK sends `location.pathname`, so query strings never arrive — the
    exposure is the PATH, and it routinely carries exactly the identifiers the
    architecture promises never to collect: `/invoice/8817`, `/u/8f3a…`,
    `/reset/<token>`. Storing those raw makes the "no PII by construction"
    claim false through a channel the zone whitelist does not cover, and a
    password-reset token in a telemetry table is a live credential.

    Templating keeps everything the product actually uses the path for —
    grouping sessions by page — while discarding the part that identifies a
    person or grants access.
    """
    parts = path.split("/")
    return "/".join(":id" if _ID_SEGMENT.match(p) else p for p in parts)


def estimate_twin_calls(personas: int, request) -> int:
    """Twin calls one request will dispatch, before it dispatches any.

    The multiplier is whatever the study fans out over — variants for a
    compare, prices for a price curve, steps for a walk, generations for the
    optimizer. Read off the request rather than hardcoded per endpoint so a new
    study cannot quietly skip the ceiling by not being listed here.
    """
    calls = personas * getattr(request, "twins_per_persona", 1)
    for attr in ("variants", "prices", "steps", "messages"):
        fan = getattr(request, attr, None)
        if fan:
            calls *= len(fan)
    # The optimizer's budget is per generation (population is NOT a multiplier —
    # the budget is split across it within a generation).
    calls *= getattr(request, "generations", 1)
    return calls


def _enforce_budget(personas: list[PersonaSeed], request) -> None:
    calls = estimate_twin_calls(len(personas), request)
    if calls > MAX_TWINS_PER_RUN:
        raise HTTPException(
            status_code=413,
            detail=(
                f"This request would dispatch {calls:,} twin calls, over the "
                f"{MAX_TWINS_PER_RUN:,} per-run ceiling. Reduce twins_per_persona, "
                f"narrow the comparison, or select fewer sessions "
                f"({len(personas)} personas are currently in scope). "
                "Raise COGNISWARM_MAX_TWINS_PER_RUN to change the ceiling."
            ),
        )


def _load_personas(session_ids: list[str], caller: Caller) -> list[PersonaSeed]:
    candidates = (
        [storage.get_session(sid, site_id=caller.site_id) for sid in session_ids]
        if session_ids
        # Filtered in SQL, not in Python after a LIMIT — see the note on
        # list_profiled_sessions. Taking the newest 200 rows and *then* looking
        # for personas meant 200 unauthenticated ingests disabled every study
        # endpoint in the product.
        else storage.list_profiled_sessions(site_id=caller.site_id)
    )
    # Hydrated per row, not as a bare comprehension. `persona` is a versionless
    # JSON blob, so ONE row written by an older schema — or by any code path
    # that stored a partial persona — raised ValidationError out of the whole
    # comprehension and took down every swarm, compare, walk and study endpoint
    # at once, for every tenant. A single bad row must cost one persona, not
    # the entire product.
    personas: list[PersonaSeed] = []
    skipped = 0
    for s in candidates:
        if s is None or not s.get("persona"):
            continue
        try:
            personas.append(PersonaSeed(**s["persona"]))
        except ValidationError:
            skipped += 1
            log.warning("session %s has an unreadable persona blob; skipping", s.get("id"))
    if skipped:
        log.warning("%d of %d personas were unreadable", skipped, skipped + len(personas))
    if not personas:
        raise HTTPException(
            status_code=400,
            detail=(
                "No profiled personas available. Profile at least one session first."
                if not skipped
                else f"No readable personas: {skipped} stored persona(s) failed validation."
            ),
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


@app.post("/v1/swarm/run")
async def swarm_run(caller: CallerDep, request: SwarmRunRequest) -> dict:
    personas = _load_personas(request.session_ids, caller)
    _enforce_budget(personas, request)
    try:
        aggregate = await run_swarm(
            personas=personas,
            scenario=request.scenario,
            twins_per_persona=request.twins_per_persona,
            cognitive_load=request.cognitive_load,
        )
    except (openai.OpenAIError, RuntimeError, ValueError) as exc:
        raise _llm_errors(exc) from exc
    run_id = storage.insert_swarm_run(
        request.model_dump(), aggregate.model_dump(), site_id=caller.site_id
    )
    result = aggregate.model_dump()
    result["run_id"] = run_id
    return result


def _sse(generator, request_model, kind: str, caller: Caller) -> StreamingResponse:
    """Wrap a simulation's async event generator as an SSE response, persisting
    the run when the terminal 'done' event passes through.

    `caller` is required rather than optional: every streaming study persists a
    run through here, so a default would silently write unowned rows that no
    scoped tenant can ever read back.
    """

    async def gen():
        try:
            async for evt in generator:
                if evt.get("type") == "done":
                    run_id = storage.insert_swarm_run(
                        request_model.model_dump(),
                        evt["result"],
                        kind=kind,
                        site_id=caller.site_id,
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


@app.post("/v1/swarm/stream")
async def swarm_stream(caller: CallerDep, request: SwarmRunRequest) -> StreamingResponse:
    personas = _load_personas(request.session_ids, caller)
    _enforce_budget(personas, request)
    return _sse(
        stream_swarm(
            personas, request.scenario, request.twins_per_persona, request.cognitive_load
        ),
        request,
        "swarm",
        caller,
    )


@app.post("/v1/swarm/compare/stream")
async def compare_stream(caller: CallerDep, request: CompareRequest) -> StreamingResponse:
    names = [v.name for v in request.variants]
    if len(set(names)) != len(names):
        raise HTTPException(status_code=400, detail="Variant names must be unique.")
    personas = _load_personas(request.session_ids, caller)
    _enforce_budget(personas, request)
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
        caller,
    )


@app.post("/v1/swarm/walk/stream")
async def walk_stream(caller: CallerDep, request: WalkRequest) -> StreamingResponse:
    personas = _load_personas(request.session_ids, caller)
    _enforce_budget(personas, request)
    return _sse(
        stream_walkthrough(
            personas, request.steps, request.twins_per_persona, request.cognitive_load
        ),
        request,
        "walk",
        caller,
    )


@app.post("/v1/studies/price/stream")
async def price_stream(caller: CallerDep, request: PriceSensitivityRequest) -> StreamingResponse:
    personas = _load_personas(request.session_ids, caller)
    _enforce_budget(personas, request)
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
        caller,
    )


@app.post("/v1/studies/objection/stream")
async def objection_stream(caller: CallerDep, request: ObjectionRequest) -> StreamingResponse:
    personas = _load_personas(request.session_ids, caller)
    _enforce_budget(personas, request)
    return _sse(
        stream_objection_scan(
            personas, request.pitch, request.twins_per_persona, request.cognitive_load
        ),
        request,
        "objection",
        caller,
    )


@app.post("/v1/studies/virality/stream")
async def virality_stream(caller: CallerDep, request: ViralityRequest) -> StreamingResponse:
    """Virality forecast: Galton-Watson branching process over twin share
    intents — R0, extinction probability, seeded cascade quantile bands."""
    personas = _load_personas(request.session_ids, caller)
    _enforce_budget(personas, request)
    return _sse(stream_virality(personas, request), request, "virality", caller)


@app.post("/v1/studies/content/stream")
async def content_stream(caller: CallerDep, request: ContentStudyRequest) -> StreamingResponse:
    """Neuro-impact study: beat-by-beat audience response with ISC,
    change-point, peak-end memory, and functional-system mapping."""
    personas = _load_personas(request.session_ids, caller)
    _enforce_budget(personas, request)
    return _sse(stream_content_study(personas, request), request, "content", caller)


@app.post("/v1/studies/optimize/stream")
async def optimize_stream(caller: CallerDep, request: CopyOptimizerRequest) -> StreamingResponse:
    """Copy optimiser: an evolutionary loop that WRITES better copy — twin-scored
    populations, Thompson-allocated budget within each generation, LLM crossover
    and mutation between them, and a win claimed only when the champion's
    credible interval clears the seed's."""
    personas = _load_personas(request.session_ids, caller)
    _enforce_budget(personas, request)
    return _sse(stream_copy_optimizer(personas, request), request, "optimize", caller)


@app.post("/v1/studies/sequence/stream")
async def sequence_stream(caller: CallerDep, request: SequenceRequest) -> StreamingResponse:
    """Message-sequence optimiser: estimates a position-adjusted precedence
    matrix from a sampled subset of the N! orderings, then solves the linear
    ordering problem heuristically for the ordering that ends with the most
    intent — reporting primacy/recency separately from message strength."""
    personas = _load_personas(request.session_ids, caller)
    _enforce_budget(personas, request)
    return _sse(stream_sequence(personas, request), request, "sequence", caller)


@app.post("/v1/swarm/compare")
async def swarm_compare(caller: CallerDep, request: CompareRequest) -> dict:
    names = [v.name for v in request.variants]
    if len(set(names)) != len(names):
        raise HTTPException(status_code=400, detail="Variant names must be unique.")
    personas = _load_personas(request.session_ids, caller)
    _enforce_budget(personas, request)
    try:
        compared = await run_compare(
            personas=personas,
            variants=request.variants,
            twins_per_persona=request.twins_per_persona,
            cognitive_load=request.cognitive_load,
        )
    except (openai.OpenAIError, RuntimeError, ValueError) as exc:
        raise _llm_errors(exc) from exc
    run_id = storage.insert_swarm_run(
        request.model_dump(), compared.model_dump(), kind="compare", site_id=caller.site_id
    )
    result = compared.model_dump()
    result["run_id"] = run_id
    return result


@app.post("/v1/swarm/walk")
async def swarm_walk(caller: CallerDep, request: WalkRequest) -> dict:
    personas = _load_personas(request.session_ids, caller)
    _enforce_budget(personas, request)
    try:
        aggregate = await run_walkthrough(
            personas=personas,
            steps=request.steps,
            twins_per_persona=request.twins_per_persona,
            cognitive_load=request.cognitive_load,
        )
    except (openai.OpenAIError, RuntimeError, ValueError) as exc:
        raise _llm_errors(exc) from exc
    run_id = storage.insert_swarm_run(
        request.model_dump(), aggregate.model_dump(), kind="walk", site_id=caller.site_id
    )
    result = aggregate.model_dump()
    result["run_id"] = run_id
    return result


@app.get("/v1/swarm/runs")
def swarm_runs(caller: CallerDep, kind: Optional[str] = None) -> list[dict]:
    return storage.list_swarm_runs(kind=kind, site_id=caller.site_id)


# ------------------------------------------------------------------ studies (advanced use cases)


@app.post("/v1/studies/price")
async def study_price(caller: CallerDep, request: PriceSensitivityRequest) -> dict:
    personas = _load_personas(request.session_ids, caller)
    _enforce_budget(personas, request)
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
    run_id = storage.insert_swarm_run(
        request.model_dump(), result.model_dump(), kind="price", site_id=caller.site_id
    )
    out = result.model_dump()
    out["run_id"] = run_id
    return out


@app.post("/v1/studies/objection")
async def study_objection(caller: CallerDep, request: ObjectionRequest) -> dict:
    personas = _load_personas(request.session_ids, caller)
    _enforce_budget(personas, request)
    try:
        result = await run_objection_scan(
            personas=personas,
            pitch=request.pitch,
            twins_per_persona=request.twins_per_persona,
            cognitive_load=request.cognitive_load,
        )
    except (openai.OpenAIError, RuntimeError, ValueError) as exc:
        raise _llm_errors(exc) from exc
    run_id = storage.insert_swarm_run(
        request.model_dump(), result.model_dump(), kind="objection", site_id=caller.site_id
    )
    out = result.model_dump()
    out["run_id"] = run_id
    return out


# ------------------------------------------------------------------ validation


@app.post("/v1/runs/{run_id}/actuals")
def record_actuals(caller: CallerDep, run_id: str, payload: ActualsPayload) -> dict:
    if payload.engagement is None and payload.intent is None:
        raise HTTPException(status_code=400, detail="Provide at least one metric.")
    if not storage.set_actuals(
        run_id, payload.model_dump(exclude_none=True), site_id=caller.site_id
    ):
        raise HTTPException(status_code=404, detail="Unknown run")
    return {"status": "recorded", "run_id": run_id}


@app.get("/v1/validation/report")
def validation_report(caller: CallerDep) -> CalibrationReport:
    return calibration_report(
        storage.list_swarm_runs(limit=500, kind="swarm", site_id=caller.site_id)
    )


# ------------------------------------------------------------------ jobs (Phase 3)


@app.post("/v1/jobs", status_code=202)
def create_job(request: JobCreate, caller: CallerDep) -> dict:
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
    job_id = jobs.enqueue(request.kind, payload, site_id=caller.site_id)
    return {"job_id": job_id, "status": "queued"}


@app.get("/v1/jobs")
def list_jobs(caller: CallerDep) -> list[dict]:
    return storage.list_jobs(site_id=caller.site_id)


@app.get("/v1/jobs/{job_id}")
def get_job(caller: CallerDep, job_id: str) -> dict:
    job = storage.get_job(job_id, site_id=caller.site_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job")
    return job


# ------------------------------------------------------------------ panel (Phase 4)


@app.post("/v1/panel/members")
def create_panel_member(caller: CallerDep, request: PanelMemberCreate) -> dict:
    """Provision an invite.

    This is the ONE response that carries the capability URL, because it is the
    one moment it is needed: the admin has to send it to the member. It is not
    retrievable afterwards — see the listing below.
    """
    member = storage.insert_panel_member(request.label)
    member["disclosure_url"] = f"/panel/{member['token']}"
    return member


@app.get("/v1/panel/members")
def panel_members(caller: CallerDep) -> list[dict]:
    """List members WITHOUT their capability tokens.

    The previous version stripped `token` from the dict and then rebuilt the
    full capability URL from that same token one expression later, under a
    comment saying tokens must not leak in listings. So the listing handed out
    every member's disclosure URL — and that URL is the capability: anyone
    holding it can consent or revoke on that member's behalf.

    A token is a bearer secret, so it is shown once at creation and never
    again. `token_hint` is the last four characters, which is enough for an
    admin to match a row against an invite they already sent and not enough to
    reconstruct anything.
    """
    return [
        {k: v for k, v in m.items() if k != "token"} | {"token_hint": m["token"][-4:]}
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
