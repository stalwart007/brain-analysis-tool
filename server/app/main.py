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
import base64
import hmac
import logging
import os
import re
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Optional
from urllib.parse import urlparse

import json
from html import escape as html_escape

import openai
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import ValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from . import jobs, storage
from .config import MAX_TWINS_PER_RUN, REPO_ROOT
from .audience import compose_audience, infer_audience, provenance_note
from .boardroom import (
    cast_diversity,
    cast_provenance,
    cast_room,
    room_call_estimate,
    stream_deliberation,
)
from .fetching import ASSET_KINDS, FetchFailed, UnsafeURL, fetch_url, probe_url
from .findings import build_findings
from .providers import describe_link
from .modality import (
    UnsupportedAsset,
    is_player_url,
    pages_from_pdf,
    preview_page,
    visible_text_length,
)
from .youtube import (
    LADDER,
    YouTubeUnavailable,
    choose_rung,
    fetch_captions,
    fetch_manifest,
    is_youtube_url,
    manifest_envelope,
    parse_video_id,
    pick_caption_track,
    plan_keyframes,
)
from .oai import Refusal
from .persona import seed_persona
from .cognition import full_cognitive_profile, iter_cognitive_profile
from .profiler import profile_features
from .schemas import (
    ActualsPayload,
    AudienceComposeRequest,
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
    ContentAsset,
    ContentStudyRequest,
    CopyOptimizerRequest,
    PageFetchRequest,
    ObjectionRequest,
    PriceSensitivityRequest,
    RoomRequest,
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


#: Where each request keeps the thing the audience is reacting to. Ordered by
#: specificity so a request carrying several fields yields the most
#: representative one. Read off the request rather than switched on per
#: endpoint, so a study added later inherits audience inference for free.
_STIMULUS_FIELDS = (
    "scenario", "content", "pitch", "brief", "seed_variant", "product", "motion"
)


def stimulus_of(request) -> str:
    """The text an inferred audience would be reacting to, or ''."""
    for field in _STIMULUS_FIELDS:
        v = getattr(request, field, None)
        if isinstance(v, str) and v.strip():
            return v.strip()
    # Multi-part stimuli: variants, walk steps, sequence messages.
    for field in ("variants", "steps", "messages"):
        v = getattr(request, field, None)
        if v:
            parts = [
                getattr(x, "scenario", None) or (x if isinstance(x, str) else None)
                for x in v
            ]
            joined = "\n\n".join(p for p in parts if p)
            if joined.strip():
                return joined.strip()
    asset = getattr(request, "asset", None)
    if asset is not None:
        for field in ("text", "brief"):
            v = getattr(asset, field, None)
            if isinstance(v, str) and v.strip():
                return v.strip()
    return ""


def estimate_twin_calls(personas: int, request) -> int:
    """Twin calls one request will dispatch, before it dispatches any.

    The multiplier is whatever the study fans out over — variants for a
    compare, prices for a price curve, steps for a walk, generations for the
    optimizer. Read off the request rather than hardcoded per endpoint so a new
    study cannot quietly skip the ceiling by not being listed here.
    """
    # The white room is the one study that does NOT fan out over the persona
    # window: it seats a fixed cast, every member speaks once per round, and the
    # whole schedule is replicated. `personas` is not its multiplier, so the
    # generic path below would score a 972-call room as one call per persona and
    # wave it past the ceiling. The cost model lives next to the engine that
    # spends it — see boardroom.room_call_estimate.
    if isinstance(request, RoomRequest):
        return room_call_estimate(request)
    calls = personas * getattr(request, "twins_per_persona", 1)
    for attr in ("variants", "prices", "steps", "messages"):
        fan = getattr(request, attr, None)
        if fan:
            calls *= len(fan)
    # The optimizer's budget is per generation (population is NOT a multiplier —
    # the budget is split across it within a generation).
    calls *= getattr(request, "generations", 1)
    return calls


def _enforce_budget(
    personas: list[PersonaSeed], request, how_to_reduce: Optional[str] = None
) -> None:
    """413 when a request would dispatch more model calls than the ceiling.

    `how_to_reduce` replaces the persona-window advice for studies that do not
    have one. The white room's cost comes from seats, rounds and replicates, and
    telling its caller to "select fewer sessions (0 personas are currently in
    scope)" would send them to a control that has no effect on the number in the
    same sentence.
    """
    calls = estimate_twin_calls(len(personas), request)
    if calls > MAX_TWINS_PER_RUN:
        raise HTTPException(
            status_code=413,
            detail=(
                f"This request would dispatch {calls:,} twin calls, over the "
                f"{MAX_TWINS_PER_RUN:,} per-run ceiling. "
                + (
                    how_to_reduce
                    or (
                        "Reduce twins_per_persona, narrow the comparison, or "
                        "select fewer sessions "
                        f"({len(personas)} personas are currently in scope). "
                    )
                )
                + "Raise COGNISWARM_MAX_TWINS_PER_RUN to change the ceiling."
            ),
        )


async def _load_personas(
    session_ids: list[str],
    caller: Caller,
    stimulus: str = "",
    supplied: Optional[list[PersonaSeed]] = None,
) -> list[PersonaSeed]:
    # A panel the researcher curated wins over everything. They have already
    # seen it and shaped it; reconstructing an audience per run would silently
    # replace the thing they approved, and two runs of "the same" study would
    # not be comparable.
    if supplied:
        return supplied
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
    if personas:
        return personas

    # COLD START. No observed telemetry — infer the audience from the stimulus
    # rather than refusing. Without this a new account can do nothing at all
    # until a collector has been installed and has accumulated traffic, which
    # is every prospective customer on day one.
    #
    # Never a blend: observed personas above return immediately, so a run is
    # entirely measured or entirely inferred. Mixing them would make the
    # aggregate uninterpretable, because no reader could tell which half moved
    # the number. The provenance rides along on every persona and is reported
    # with every result.
    if stimulus and not skipped:
        try:
            inferred = await infer_audience(stimulus)
            if inferred:
                log.info("no observed personas; inferred %d from stimulus", len(inferred))
                return inferred
        except (openai.OpenAIError, Refusal, RuntimeError, ValueError) as exc:
            # Fall through to the original error. The user asked for a study,
            # not for inference, so "no personas" is the message they can act
            # on — a failure inside a fallback they never requested would only
            # confuse.
            log.warning("audience inference failed: %s", exc)

    raise HTTPException(
        status_code=400,
        detail=(
            "No profiled personas available, and the audience could not be "
            "inferred from this request. Profile at least one session, or "
            "include a scenario or brief to infer an audience from."
            if not skipped
            else f"No readable personas: {skipped} stored persona(s) failed validation."
        ),
    )


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
    personas = await _load_personas(
        request.session_ids, caller, stimulus_of(request),
        supplied=getattr(request, "personas", None),
    )
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
    """Wrap a simulation's async event generator as an SSE response: run the
    findings synthesis over the terminal result, persist, and emit.

    `caller` is required rather than optional: every streaming study persists a
    run through here, so a default would silently write unowned rows that no
    scoped tenant can ever read back.

    THE FINDINGS PASS LIVES HERE, once, rather than inside each of the nine
    study modules. Every stream endpoint already funnels its terminal event
    through this function, so this is the single seam where "the study
    finished" becomes "the study has been interpreted" — and putting it in one
    place is what makes it impossible for a study added later to quietly ship
    without the synthesis, the FDR control, or the citation validation.

    It runs BEFORE persistence so the stored run carries its findings and the
    history view does not have to recompute them, and it is streamed as its own
    `findings` stage first so the UI can show that the last (and slowest) step
    is analysis rather than a hang.
    """

    async def gen():
        try:
            async for evt in generator:
                if evt.get("type") == "done":
                    yield f"data: {json.dumps({'type': 'stage', 'stage': 'findings', 'detail': 'harvesting evidence, controlling for multiplicity, synthesising'})}\n\n"
                    # Never fatal. A completed study is worth returning even if
                    # the interpretation on top of it failed — `build_findings`
                    # already returns its own error field rather than raising,
                    # and this guard covers the unexpected remainder.
                    try:
                        evt["result"]["findings"] = await build_findings(
                            kind,
                            evt["result"],
                            question=getattr(request_model, "research_question", ""),
                            stimulus=stimulus_of(request_model),
                        )
                    except Exception:  # noqa: BLE001 - never lose a finished run
                        log.exception("findings synthesis failed for kind=%s", kind)
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
    personas = await _load_personas(
        request.session_ids, caller, stimulus_of(request),
        supplied=getattr(request, "personas", None),
    )
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
    personas = await _load_personas(
        request.session_ids, caller, stimulus_of(request),
        supplied=getattr(request, "personas", None),
    )
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
    personas = await _load_personas(
        request.session_ids, caller, stimulus_of(request),
        supplied=getattr(request, "personas", None),
    )
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
    personas = await _load_personas(
        request.session_ids, caller, stimulus_of(request),
        supplied=getattr(request, "personas", None),
    )
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
    personas = await _load_personas(
        request.session_ids, caller, stimulus_of(request),
        supplied=getattr(request, "personas", None),
    )
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
    personas = await _load_personas(
        request.session_ids, caller, stimulus_of(request),
        supplied=getattr(request, "personas", None),
    )
    _enforce_budget(personas, request)
    return _sse(stream_virality(personas, request), request, "virality", caller)


@app.post("/v1/studies/content/stream")
async def content_stream(caller: CallerDep, request: ContentStudyRequest) -> StreamingResponse:
    """Neuro-impact study: beat-by-beat audience response with ISC,
    change-point, peak-end memory, and functional-system mapping."""
    personas = await _load_personas(
        request.session_ids, caller, stimulus_of(request),
        supplied=getattr(request, "personas", None),
    )
    _enforce_budget(personas, request)
    return _sse(stream_content_study(personas, request), request, "content", caller)


@app.post("/v1/studies/optimize/stream")
async def optimize_stream(caller: CallerDep, request: CopyOptimizerRequest) -> StreamingResponse:
    """Copy optimiser: an evolutionary loop that WRITES better copy — twin-scored
    populations, Thompson-allocated budget within each generation, LLM crossover
    and mutation between them, and a win claimed only when the champion's
    credible interval clears the seed's."""
    personas = await _load_personas(
        request.session_ids, caller, stimulus_of(request),
        supplied=getattr(request, "personas", None),
    )
    _enforce_budget(personas, request)
    return _sse(stream_copy_optimizer(personas, request), request, "optimize", caller)


@app.post("/v1/studies/sequence/stream")
async def sequence_stream(caller: CallerDep, request: SequenceRequest) -> StreamingResponse:
    """Message-sequence optimiser: estimates a position-adjusted precedence
    matrix from a sampled subset of the N! orderings, then solves the linear
    ordering problem heuristically for the ordering that ends with the most
    intent — reporting primacy/recency separately from message strength."""
    personas = await _load_personas(
        request.session_ids, caller, stimulus_of(request),
        supplied=getattr(request, "personas", None),
    )
    _enforce_budget(personas, request)
    return _sse(stream_sequence(personas, request), request, "sequence", caller)


@app.post("/v1/swarm/compare")
async def swarm_compare(caller: CallerDep, request: CompareRequest) -> dict:
    names = [v.name for v in request.variants]
    if len(set(names)) != len(names):
        raise HTTPException(status_code=400, detail="Variant names must be unique.")
    personas = await _load_personas(
        request.session_ids, caller, stimulus_of(request),
        supplied=getattr(request, "personas", None),
    )
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
    personas = await _load_personas(
        request.session_ids, caller, stimulus_of(request),
        supplied=getattr(request, "personas", None),
    )
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
    personas = await _load_personas(
        request.session_ids, caller, stimulus_of(request),
        supplied=getattr(request, "personas", None),
    )
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
    personas = await _load_personas(
        request.session_ids, caller, stimulus_of(request),
        supplied=getattr(request, "personas", None),
    )
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


# ------------------------------------------------------------------ audience


@app.post("/v1/audience/compose")
async def audience_compose(caller: CallerDep, request: AudienceComposeRequest) -> dict:
    """Build or refine an audience panel from a plain-language instruction.

    Returns the FULL resulting panel, not a diff — a diff would need merge
    rules, and every one of those is a place for the panel the researcher sees
    to drift from the panel that actually runs.

    Everything returned is `inferred`, including segments the researcher asked
    for by name: asserting a segment exists is a stated assumption, not
    evidence, and marking it observed because a human typed it is exactly the
    confusion provenance exists to prevent.
    """
    try:
        if request.instruction.strip():
            personas = await compose_audience(
                request.stimulus, request.instruction, request.existing
            )
        else:
            personas = await infer_audience(request.stimulus)
    except (openai.OpenAIError, Refusal, RuntimeError, ValueError) as exc:
        raise _llm_errors(exc) from exc
    return {
        "personas": [p.model_dump() for p in personas],
        "provenance": provenance_note(personas),
    }


# ------------------------------------------------------------------ white room


def _room_guard(request: RoomRequest) -> None:
    """What a room needs before it is worth paying for.

    Both checks are 400s at the door rather than errors inside the stream. An
    SSE generator can only report a problem as an `error` frame after the
    headers are sent, which the client cannot distinguish from a run that failed
    halfway — and these are both properties of the request, knowable before a
    single call is dispatched.
    """
    if request.members:
        if len(request.members) < 3:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"A room needs at least three characters; {len(request.members)} "
                    "were supplied. With fewer there is no deliberation to measure — "
                    "one member has nobody to hear and two is a negotiation."
                ),
            )
        names = [m.name.strip().casefold() for m in request.members]
        if len(set(names)) != len(names):
            raise HTTPException(
                status_code=400,
                detail=(
                    "Member names must be unique. Names are how statements are "
                    "attributed at the table and what `conceded_to` refers to, so "
                    "two members with one name make every concession ambiguous."
                ),
            )


@app.post("/v1/room/cast")
async def room_cast(caller: CallerDep, request: RoomRequest) -> dict:
    """Instantiate the characters, and return them for approval before a run.

    Deliberately a separate call from `/v1/room/stream`, for the reason
    `/v1/content/fetch` is separate from running a study: the researcher sees
    what was actually built — the voices, the stakes, the declared dispositions
    — BEFORE spending the fan-out on it. A room that turned out to be five
    variations on one person is worse than a refusal, because it produces
    numbers.

    Takes the full `RoomRequest` so the dashboard can post one body to both
    endpoints; only `motion`, `cast_brief` and `seats` are read here. The
    returned members go back in `request.members` to run the cast that was
    approved rather than a fresh one cast per run.

    Two labels ride along and neither is decoration. `provenance` says whether
    these characters came from the researcher's brief or were inferred from the
    motion, because nobody in this room is observed and the result must not be
    read as though somebody were. `diversity` says whether the cast can disagree
    at all — surfaced, never refused, since a room that cannot disagree is a
    legitimate thing to study as long as nobody mistakes its consensus for one.
    """
    try:
        members = await cast_room(request.motion, request.cast_brief, request.seats)
    except (openai.OpenAIError, Refusal, RuntimeError, ValueError) as exc:
        raise _llm_errors(exc) from exc
    return {
        "members": [m.model_dump() for m in members],
        "provenance": cast_provenance(request.cast_brief, members),
        "diversity": cast_diversity(members),
        # What running this cast would cost, and against what — so the approval
        # step is also the budget step, before the 413 can surprise anyone.
        "estimated_calls": room_call_estimate(request),
        "call_ceiling": MAX_TWINS_PER_RUN,
    }


@app.post("/v1/room/stream")
async def room_stream(caller: CallerDep, request: RoomRequest) -> StreamingResponse:
    """Put the motion to the room and stream the table live.

    No `_load_personas` call, and that absence is the design rather than an
    omission: a room is a cast of characters, not a sample of an audience. There
    is no persona window to draw from, nothing here is derived from telemetry,
    and blending the two would produce an aggregate over people who were
    measured and people who were invented with no way to tell which half moved
    it — the exact confusion `audience.provenance_note` exists to prevent.

    The private half of every turn is withheld from the live stream and appears
    only in the terminal result; `boardroom.stream_deliberation` documents why.
    """
    _room_guard(request)
    _enforce_budget(
        [],
        request,
        how_to_reduce=(
            f"A room costs seats × (1 + rounds) × (replicates + placebo_replicates) "
            f"= {len(request.members) or request.seats} × (1 + {request.rounds}) × "
            f"({request.replicates} + {request.placebo_replicates}). Seat fewer "
            "members, run fewer rounds, or run fewer replicates — but keep at "
            "least one placebo replicate, because without it a convergence "
            "cannot be told from the model being agreeable. "
        ),
    )
    return _sse(stream_deliberation(request), request, "room", caller)


# ------------------------------------------------------------------ page fetch


async def _metadata_rung(url: str, html: str, envelope: dict) -> dict:
    """The floor under every HTML URL: study how the thing presents itself.

    Reached in two cases that used to be 422s — a player page (whose prose is
    furniture) and a page that renders its copy in the browser (whose prose is
    absent). Both are extremely common and neither is a reason to hand back
    nothing.

    What comes out is a REAL study of a REAL asset, and a narrower one than the
    caller asked for. When the page publishes a hero image, that image is
    fetched and studied for visual hierarchy, with the title and description as
    the brief — which is exactly the study you want for "does this thumbnail
    earn a click". When there is no image but there is copy, the copy is
    studied as text. `rung` carries the distinction upward so the UI can say
    which one happened, because a study of a listing presented as a study of
    the video would be the most misleading thing this endpoint could do.
    """
    meta = await describe_link(url, html)
    if meta is None or not meta.usable:
        # Genuinely nothing. Report the measurement rather than a theory about
        # why — the same discipline the prose path already follows.
        visible = visible_text_length(html)
        raise HTTPException(
            status_code=422,
            detail=(
                f"We reached that page and found {visible:,} characters of readable "
                "text, no sections long enough to read as beats, and no preview "
                "image or description in its markup — so there is nothing to build "
                "a study from. Paste the copy as a script, or upload a screenshot "
                "and study it as an image."
            ),
        )

    label = meta.site_name or urlparse(url).hostname or "that page"
    shared = {
        **envelope,
        "rung": "metadata",
        "meta": {
            "title": meta.title,
            "author": meta.author,
            "description": meta.description,
            "site_name": meta.site_name,
            "og_type": meta.og_type,
            "duration_s": meta.duration_s,
            "sources": meta.sources,
            "image_url": meta.image_url,
        },
    }

    # ── the hero image, when there is one ───────────────────────────────
    if meta.image_url:
        try:
            hero = await fetch_url(meta.image_url, kinds=("image",))
        except (UnsafeURL, FetchFailed):
            hero = None
        if hero is not None and hero.bytes_read > 0:
            return {
                **shared,
                "kind": "image",
                "asset": {
                    "kind": "image",
                    "image_b64": base64.b64encode(hero.content).decode("ascii"),
                    "media_type": hero.content_type,
                    "brief": meta.as_brief()[:1000],
                },
                "note": (
                    f"{label} does not publish readable page copy, so this studies "
                    f"its PREVIEW — the image and headline a scroller actually sees "
                    f"before deciding to click, not the content behind it. "
                    f"Read from {', '.join(meta.sources)}."
                ),
            }

    # ── the listing copy, when there is no usable image ─────────────────
    return {
        **shared,
        "kind": "text",
        "asset": {"kind": "text", "text": meta.as_brief()},
        "note": (
            f"{label} publishes no readable page copy and no usable preview image, "
            f"so this studies its DESCRIPTION — the {len(meta.description)} "
            f"characters it uses to sell itself, not the content behind it. "
            f"Read from {', '.join(meta.sources)}."
        ),
    }


@app.post("/v1/content/fetch")
async def content_fetch(caller: CallerDep, request: PageFetchRequest) -> dict:
    """Turn a pasted URL into whatever kind of study asset it actually is.

    This used to accept only landing pages: anything whose content type was not
    HTML was refused with "upload the file instead", so a link to an ad image,
    a hosted MP4, a podcast episode or a PDF deck — most of the things a
    researcher has a URL for — could not be studied from a link at all. The
    modality layer downstream has always handled all five; only the front door
    was narrow.

    ROUTING IS BY CONTENT TYPE, NEVER BY EXTENSION. `…/promo.mp4` served as
    `text/html` is a player page and gets studied as one; `…/x?id=9` served as
    `image/png` is an image. The bytes are what the study consumes, so the
    bytes' declared type is what decides.

    Still a SEPARATE step from running the study rather than a `url` field on
    the asset. Three reasons, in order of how much they matter:

    1. The researcher sees what was actually retrieved — the final URL after
       redirects, the kind, the sections or pages found — BEFORE spending twin
       calls on it. A study that silently ran against a cookie wall is worse
       than one that refused, because it produces numbers.
    2. `ContentAsset` keeps its no-URL guarantee, so no other endpoint can grow
       a server-side fetch without coming through here.
    3. The failure modes are completely different. "That host is not reachable"
       and "the model refused" want different messages, and folding them into
       one call means reporting a network problem as an analysis problem.
    """
    # ── YouTube: read properly, rather than refused ─────────────────────
    #
    # This used to be a flat 422 alongside every other player host, on the
    # reasoning that fetching youtube.com/watch returns an HTML shell whose
    # prose is the title and the comment policy. That observation is correct
    # and the conclusion drawn from it was not: the shell is what the PAGE
    # fetcher gets, not what a server can get. The InnerTube player endpoint
    # gives duration, chapters, the description and the caption list, and the
    # scrub-bar storyboard gives real frames every two seconds. See
    # `app.youtube` for what is and is not touched, and why.
    if is_youtube_url(request.url):
        video_id = parse_video_id(request.url)
        if not video_id:
            raise HTTPException(
                status_code=422,
                detail=(
                    "that is a YouTube link but not to a single video — channel "
                    "pages, playlists and search results have no content to "
                    "study. Paste a link to one video."
                ),
            )
        try:
            manifest = await fetch_manifest(video_id)
        except YouTubeUnavailable as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except (UnsafeURL, FetchFailed) as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

        track = pick_caption_track(manifest.caption_tracks)
        # Best-effort and deliberately not awaited behind a failure path: an
        # absent transcript costs the study its speech channel, not its life.
        cues = await fetch_captions(track) if track else []
        frames = plan_keyframes(manifest)
        rung = choose_rung(manifest, cues, frames)
        if rung == "none":
            raise HTTPException(
                status_code=422,
                detail=(
                    f"'{manifest.title}' has no filmstrip, no readable transcript "
                    "and no chapters, so there is nothing to build beats from. "
                    "Download the video and upload it — keyframes are extracted "
                    "in your browser."
                ),
            )

        # ── the bottom rung: the thumbnail ──────────────────────────────
        #
        # Reached whenever InnerTube refused us and only the public preview
        # survived — which, measured from Fly, is the COMMON case rather than an
        # edge one. Answered as an ordinary image study of the thumbnail rather
        # than as a video study with nothing in it, because that is what it
        # actually is, and returning `kind: video` here would fail downstream on
        # "a video asset needs keyframes" after the researcher had been told the
        # link was read successfully.
        if rung == "metadata":
            envelope = manifest_envelope(manifest, cues, frames)
            try:
                hero = await fetch_url(manifest.thumbnail_url, kinds=("image",))
            except (UnsafeURL, FetchFailed) as exc:
                raise HTTPException(
                    status_code=502,
                    detail=(
                        f"YouTube would not describe '{manifest.title}' to this "
                        f"server and its thumbnail could not be fetched either "
                        f"({exc}). Download the video and upload it instead."
                    ),
                ) from exc
            return {
                "kind": "image",
                "final_url": manifest.watch_url,
                "hops": [request.url],
                "bytes": hero.bytes_read,
                "content_type": hero.content_type,
                "rung": "metadata",
                "youtube": envelope,
                "asset": {
                    "kind": "image",
                    "image_b64": base64.b64encode(hero.content).decode("ascii"),
                    "media_type": hero.content_type,
                    "brief": (
                        f'YouTube thumbnail for "{manifest.title}"'
                        + (f" by {manifest.author}" if manifest.author else "")
                        + ". The video itself could not be read."
                    ),
                },
                "note": envelope["note"],
            }

        return {
            # Declared as a video whatever rung it lands on, so the client's
            # kind check passes and the ladder is reported in `youtube.rung`
            # rather than by silently changing what the caller asked for.
            "kind": "video",
            "final_url": manifest.watch_url,
            "hops": [request.url],
            "bytes": 0,
            "content_type": "application/x-youtube-manifest",
            "youtube": manifest_envelope(manifest, cues, frames),
            "asset": {"kind": "video"},
            "note": manifest_envelope(manifest, cues, frames)["note"],
        }

    # Other player pages are NO LONGER refused here.
    #
    # The original refusal rested on a correct observation — the visible prose
    # of a Vimeo or TikTok page is furniture, and a "landing page" study of it
    # would describe a nav bar — and then drew a wrong conclusion from it,
    # that there was therefore nothing to study. There is: the page describes
    # itself in its own markup, in OpenGraph tags that exist precisely so a
    # link unfurls, and the hero image those tags point at is a real asset.
    #
    # So the prose path is SKIPPED for these hosts (the original insight is
    # kept) and they fall through to the metadata rung below, which studies the
    # listing — the thumbnail and the copy that decide whether anyone clicks.
    # That is a different question from what the video does, and it is labelled
    # as a different question everywhere it surfaces.
    skip_prose = is_player_url(request.url) is not None

    # Identify before downloading. A hosted video is the common case here and
    # the ONLY thing this call needs to know about one is that it is a video —
    # a fact the response headers carry. Reading the body first pulled up to
    # eighty megabytes into memory so that the relay could pull the same eighty
    # megabytes again a moment later.
    try:
        probed = await probe_url(request.url)
    except UnsafeURL as exc:
        # 400, not 403: the caller is authenticated and permitted, the URL is
        # the thing that is wrong, and they can fix it by pasting another one.
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FetchFailed as exc:
        # 502: we are reporting someone else's failure, not our own.
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    if probed.kind in {"video", "audio"}:
        # ── video / audio: relayed to the BROWSER, decoded there ────────
        #
        # The server never decodes this. Running a media decoder over bytes an
        # arbitrary URL supplied is a large and historically hostile attack
        # surface, and the browser already has a decoder plus a sandbox around
        # it. So the probe proves the URL is safe and reachable, and the actual
        # bytes go to the client through the relay below, which re-validates.
        declared = probed.declared_length
        cap = ASSET_KINDS[probed.kind][1]
        if declared is not None and declared > cap:
            # Caught here rather than after the download, so the refusal costs
            # one round trip instead of eighty megabytes of transfer that ends
            # in the same refusal.
            raise HTTPException(
                status_code=422,
                detail=(
                    f"that file declares {declared / 1_000_000:.0f} MB, over the "
                    f"{cap // 1_000_000} MB ceiling for {probed.kind}. Trim it, or "
                    "upload a shorter cut — keyframes are extracted in your browser."
                ),
            )
        return {
            "kind": probed.kind,
            "final_url": probed.final_url,
            "hops": list(probed.hops),
            "bytes": declared or 0,
            "content_type": probed.content_type,
            "asset": {"kind": probed.kind},
            "media_relay": "/content/media",
            "note": (
                (f"{declared / 1_000_000:.1f} MB of " if declared else "")
                + f"{probed.content_type}. "
                + (
                    "Keyframes are extracted in your browser — the file is never "
                    "decoded on the server."
                    if probed.kind == "video"
                    else "Paste a transcript below; timings turn it into a timeline."
                )
            ),
        }

    # Everything else is small enough to read, and has to be read to be useful:
    # a page has to be parsed for sections, a PDF for pages, an image encoded
    # for the vision call.
    try:
        fetched = await fetch_url(request.url)
    except UnsafeURL as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FetchFailed as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    envelope = {
        "kind": fetched.kind,
        "final_url": fetched.final_url,
        "hops": list(fetched.hops),
        "bytes": fetched.bytes_read,
        "content_type": fetched.content_type,
    }

    # ── image: straight to the vision path ──────────────────────────────
    if fetched.kind == "image":
        return {
            **envelope,
            "asset": {
                "kind": "image",
                "image_b64": base64.b64encode(fetched.content).decode("ascii"),
                "media_type": fetched.content_type,
            },
            "note": (
                "The image will be read for its visual hierarchy — what the eye "
                "lands on, in order. Time-dependent statistics are withheld: the "
                "regions of a static image are present simultaneously."
            ),
        }

    # Video and audio never reach here — they are answered from the probe
    # above, without a download.

    # ── PDF: pages are beats, already segmented by the document ─────────
    if fetched.kind == "document":
        try:
            pages, total = pages_from_pdf(fetched.content)
        except UnsupportedAsset as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        usable = [p for p in pages if len(p) >= 40]
        if len(usable) < 2:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"that PDF has {total} page(s) but fewer than two carry "
                    "readable text — it is probably scanned images with no text "
                    "layer. Export the slides as images and study them that way."
                ),
            )
        return {
            **envelope,
            "asset": {"kind": "document", "pages": usable},
            "pages": usable,
            "page_count": total,
            "truncated": total > len(pages),
            "note": (
                f"{len(usable)} readable page(s) of {total}"
                + (
                    f" — only the first {len(pages)} were read, so this studies "
                    "the opening of the document rather than all of it."
                    if total > len(pages)
                    else ""
                )
            ),
        }

    # ── page: unchanged, and still the subtlest of the five ─────────────
    #
    # Player hosts skip straight to the metadata rung. Their prose EXISTS and
    # is furniture, which is the one case the section extractor cannot detect
    # for itself — it would happily return "Sign in", "Subscribe" and a comment
    # policy as three beats.
    if skip_prose:
        return await _metadata_rung(request.url, fetched.html, envelope)

    try:
        sections = preview_page(ContentAsset.model_construct(kind="page", text=fetched.html))
    except UnsupportedAsset:
        # Reached and read, but no prose came out.
        #
        # This deliberately does NOT diagnose why. Two heuristics were tried —
        # an absolute character threshold and a prose-to-markup ratio — and
        # both misclassified real sites in both directions: tailwindcss.com
        # SUCCEEDS at a lower prose ratio (0.008) than stripe.com FAILS at
        # (0.014), because what matters is not how much text a page has but
        # whether any of it sits in blocks long enough to be a beat.
        #
        # It used to end here, as a 422 listing the caller's options. It no
        # longer has to: a page that builds its copy in the browser still
        # published OpenGraph tags server-side for the link preview, and those
        # describe the thing well enough to study. The refusal is kept inside
        # `_metadata_rung` for the case where even that comes back empty.
        return await _metadata_rung(request.url, fetched.html, envelope)

    # Send back the EXTRACTED PROSE, not the page.
    #
    # Returning raw HTML for the client to submit was wrong twice over. It blew
    # ContentAsset's 200 kB text limit on every real marketing site tested —
    # Stripe, Notion, Vercel and Tailwind all failed with an unhandled
    # ValidationError, which is to say the feature did not work on the web —
    # and it round-tripped a hostile document through the browser for no
    # reason. Escaped and re-wrapped, the payload is a few kB of the same text
    # the study will actually consume, and the researcher can edit it.
    body = "\n".join(f"<p>{html_escape(s)}</p>" for s in sections)
    return {
        **envelope,
        "text": body,
        "asset": {"kind": "page", "text": body},
        "sections": sections,
        "note": (
            f"{len(sections)} sections in DOM order — the order a scroller meets "
            "them. Hidden elements and unopened menus are excluded."
        ),
    }


@app.post("/v1/content/media")
async def content_media(caller: CallerDep, request: PageFetchRequest) -> StreamingResponse:
    """Relay video/audio bytes to the browser, which decodes them.

    THE PROBLEM THIS SOLVES. Keyframe extraction already happens client-side —
    deliberately, because decoding caller-supplied media server-side means
    running a decoder on untrusted input and would put ffmpeg in a container
    that has no native dependencies at all. But a browser cannot fetch a video
    from another origin: cross-origin reads are blocked, and no amount of
    wanting changes that. So a hosted MP4 could be studied by uploading the
    file and not by pasting its link, which is the wrong way round for the case
    where the researcher does not have the file.

    This closes it without moving the decoder. The bytes pass THROUGH the
    server, which validates the URL exactly as every other fetch does and never
    interprets what it is carrying, and the browser at the far end does the
    decoding it was always going to do.

    IMAGES TOO, and for the same reason rather than as a widening. A YouTube
    storyboard sheet is a grid of frames on i.ytimg.com that the browser has to
    crop on a canvas, and the dashboard's CSP is `img-src 'self' data: blob:` —
    so the browser cannot reach it either. The alternative was to add a third
    party to `img-src` and `connect-src` permanently, which buys a hostile
    script an approved exfiltration destination on every page of the app. A
    sheet is ~25 kB; relaying it costs nothing and keeps the policy shut.

    NOT AN OPEN PROXY, in the four ways that matter: it is behind the same API
    key as everything else, it re-runs the full SSRF validation rather than
    trusting that `/content/fetch` already did (the two calls are separate
    requests and a name can be re-pointed between them), it accepts only
    video, audio and image content types, and it caps bytes. The response is
    forced to `application/octet-stream` with a nosniff header so nothing
    relayed through here can be interpreted as script — or as an image — by the
    browser that receives it.
    """
    if is_player_url(request.url) and not is_youtube_url(request.url):
        raise HTTPException(
            status_code=422, detail="That is a player page, not a media file."
        )
    try:
        fetched = await fetch_url(request.url, kinds=("video", "audio", "image"))
    except UnsafeURL as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FetchFailed as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    def body():
        yield fetched.content

    return StreamingResponse(
        body(),
        # Deliberately NOT the upstream content type. The browser asked for
        # bytes to hand to a decoder, and labelling a relayed payload with a
        # type a browser will act on is how a relay becomes a delivery
        # mechanism for someone else's content on our origin.
        media_type="application/octet-stream",
        headers={
            "X-Content-Type-Options": "nosniff",
            "Content-Disposition": "attachment",
            "X-Upstream-Content-Type": fetched.content_type,
            "X-Upstream-Final-Url": fetched.final_url,
            "Cache-Control": "no-store",
        },
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
