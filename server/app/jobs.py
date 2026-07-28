"""Durable-ish background job queue for long swarm work.

Design: jobs persist in SQLite; one asyncio worker drains them sequentially
(each job already fans out internally under the swarm semaphore). On startup,
queued jobs AND jobs orphaned in 'running' by a crash are re-enqueued — job
execution is idempotent (a re-run just produces a fresh swarm_runs row).

This is deliberately a single-process queue with the exact seams a Temporal
migration needs later: executors are pure async functions keyed by kind, and
the storage schema already models the full job lifecycle.
"""

import asyncio
import logging
from typing import Any, Awaitable, Callable, Optional

from . import storage
from .batch_swarm import execute_batch_swarm
from .schemas import CompareRequest, PersonaSeed, SwarmRunRequest, WalkRequest
from .swarm import run_compare, run_swarm, run_walkthrough

log = logging.getLogger("cogniswarm.jobs")

# executor: payload -> (run_kind, result_dict). Registered per job kind.
Executor = Callable[[dict[str, Any]], Awaitable[tuple[str, dict[str, Any]]]]

_queue: asyncio.Queue[str] = asyncio.Queue()
_worker_task: asyncio.Task | None = None


def _personas(session_ids: list[str]) -> list[PersonaSeed]:
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
        raise RuntimeError("No profiled personas available.")
    return personas


async def _exec_swarm(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    req = SwarmRunRequest(**payload)
    agg = await run_swarm(
        _personas(req.session_ids), req.scenario, req.twins_per_persona, req.cognitive_load
    )
    return "swarm", agg.model_dump()


async def _exec_compare(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    req = CompareRequest(**payload)
    result = await run_compare(
        _personas(req.session_ids), req.variants, req.twins_per_persona, req.cognitive_load
    )
    return "compare", result.model_dump()


async def _exec_walk(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    req = WalkRequest(**payload)
    agg = await run_walkthrough(
        _personas(req.session_ids), req.steps, req.twins_per_persona, req.cognitive_load
    )
    return "walk", agg.model_dump()


async def _exec_batch_swarm(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    req = SwarmRunRequest(**payload)
    agg = await execute_batch_swarm(
        _personas(req.session_ids), req.scenario, req.twins_per_persona, req.cognitive_load
    )
    result = agg.model_dump()
    return "swarm", result


EXECUTORS: dict[str, Executor] = {
    "swarm": _exec_swarm,
    "compare": _exec_compare,
    "walk": _exec_walk,
    "batch_swarm": _exec_batch_swarm,
}


def enqueue(kind: str, payload: dict[str, Any], site_id: Optional[str] = None) -> str:
    if kind not in EXECUTORS:
        raise ValueError(f"Unknown job kind: {kind}")
    job_id = storage.insert_job(kind, payload, site_id=site_id)
    _queue.put_nowait(job_id)
    return job_id


async def _process(job_id: str) -> None:
    job = storage.get_job(job_id)
    if job is None or job["status"] in ("done", "error"):
        return
    storage.update_job(job_id, "running")
    try:
        run_kind, result = await EXECUTORS[job["kind"]](job["payload"])
        payload = dict(job["payload"])
        payload["job_id"] = job_id
        if job["kind"] == "batch_swarm":
            payload["mode"] = "batch"
        # Carried from the job row, not from a request context — the worker
        # has none. Without it every async run was written unowned.
        run_id = storage.insert_swarm_run(
            payload, result, kind=run_kind, site_id=job.get("site_id")
        )
        storage.update_job(job_id, "done", run_id=run_id)
    except Exception as exc:  # noqa: BLE001 — jobs must never kill the worker
        log.exception("job %s failed", job_id)
        storage.update_job(job_id, "error", error=str(exc)[:500])


async def _worker() -> None:
    while True:
        job_id = await _queue.get()
        try:
            await _process(job_id)
        finally:
            _queue.task_done()


def start_worker() -> None:
    """Call from app startup: recover interrupted jobs, then start draining."""
    global _worker_task
    for job_id in storage.queued_job_ids():
        storage.update_job(job_id, "queued")
        _queue.put_nowait(job_id)
    _worker_task = asyncio.get_event_loop().create_task(_worker())


def stop_worker() -> None:
    if _worker_task is not None:
        _worker_task.cancel()
