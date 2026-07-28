"""SQLite persistence for Phase 1 (dev). Swap for Postgres + ClickHouse in Phase 2/3 —
the interface is deliberately thin so the migration is mechanical."""

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Optional

from .config import DB_PATH

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    site_id TEXT NOT NULL,
    page_path TEXT NOT NULL,
    received_at TEXT NOT NULL,
    features TEXT NOT NULL,          -- FeaturePayload JSON
    signal TEXT,                     -- BehavioralSignal JSON (null until profiled)
    persona TEXT                     -- PersonaSeed JSON (null until seeded)
);
CREATE TABLE IF NOT EXISTS swarm_runs (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    -- swarm | compare | walk | price | objection | virality | content | optimize | sequence
    kind TEXT NOT NULL DEFAULT 'swarm',
    request TEXT NOT NULL,
    result TEXT NOT NULL,
    actuals TEXT                          -- ActualsPayload JSON (validation harness)
);
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    kind TEXT NOT NULL,                   -- swarm | compare | walk | batch_swarm
    status TEXT NOT NULL,                 -- queued | running | done | error
    payload TEXT NOT NULL,
    run_id TEXT,                          -- resulting swarm_runs row when done
    error TEXT
);
CREATE TABLE IF NOT EXISTS panel_members (
    id TEXT PRIMARY KEY,
    token TEXT NOT NULL UNIQUE,           -- invite/enrollment token (capability URL)
    label TEXT NOT NULL,                  -- admin-facing label; never PII by policy
    created_at TEXT NOT NULL,
    consented_at TEXT,                    -- null until the member accepts the disclosure
    revoked_at TEXT                       -- revocation erases the member's telemetry
);
"""


#: Every list query in this module is an ORDER BY or a WHERE over a column with
#: no index, so each one is a full table scan — including the two the dashboard
#: polls every 15 seconds and the JOIN behind panel-member erasure, which scans
#: `sessions` once per member. SQLite will not create these itself.
_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_sessions_received  ON sessions(received_at DESC);
CREATE INDEX IF NOT EXISTS idx_sessions_panel     ON sessions(panel_member_id);
CREATE INDEX IF NOT EXISTS idx_runs_kind_created  ON swarm_runs(kind, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_runs_created       ON swarm_runs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_jobs_created       ON jobs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_jobs_status        ON jobs(status);
"""


@contextmanager
def _conn():
    # `timeout` is the busy-wait before raising "database is locked". The
    # default of 5 s is short for a process that writes from the request
    # threadpool, the job worker and (blockingly) from inside SSE generators
    # all at once.
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    conn.row_factory = sqlite3.Row
    # WAL lets readers proceed during a write instead of blocking on it, which
    # is the entire concurrency story here: without it a single in-flight write
    # stalls every concurrent request, and under load that surfaces as an
    # unhandled OperationalError mid-stream. NORMAL synchronous is the standard
    # pairing — durable against process crash, which is the failure this cares
    # about, without an fsync per transaction.
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=30000")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with _conn() as c:
        c.executescript(_SCHEMA)
        # dev-friendly migrations for databases created before newer columns existed
        cols = {r["name"] for r in c.execute("PRAGMA table_info(swarm_runs)")}
        if "kind" not in cols:
            c.execute("ALTER TABLE swarm_runs ADD COLUMN kind TEXT NOT NULL DEFAULT 'swarm'")
        if "actuals" not in cols:
            c.execute("ALTER TABLE swarm_runs ADD COLUMN actuals TEXT")
        session_cols = {r["name"] for r in c.execute("PRAGMA table_info(sessions)")}
        if "panel_member_id" not in session_cols:
            c.execute("ALTER TABLE sessions ADD COLUMN panel_member_id TEXT")
        if "cognition" not in session_cols:
            c.execute("ALTER TABLE sessions ADD COLUMN cognition TEXT")
        # after the migrations, so indexes on added columns are creatable
        c.executescript(_INDEXES)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def insert_session(
    site_id: str,
    page_path: str,
    features: dict[str, Any],
    panel_member_id: Optional[str] = None,
) -> str:
    session_id = uuid.uuid4().hex[:12]
    with _conn() as c:
        c.execute(
            "INSERT INTO sessions (id, site_id, page_path, received_at, features, panel_member_id)"
            " VALUES (?,?,?,?,?,?)",
            (session_id, site_id, page_path, _now(), json.dumps(features), panel_member_id),
        )
    return session_id


def get_session(session_id: str) -> Optional[dict[str, Any]]:
    with _conn() as c:
        row = c.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
    return _hydrate(row) if row else None


def list_sessions(limit: int = 200) -> list[dict[str, Any]]:
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM sessions ORDER BY received_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [_hydrate(r) for r in rows]


def set_signal(session_id: str, signal: dict[str, Any]) -> None:
    with _conn() as c:
        c.execute("UPDATE sessions SET signal = ? WHERE id = ?", (json.dumps(signal), session_id))


def set_persona(session_id: str, persona: dict[str, Any]) -> None:
    with _conn() as c:
        c.execute("UPDATE sessions SET persona = ? WHERE id = ?", (json.dumps(persona), session_id))


def set_cognition(session_id: str, profile: dict[str, Any]) -> None:
    with _conn() as c:
        c.execute(
            "UPDATE sessions SET cognition = ? WHERE id = ?",
            (json.dumps(profile), session_id),
        )


def insert_swarm_run(
    request: dict[str, Any], result: dict[str, Any], kind: str = "swarm"
) -> str:
    run_id = uuid.uuid4().hex[:12]
    with _conn() as c:
        c.execute(
            "INSERT INTO swarm_runs (id, created_at, kind, request, result) VALUES (?,?,?,?,?)",
            (run_id, _now(), kind, json.dumps(request), json.dumps(result)),
        )
    return run_id


def get_swarm_run(run_id: str) -> Optional[dict[str, Any]]:
    with _conn() as c:
        row = c.execute("SELECT * FROM swarm_runs WHERE id = ?", (run_id,)).fetchone()
    return _hydrate_run(row) if row else None


def set_actuals(run_id: str, actuals: dict[str, Any]) -> bool:
    with _conn() as c:
        cur = c.execute(
            "UPDATE swarm_runs SET actuals = ? WHERE id = ?",
            (json.dumps(actuals), run_id),
        )
        return cur.rowcount > 0


def list_swarm_runs(limit: int = 50, kind: Optional[str] = None) -> list[dict[str, Any]]:
    query = "SELECT * FROM swarm_runs"
    params: tuple[Any, ...] = ()
    if kind:
        query += " WHERE kind = ?"
        params = (kind,)
    query += " ORDER BY created_at DESC LIMIT ?"
    with _conn() as c:
        rows = c.execute(query, (*params, limit)).fetchall()
    return [_hydrate_run(r) for r in rows]


def _hydrate_run(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    d["request"] = json.loads(d["request"])
    d["result"] = json.loads(d["result"])
    d["actuals"] = json.loads(d["actuals"]) if d.get("actuals") else None
    return d


# ------------------------------------------------------------------ jobs


def insert_job(kind: str, payload: dict[str, Any]) -> str:
    job_id = uuid.uuid4().hex[:12]
    now = _now()
    with _conn() as c:
        c.execute(
            "INSERT INTO jobs (id, created_at, updated_at, kind, status, payload) VALUES (?,?,?,?,?,?)",
            (job_id, now, now, kind, "queued", json.dumps(payload)),
        )
    return job_id


def update_job(
    job_id: str,
    status: str,
    run_id: Optional[str] = None,
    error: Optional[str] = None,
) -> None:
    with _conn() as c:
        c.execute(
            "UPDATE jobs SET status = ?, run_id = COALESCE(?, run_id),"
            " error = ?, updated_at = ? WHERE id = ?",
            (status, run_id, error, _now(), job_id),
        )


def get_job(job_id: str) -> Optional[dict[str, Any]]:
    with _conn() as c:
        row = c.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    return _hydrate_job(row) if row else None


def list_jobs(limit: int = 50) -> list[dict[str, Any]]:
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [_hydrate_job(r) for r in rows]


def queued_job_ids() -> list[str]:
    """Jobs to (re)process at startup: queued, plus any 'running' orphaned by a
    previous process crash/restart — they are safe to re-run (idempotent inserts)."""
    with _conn() as c:
        rows = c.execute(
            "SELECT id FROM jobs WHERE status IN ('queued', 'running') ORDER BY created_at"
        ).fetchall()
    return [r["id"] for r in rows]


def _hydrate_job(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    d["payload"] = json.loads(d["payload"])
    return d


# ------------------------------------------------------------------ panel


def insert_panel_member(label: str) -> dict[str, Any]:
    member_id = uuid.uuid4().hex[:12]
    token = uuid.uuid4().hex  # capability token for the disclosure/consent URL
    with _conn() as c:
        c.execute(
            "INSERT INTO panel_members (id, token, label, created_at) VALUES (?,?,?,?)",
            (member_id, token, label, _now()),
        )
    return {"id": member_id, "token": token, "label": label}


def get_member_by_token(token: str) -> Optional[dict[str, Any]]:
    with _conn() as c:
        row = c.execute(
            "SELECT * FROM panel_members WHERE token = ?", (token,)
        ).fetchone()
    return dict(row) if row else None


def list_panel_members() -> list[dict[str, Any]]:
    with _conn() as c:
        rows = c.execute(
            "SELECT m.*, COUNT(s.id) AS session_count FROM panel_members m"
            " LEFT JOIN sessions s ON s.panel_member_id = m.id"
            " GROUP BY m.id ORDER BY m.created_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def set_member_consent(member_id: str) -> None:
    with _conn() as c:
        c.execute(
            "UPDATE panel_members SET consented_at = ? WHERE id = ? AND revoked_at IS NULL",
            (_now(), member_id),
        )


def revoke_member(member_id: str) -> int:
    """Revoke consent AND erase the member's telemetry (right to erasure).
    Returns the number of deleted sessions."""
    with _conn() as c:
        c.execute(
            "UPDATE panel_members SET revoked_at = ? WHERE id = ?", (_now(), member_id)
        )
        cur = c.execute("DELETE FROM sessions WHERE panel_member_id = ?", (member_id,))
        return cur.rowcount


def _hydrate(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    d["features"] = json.loads(d["features"])
    d["signal"] = json.loads(d["signal"]) if d["signal"] else None
    d["persona"] = json.loads(d["persona"]) if d["persona"] else None
    d["cognition"] = json.loads(d["cognition"]) if d.get("cognition") else None
    return d
