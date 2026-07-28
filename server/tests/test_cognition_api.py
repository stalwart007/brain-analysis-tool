"""API wiring for the cognitive-profile pipeline: rich telemetry in, streamed
model computation out. Runs key-free — the LLM stages must fail cleanly AFTER
the mathematical stages streamed real results."""

import json
import math
import random

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _rich_features():
    rng = random.Random(31)
    t, stream, vel = 0.0, [], []
    for _ in range(260):
        t += rng.expovariate(1 / 150.0)
        stream.append([round(t), rng.choices(range(7), weights=[5, 3, 2, 1, 0.3, 0.7, 1])[0]])
        vel.append([round(t), round(abs(math.sin(t / 800.0)) + 0.05 * rng.random(), 4)])
    return {
        "segment_start": 0,
        "segment_end": t,
        "event_count": len(stream),
        "scroll_velocity_mean": 0.6,
        "scroll_velocity_variance": 0.2,
        "scroll_direction_changes": 9,
        "micro_hesitation_count": 5,
        "hesitation_total_ms": 8000,
        "rage_click_bursts": 0,
        "click_count": 12,
        "backtrack_ratio": 0.2,
        "zone_dwell_ms": {"hero": 4000},
        "zone_click_counts": {"cta": 3},
        "event_stream": stream,
        "velocity_series": vel,
        "decision_latencies_ms": [max(120, rng.gauss(640, 150)) for _ in range(24)],
    }


def _ingest_rich() -> str:
    r = client.post(
        "/v1/ingest",
        json={
            "site_id": "cog-site",
            "consent": True,
            "sdk_version": "0.2.0",
            "page_path": "/rich",
            "features": _rich_features(),
        },
    )
    assert r.status_code == 202, r.text
    return r.json()["session_id"]


def test_ingest_accepts_cognitive_series():
    sid = _ingest_rich()
    session = next(s for s in client.get("/v1/sessions").json() if s["id"] == sid)
    assert len(session["features"]["event_stream"]) == 260
    assert len(session["features"]["decision_latencies_ms"]) == 24


def test_profile_stream_emits_model_stages_then_llm_failure_without_key():
    sid = _ingest_rich()
    with client.stream("POST", f"/v1/sessions/{sid}/profile/stream") as resp:
        assert resp.status_code == 200
        frames = []
        for line in resp.iter_lines():
            if line.startswith("data:"):
                frames.append(json.loads(line[5:]))
    stages = [f.get("stage") for f in frames if f.get("type") == "cognition"]
    for st in ["ddm", "hmm", "information", "spectral", "foraging", "habituation", "complete"]:
        assert st in stages, f"missing stage {st}"
    # the HMM must have streamed its actual EM convergence trace
    conv = next(
        f for f in frames if f.get("stage") == "hmm" and f.get("status") == "converged"
    )
    assert len(conv["ll_trace"]) >= 2 and "bic_by_k" in conv
    # key-free: after the math, the LLM stage fails with a clean error frame
    assert frames[-1]["type"] == "error"
    # and the computed cognition profile was persisted despite the LLM failure
    session = next(s for s in client.get("/v1/sessions").json() if s["id"] == sid)
    assert session["cognition"] is not None
    assert "ez_diffusion" in session["cognition"]["models_run"]
    assert "state_occupancy" in session["cognition"]["summary"]


def test_profile_stream_unknown_session_404s():
    r = client.post("/v1/sessions/nope/profile/stream")
    assert r.status_code == 404
