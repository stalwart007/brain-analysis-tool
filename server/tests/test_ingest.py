from fastapi.testclient import TestClient

from app.main import app  # test DB is configured in conftest.py

client = TestClient(app)

FEATURES = {
    "segment_start": 0,
    "segment_end": 12000,
    "event_count": 140,
    "scroll_velocity_mean": 0.8,
    "scroll_velocity_variance": 0.3,
    "scroll_direction_changes": 6,
    "micro_hesitation_count": 4,
    "hesitation_total_ms": 7100,
    "rage_click_bursts": 1,
    "click_count": 9,
    "backtrack_ratio": 0.22,
    "zone_dwell_ms": {"pricing-table": 6400, "hero": 900},
    "zone_click_counts": {"cta": 4},
}


def _envelope(consent=True):
    return {
        "site_id": "test-site",
        "consent": consent,
        "sdk_version": "0.1.0",
        "page_path": "/",
        "features": FEATURES,
    }


def test_ingest_requires_consent():
    res = client.post("/v1/ingest", json=_envelope(consent=False))
    assert res.status_code == 403


def test_ingest_accepts_and_lists():
    res = client.post("/v1/ingest", json=_envelope())
    assert res.status_code == 202
    session_id = res.json()["session_id"]

    rows = client.get("/v1/sessions").json()
    assert any(r["id"] == session_id for r in rows)
    row = next(r for r in rows if r["id"] == session_id)
    assert row["features"]["rage_click_bursts"] == 1
    assert row["signal"] is None  # unprofiled until /profile is called


def test_empty_segment_ignored():
    envelope = _envelope()
    envelope["features"] = {**FEATURES, "event_count": 0}
    res = client.post("/v1/ingest", json=envelope)
    assert res.status_code == 202
    assert res.json()["status"] == "ignored"


def test_swarm_requires_personas():
    res = client.post(
        "/v1/swarm/run",
        json={"scenario": "A landing page", "twins_per_persona": 1, "cognitive_load": "low"},
    )
    # test DB has sessions but no profiled personas
    assert res.status_code == 400
