from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

FEATURES = {"segment_start": 0, "segment_end": 1000, "event_count": 5}


def _ingest(panel_token=None):
    body = {
        "site_id": "partner-site",
        "consent": True,
        "sdk_version": "0.1.0",
        "page_path": "/",
        "features": FEATURES,
    }
    if panel_token:
        body["panel_token"] = panel_token
    return client.post("/v1/ingest", json=body)


def test_full_panel_lifecycle_with_erasure():
    # 1. admin provisions an invite
    member = client.post("/v1/panel/members", json={"label": "panelist 1"}).json()
    token = member["token"]

    # 2. disclosure page serves with the token baked in
    page = client.get(f"/panel/{token}")
    assert page.status_code == 200
    assert token in page.text
    assert "Revoke" in page.text

    # 3. telemetry BEFORE consent is rejected
    assert _ingest(panel_token=token).status_code == 403

    # 4. member consents -> telemetry accepted and tagged
    assert client.post(f"/v1/panel/{token}/consent").status_code == 200
    res = _ingest(panel_token=token)
    assert res.status_code == 202
    session_id = res.json()["session_id"]
    rows = client.get("/v1/sessions").json()
    assert any(r["id"] == session_id for r in rows)

    # 5. member list shows enrollment without leaking tokens
    members = client.get("/v1/panel/members").json()
    me = next(m for m in members if m["label"] == "panelist 1")
    assert "token" not in me
    assert me["consented_at"] is not None
    assert me["session_count"] >= 1

    # 6. revocation erases every session linked to the member
    revoked = client.post(f"/v1/panel/{token}/revoke").json()
    assert revoked["deleted_sessions"] >= 1
    rows = client.get("/v1/sessions").json()
    assert not any(r["id"] == session_id for r in rows)

    # 7. post-revocation: telemetry and re-consent are both rejected
    assert _ingest(panel_token=token).status_code == 403
    assert client.post(f"/v1/panel/{token}/consent").status_code == 404


def test_unknown_token_rejected():
    assert _ingest(panel_token="bogus").status_code == 403
    assert client.get("/panel/bogus").status_code == 404


def test_non_panel_ingest_unaffected():
    assert _ingest().status_code == 202
