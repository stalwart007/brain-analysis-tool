from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_analysis_surface_locked_when_keys_configured(monkeypatch):
    monkeypatch.setenv("COGNISWARM_API_KEYS", "key-alpha, key-beta")

    assert client.get("/v1/sessions").status_code == 401
    assert client.get("/v1/sessions", headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.get("/v1/sessions", headers={"X-API-Key": "key-alpha"}).status_code == 200
    assert client.get("/v1/sessions", headers={"X-API-Key": "key-beta"}).status_code == 200


def test_ingest_stays_public_with_keys_configured(monkeypatch):
    """Ingest is hit by end-user browsers on client sites — consent is its gate,
    not an API key."""
    monkeypatch.setenv("COGNISWARM_API_KEYS", "key-alpha")
    res = client.post(
        "/v1/ingest",
        json={
            "site_id": "s",
            "consent": True,
            "sdk_version": "0.1.0",
            "page_path": "/",
            "features": {"segment_start": 0, "segment_end": 1, "event_count": 1},
        },
    )
    assert res.status_code == 202


def test_fails_closed_when_nothing_is_configured(monkeypatch):
    """No keys and no explicit opt-out ⇒ the analysis surface is CLOSED.

    Defaulting to open would mean a missing or misspelled COGNISWARM_API_KEYS
    silently publishes sessions, panel members, run history (including every
    twin's inner monologue) and every credit-spending endpoint. An auth check
    that defaults to "allow everything" is not an auth check, so the default is
    a refusal that names its own fix.
    """
    monkeypatch.delenv("COGNISWARM_API_KEYS", raising=False)
    monkeypatch.delenv("COGNISWARM_ALLOW_ANONYMOUS", raising=False)
    res = client.get("/v1/sessions")
    assert res.status_code == 503
    assert "COGNISWARM_API_KEYS" in res.json()["detail"]


def test_anonymous_opt_in_is_explicit(monkeypatch):
    """Local dev opens the surface deliberately, never by omission."""
    monkeypatch.delenv("COGNISWARM_API_KEYS", raising=False)
    monkeypatch.setenv("COGNISWARM_ALLOW_ANONYMOUS", "1")
    assert client.get("/v1/sessions").status_code == 200


def test_configured_keys_outrank_the_anonymous_flag(monkeypatch):
    """The escape hatch must not be able to weaken a server that has keys."""
    monkeypatch.setenv("COGNISWARM_API_KEYS", "key-alpha")
    monkeypatch.setenv("COGNISWARM_ALLOW_ANONYMOUS", "1")
    assert client.get("/v1/sessions").status_code == 401
    assert client.get("/v1/sessions", headers={"X-API-Key": "key-alpha"}).status_code == 200
