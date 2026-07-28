"""Guards for the deployment contract.

These fail on exactly the misconfigurations that are invisible in development
and total in production: a health check that cannot answer, and a CORS
allowlist that silently drops every byte of telemetry.
"""

from __future__ import annotations

import importlib
import os

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


# ── health ────────────────────────────────────────────────────────────────


def test_healthz_is_reachable_without_an_api_key():
    """It runs before the app is reachable by anything else. Behind
    `require_api_key` the orchestrator kills a perfectly working container as
    unhealthy — and the analysis surface fails CLOSED, so that is exactly what
    would happen on any correctly-secured deployment."""
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_healthz_leaks_no_configuration():
    """Public endpoint. A prober is entitled to know up or not up, nothing
    else — not the DB path, not the version, not the model names."""
    body = client.get("/healthz").json()
    assert set(body) == {"status"}


def test_healthz_touches_the_datastore(monkeypatch):
    """The failure this exists to catch is a volume that did not mount, where
    the process is alive and serving an empty database. A health check that
    only proves the event loop is running would report that as healthy."""
    from app import main as main_module

    def boom() -> None:
        raise RuntimeError("disk gone")

    monkeypatch.setattr(main_module.storage, "ping", boom)
    assert client.get("/healthz").status_code == 503


# ── CORS: the silent-telemetry-loss guard ─────────────────────────────────


def _preflight(c: TestClient, origin: str):
    return c.options(
        "/v1/ingest",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "POST",
            # The beacon is sent as a Blob typed application/json, which is NOT
            # a CORS-safelisted content type — this header is the whole reason
            # a preflight happens at all.
            "Access-Control-Request-Headers": "content-type",
        },
    )


def test_configured_origin_may_post_telemetry(monkeypatch):
    monkeypatch.setenv("COGNISWARM_ALLOWED_ORIGINS", "https://acme.com")
    import app.main as main_module

    reloaded = importlib.reload(main_module)
    with TestClient(reloaded.app) as c:
        r = _preflight(c, "https://acme.com")
        assert r.status_code == 200
        assert r.headers.get("access-control-allow-origin") == "https://acme.com"


def test_unconfigured_origin_is_refused(monkeypatch):
    """The default is an empty allowlist, which means a correctly integrated
    customer site drops 100% of its telemetry — and finds out never, because
    `navigator.sendBeacon` returns true on queueing and the request never
    reaches a route to be logged. This asserts the refusal is real, so the
    allowlist is understood to be load-bearing rather than optional."""
    monkeypatch.setenv("COGNISWARM_ALLOWED_ORIGINS", "https://acme.com")
    import app.main as main_module

    reloaded = importlib.reload(main_module)
    with TestClient(reloaded.app) as c:
        r = _preflight(c, "https://not-acme.com")
        assert r.headers.get("access-control-allow-origin") is None


def test_allowlist_does_not_open_the_analysis_surface(monkeypatch):
    """An origin entry is permission to POST telemetry, not permission to drive
    the API from someone's browser.

    Note what actually enforces this. CORSMiddleware is app-wide, so it echoes
    the allowed origin on a rejected preflight too — the protection is the
    METHOD allowlist, not a missing header. A browser refuses the request
    because the preflight fails and GET is absent from
    access-control-allow-methods. Asserting on the origin header instead would
    pass for the wrong reason and break the moment the middleware's error
    formatting changed.
    """
    monkeypatch.setenv("COGNISWARM_ALLOWED_ORIGINS", "https://acme.com")
    import app.main as main_module

    reloaded = importlib.reload(main_module)
    with TestClient(reloaded.app) as c:
        r = c.options(
            "/v1/sessions",
            headers={
                "Origin": "https://acme.com",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert r.status_code == 400  # preflight refused
        assert "GET" not in (r.headers.get("access-control-allow-methods") or "")


# ── database location ─────────────────────────────────────────────────────


def test_explicit_db_env_always_wins():
    """Autodetection must never override an operator's explicit choice."""
    from app import config as config_module

    os.environ["COGNISWARM_DB"] = "/tmp/explicit-choice.db"
    try:
        reloaded = importlib.reload(config_module)
        assert str(reloaded.DB_PATH) == "/tmp/explicit-choice.db"
    finally:
        del os.environ["COGNISWARM_DB"]
        importlib.reload(config_module)


# ── annotation ordering: caught only by the container ─────────────────────


def test_every_schema_resolves_without_forward_references():
    """Guards a bug that the local suite structurally cannot catch.

    Python 3.14 defers annotation evaluation (PEP 649), so a model that
    references a class defined LATER in the file imports fine on this machine.
    Python 3.13 — which is what the container runs, and which pyproject
    declares support for — evaluates eagerly and raises NameError at import,
    so the service crashed on startup while every test passed locally.

    Building a TypeAdapter forces every annotation to resolve now, reproducing
    the eager behaviour regardless of the interpreter running the tests.
    """
    import inspect

    from pydantic import BaseModel, TypeAdapter

    from app import schemas

    models = [
        obj
        for _name, obj in inspect.getmembers(schemas, inspect.isclass)
        if issubclass(obj, BaseModel) and obj.__module__ == schemas.__name__
    ]
    assert len(models) > 20, "sanity: the schema module should expose many models"
    for model in models:
        TypeAdapter(model)  # raises if any annotation cannot be resolved
