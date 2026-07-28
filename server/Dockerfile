# CogniSwarm backend.
#
# Every estimator in this service is implemented on the stdlib — there is no
# numpy, no scipy, nothing that needs a compiler — so the runtime image is
# genuinely small and neither stage needs build-essential.
#
# ONE INSTANCE ONLY. jobs.py re-queues every queued+running job at startup with
# no compare-and-swap claim, so a second replica executes every job a second
# time and bills OpenAI twice for it. That, SQLite's single writer, and the
# in-process worker thread together mean this service cannot be scaled
# horizontally and cannot run serverless. It stays true until Postgres and a
# real queue land.

FROM python:3.13-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Dependencies before source, so editing app code does not re-resolve the
# dependency tree on every rebuild.
COPY requirements.txt .
RUN pip install -r requirements.txt


FROM python:3.13-slim AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Explicit, rather than relying on config.py's /data autodetection: if the
    # volume fails to mount, this path is unwritable and the service fails
    # loudly at startup instead of silently opening a fresh empty database on
    # the container layer and serving an app that looks fine and has no data.
    COGNISWARM_DB=/data/cogniswarm.db

COPY --from=builder /opt/venv /opt/venv

WORKDIR /srv
COPY app ./app

# Non-root. The volume is chowned so the app can write its database; /srv stays
# owned by root and read-only to the app, so a compromise cannot rewrite the
# code it is running.
RUN useradd --system --uid 10001 --create-home cogniswarm \
    && mkdir -p /data \
    && chown -R cogniswarm:cogniswarm /data
USER cogniswarm

EXPOSE 8000

# Not a substitute for the platform's own probe, but it means `docker run` and
# compose report the same health the orchestrator will.
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2).status==200 else 1)"

# Single worker, deliberately — see the note at the top of this file.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
