# Deploying CogniSwarm

Three units with different shapes: a **private** backend, a **public**
dashboard, and an **SDK** that runs on other people's websites.

---

## The one constraint that governs everything

**The backend runs as exactly one instance.** Not "should" — cannot.

- `jobs.py` re-queues every `queued`+`running` job at startup with no
  compare-and-swap claim, so a second replica executes every job a second time
  and bills OpenAI twice for it.
- SQLite has a single writer.
- The job worker is an in-process thread, so it dies with a serverless function.

Do not set `--workers > 1`, do not scale to 2 machines, do not deploy the
backend to anything that suspends idle instances mid-run. This holds until
Postgres and a real queue land (`SCALING-PLAN.md` step 1).

The dashboard has no such constraint and scales horizontally.

---

## Topology

```
        internet
           │
           ▼
   ┌───────────────┐
   │   dashboard   │  public :3000
   │   (Next.js)   │
   └───────┬───────┘
           │  private network only
           ▼
   ┌───────────────┐
   │    backend    │  NO public address
   │   (FastAPI)   │
   └───────┬───────┘
           ▼
     volume /data     SQLite + WAL
```

**The backend must not have a public address.** The analysis surface exposes
run history containing every twin's inner monologue, panel membership, and
every endpoint that spends OpenAI credit. Tenant isolation and the
persona-eviction path are now fixed, so this is defence in depth rather than
the only thing standing between a tenant's data and the internet — but it is
free, and it means the next hole found in that surface is not remotely
reachable while it is being fixed.

Telemetry still reaches it: the SDK posts to the dashboard's public
`/api/ingest`, which forwards only that one path and injects no API key.

---

## Deploy

### Local, production-shaped

```bash
cp server/.env.example server/.env      # add your OPENAI_API_KEY
export AUTH_SECRET=$(openssl rand -base64 32)
docker compose up --build
```

The backend is deliberately unpublished in `docker-compose.yml`. If something
works via `localhost:8000`, that is a hole that will not exist in production
and a bug you will only find after deploying.

Provision a member (there is no signup flow, by design):

```bash
npm run add-user --prefix dashboard -- you@team.com your-password "Your Name"
```

### Fly.io

Two apps in one org. The backend gets a volume and **no public services block**;
the dashboard reaches it over the private `.internal` network.

```bash
fly launch --no-deploy --name cogniswarm-api   --path server
fly volumes create data --size 3 --app cogniswarm-api
fly secrets set --app cogniswarm-api \
    OPENAI_API_KEY=sk-... \
    COGNISWARM_API_KEYS=$(openssl rand -hex 24) \
    COGNISWARM_ALLOWED_ORIGINS=https://your-dashboard-host
fly deploy --path server

fly launch --no-deploy --name cogniswarm-app   --path dashboard
fly secrets set --app cogniswarm-app \
    AUTH_SECRET=$(openssl rand -base64 32) \
    COGNISWARM_BACKEND=http://cogniswarm-api.internal:8000 \
    COGNISWARM_API_KEY=<the same value as COGNISWARM_API_KEYS above>
fly deploy --path dashboard
```

In the backend's `fly.toml`, mount the volume at `/data`, set
`auto_stop_machines = false` (a suspended machine kills in-flight swarm runs
and the job worker), and **delete the `[[services]]` block** so it has no public
address.

Expect roughly $10/mo: two shared-cpu-1x machines and a 3 GB volume.

---

## Environment

### Backend

| | |
|---|---|
| `OPENAI_API_KEY` | required |
| `COGNISWARM_API_KEYS` | **required unless** `COGNISWARM_ALLOW_ANONYMOUS=1`. Auth fails closed: with neither, every `/v1/*` returns 503. Two forms — `key:site` scopes a key to one tenant, a bare `key` is unscoped (admin / single-tenant) |
| `COGNISWARM_ALLOWED_ORIGINS` | customer origins that may POST telemetry, comma separated, scheme included. **Empty means the SDK silently drops everything** — see below |
| `COGNISWARM_DB` | set to `/data/cogniswarm.db`. Defaults into the repo tree, which in a container is an ephemeral image layer |
| `COGNISWARM_CONCURRENCY` | max concurrent twin calls, default 8 |
| `COGNISWARM_MAX_TWINS_PER_RUN` | hard per-request fan-out ceiling, default 2000. A request over it is refused (413) with the computed number |
| `COGNISWARM_INGEST_RATE_LIMIT` | segments per window per (site, client), default 120 |
| `COGNISWARM_INGEST_RATE_WINDOW_S` | rate-limit window, default 60 |

### Dashboard

| | |
|---|---|
| `AUTH_SECRET` | required — the app refuses to start in production without it |
| `COGNISWARM_BACKEND` | the private backend URL |
| `COGNISWARM_API_KEY` | must match one of the backend's `COGNISWARM_API_KEYS` |
| `COGNISWARM_USERS_FILE` | path to the mounted member list. **This is state, not config** |

---

## The two things most likely to bite you

**1. `users.json` is state.** It is written by `npm run add-user` and holds
bcrypt hashes. Bake it into an image and every account disappears on the next
rebuild — and the symptom is "my password stopped working", which reads as an
auth bug rather than a deployment one. Mount it at `COGNISWARM_USERS_FILE`.

**2. Backups cannot be `cp`.** WAL keeps recent transactions in a `-wal`
sidecar until a checkpoint, so a copied file opens cleanly, looks like a
database, and is missing everything written since the last checkpoint. You find
out when you restore. Use:

```bash
docker compose exec backend python -c \
  "import sys; sys.path.insert(0,'/srv'); from app import storage; \
   print(storage.backup_to('/data/backup.db'))"
```

---

## The SDK

Built and published to the dashboard's static tree at an **immutable, versioned**
URL:

```bash
npm run release --prefix packages/collector
```

That writes `dashboard/public/sdk/v<version>/cogniswarm.min.js` and prints the
embed snippet with its SRI hash. It refuses to overwrite a published version
with different bytes — customers embed these URLs on their own production
sites, so a mutable URL means one bad build reaches all of them at once with no
staging and no rollback.

**The allowlist is not optional.** `transport.ts` sends the beacon as a Blob
typed `application/json`, which is not a CORS-safelisted content type, so every
cross-origin POST is preceded by a preflight. With the customer's origin absent
from `COGNISWARM_ALLOWED_ORIGINS` that preflight is refused and **100% of their
telemetry is dropped** — silently at both ends, because `navigator.sendBeacon`
returns `true` on queueing and the request never reaches a route to be logged.
Origins match exactly: `https://acme.com` does not cover `https://www.acme.com`.

---

## Known gaps you are deploying with

Not blockers for a demo or early customers, but know them:

- **Rate limiting is in-process and best-effort.** It is a backstop against a
  script, not a defence against a distributed flood — that belongs at the edge.
  Because the backend sits behind the dashboard passthrough, the socket peer is
  the proxy for every request, so `X-Forwarded-For` is what distinguishes
  callers and it is spoofable. The per-site half of the key still holds.
- **Still set an OpenAI spend limit.** `COGNISWARM_MAX_TWINS_PER_RUN` bounds a
  single request; it does not bound many requests.
- **Anonymous mode is unscoped.** `COGNISWARM_ALLOW_ANONYMOUS=1` has no key to
  carry a tenant, so it sees everything. Correct for local development and
  single-tenant servers; never set it on a shared deployment.
- **Runs written before tenancy have no owner.** They stay visible to an
  unscoped caller and invisible to a scoped one. Backfill `swarm_runs.site_id`
  if you need historical runs attributed.
- **No abort propagation.** Closing a tab mid-run does not cancel the fan-out;
  it bills to completion.
- **Observability is one log line.** No request IDs, no metrics, and
  `completion.usage` is never read — there is no per-run token or cost figure
  anywhere.
