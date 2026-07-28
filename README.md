# CogniSwarm — Phase 1

Digital-phenotyping + cognitive-simulation platform. Phase 1 delivers the full
vertical slice: consent-gated telemetry collection → deterministic feature
extraction → LLM behavioral profiling → persona seeding → a mini twin swarm
with an aggregation dashboard.

**Domain-agnostic by design** — the swarm scenario is free-form, so the same
pipeline serves ad A/B pre-testing, content drop-off forecasting, and UX flow
simulation.

## Layout

```
packages/collector/     TypeScript SDK (Layer 1–3): consent gate, rAF batching,
                        Web Worker feature math, sendBeacon transport
server/app/             FastAPI: ingest, Layer-4 profiler (OpenAI structured
                        outputs), persona seeding, async swarm orchestrator
dashboard/              Next.js App Router product UI: closed-group auth, bento
                        dashboard, R3F 3D hero, Recharts, headless HLS player
examples/demo-site/     Instrumented demo page to generate telemetry locally
```

## Quickstart

```bash
# 1. Build the collector
cd packages/collector && npm install && npm run build

# 2. Install server deps
cd ../../server && python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. Credentials — copy the template, then paste your key into .env
cp .env.example .env

# 4. Run the API (port 8000)
./run.sh

# 5. Dashboard (separate shell)
cd ../dashboard && npm install
cp .env.example .env.local                                    # AUTH_SECRET etc.
npm run add-user -- you@team.com your-password "Your Name"   # provision access
npm run dev                                                   # port 3000
```

**Python 3.12+ is required** (`random.binomialvariate`). The analysis surface
**fails closed**: with neither `COGNISWARM_API_KEYS` nor
`COGNISWARM_ALLOW_ANONYMOUS` set, every `/v1/*` endpoint returns 503 rather
than silently publishing run history and credit-spending endpoints. The
`.env.example` templates set the local-development escape hatch explicitly —
which is why step 3 copies them rather than exporting one variable by hand.

Then:

1. Open `http://localhost:8000/demo`, **Accept** the consent banner, scroll /
   hover / rage-click the CTA for ~15 seconds.
2. Open `http://localhost:3000`, sign in, and the session segments appear →
   **Profile** one (OpenAI turns the feature vector into a behavioral segment +
   persona seed).
3. Paste a scenario (ad copy, transcript, UI flow description), pick a
   cognitive load, **Run swarm** → aggregated engagement/intent/drop-off/friction.

Tests: `cd server && pytest` (no API key needed — LLM calls aren't exercised).

## Dashboard notes

- **Closed-group auth, no signup.** Members are provisioned only via
  `npm run add-user` (bcrypt hashes in `dashboard/data/users.json`, gitignored).
  Sessions are signed httpOnly JWT cookies; middleware gates every route and
  login is rate-limited (5 attempts / 15 min). Set `AUTH_SECRET` in production
  (the app refuses to start without it).
- **Performance budget.** The 3D layer (React Three Fiber) and the HLS engine
  (hls.js) are lazy-loaded and never in the initial bundle; the render loop
  pauses when the canvas leaves the viewport or the tab hides, and honors
  `prefers-reduced-motion`. Login page first-load JS: ~144 kB.
- **Charts** use the validated dark data-viz palette (series `#3987e5` /
  `#d95926` — CVD-checked against the `#1a1a19` surface).
- The backend origin is proxied under `/api/cs/*` (configure with
  `COGNISWARM_BACKEND`), so the browser only talks to one origin.

## Architecture rules encoded in this repo

- **The LLM never sees raw time-series.** Layer 1 (worker math) → Layer 2 (zone
  attribution, whitelist-only) → Layer 3 (dense Heuristic JSON) → Layer 4
  (OpenAI with schema-validated structured output).
- **Consent is a hard gate at both ends.** The SDK collects nothing without
  affirmative consent and honors DNT/Global Privacy Control; the server rejects
  non-consented payloads with 403. Revocation drops the in-flight buffer.
- **No PII by construction.** The collector captures kinematics and whitelisted
  semantic zone names — never DOM text, keystrokes content, or identifiers.
- **Segments, not verdicts.** The profiler emits behavioral segment signals
  with evidence + confidence. The Big Five vector is a *soft simulation seed*
  derived by transparent rules in `persona.py` with wide confidence bands —
  explicitly not a psychometric claim about a person.
- **Cognitive state via temperature + memory throttling.** Twins model
  fatigue/distraction three ways: persona-state conditioning in the prompt,
  sampling **temperature** raised with load (low 0.45 → high 1.15), and a
  rolling memory window that shrinks to 2 steps under high load in multi-step
  walkthroughs.
- **Swarm = fan-out/fan-in.** Twins are isolated; no inter-agent chatter. The
  prompt is layered `[frozen preamble][scenario][persona]` so OpenAI's
  automatic prompt caching absorbs almost all input cost at scale.

## Models

Backend: **OpenAI** (structured outputs / strict `json_schema`). Set
`OPENAI_API_KEY` to run the LLM paths.

| Role | Default | Env override |
|---|---|---|
| Layer-4 profiler + aggregation | `gpt-4o` | `COGNISWARM_PROFILER_MODEL` |
| Swarm twins | `gpt-4o-mini` | `COGNISWARM_TWIN_MODEL` |

## API surface

| Endpoint | What it does |
|---|---|
| `POST /v1/ingest` | Consent-gated telemetry (public — end-user browsers); accepts an optional `panel_token` |
| `GET /v1/sessions` · `POST /v1/sessions/{id}/profile` | Segments + Layer-4 profiling |
| `POST /v1/swarm/run` | One scenario across the twin swarm |
| `POST /v1/swarm/compare` | Zero-shot A/B: 2–8 variants ranked, lift vs control |
| `POST /v1/swarm/walk` | Multi-step flow walkthrough — twins' memory window shrinks with cognitive load (20/8/2 steps) |
| `GET /v1/swarm/runs?kind=` | Run history (swarm / compare / walk) |
| `POST /v1/runs/{id}/actuals` · `GET /v1/validation/report` | Record real outcomes; MAE/bias calibration |
| `POST /v1/studies/price` | Price Sensitivity: demand + revenue curves across candidate prices, revenue-maximizing point |
| `POST /v1/studies/objection` | Objection Radar: ranked objections, dealbreaker rate, sentiment split for a pitch |
| `POST /v1/jobs` · `GET /v1/jobs[/{id}]` | Durable background jobs; `kind: batch_swarm` runs via the OpenAI Batch API (50% cost) |
| `POST /v1/panel/members` · `GET /v1/panel/members` | Provision panel invites; list members (tokens never listed) |
| `GET /panel/{token}` · `POST /v1/panel/{token}/{consent,revoke}` | Member disclosure page; consent; revoke + erase |

**Headless access:** the analysis surface **fails closed**. Set
`COGNISWARM_API_KEYS=key1,key2` on the server to require `X-API-Key`, or
`COGNISWARM_ALLOW_ANONYMOUS=1` to opt out explicitly for local development;
with neither, every `/v1/*` endpoint returns **503**. Configured keys always
win, so the anonymous escape hatch can never weaken a server that has keys.
Ingest and the member-facing panel consent/revoke routes stay public — consent
is their gate. Give the dashboard its key via `COGNISWARM_API_KEY`; the key is
injected server-side by the session-gated forwarder, never exposed to the
browser.

## Scale & durability (Phase 3)

- **Job queue** ([jobs.py](server/app/jobs.py)): swarm/compare/walk/batch runs
  persist in SQLite and drain on a background worker. On restart, queued *and*
  crash-orphaned `running` jobs are re-enqueued; execution is idempotent. The
  seams (per-kind async executors, full lifecycle in storage) are exactly what a
  Temporal migration needs — swap the queue, keep the executors.
- **Batch swarms** ([batch_swarm.py](server/app/batch_swarm.py)): large
  non-interactive runs go through the OpenAI **Batch API** (file-based JSONL) at half
  the token price, sharing the same prompt-cached layering as the live fan-out.

## Research panel (Phase 4) — the only defensible cross-site model

Not a covert extension: a **disclosed, opt-in, compensated** panel.

- Admin provisions an invite (`/panel` page) → member gets a private
  capability-URL disclosure page listing exactly what is and isn't collected.
- Telemetry is accepted **only** for members who have actively consented;
  the collector passes `panelToken`, the server verifies an active,
  non-revoked consent record before storing anything.
- **Right to erasure is built in:** revocation permanently deletes every
  session ever linked to the member's pseudonymous panel ID — verified by test
  and end-to-end. No archive, no grace period.

## Phase roadmap

1. **Ingest & Profile** ✅ — SDK, pipeline, profiler, dashboard.
2. **Experiments** ✅ — zero-shot A/B compare, context-throttled flow
   walkthrough, calibration harness (actuals → MAE/bias), persona detail view.
3. **Scale & durability** ✅ — durable job queue, Batch API swarms.
4. **Opt-in research panel** ✅ — disclosed, compensated, consent-first, with
   right-to-erasure.

**Next (beyond this repo):** swap the in-process queue for Temporal;
per-segment reliability scores once real outcome data accumulates; Postgres +
ClickHouse for the storage split described in the original blueprint.
