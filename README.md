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
   cognitive load, **Forecast response** → aggregated engagement/intent/drop-off/friction.

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
- **The White Room is the one exception to isolation, and that is the point.**
  Every other study fans out to isolated twins so the spread you read is genuine
  disagreement rather than a herd. The room lets N characters hear each other and
  update, which makes the isolated condition a *control arm* rather than a
  different product: same cast, two conditions, and the delta is what
  deliberation did. It reports whose opinion the consensus actually weights (the
  Perron vector of a fitted Friedkin-Johnsen influence matrix — structural
  position, not correctness), what the room lands on if each member is removed,
  and whether it converged on information or on each other (observed variance
  collapse against a Bayesian pooling benchmark). Speaking order is randomised
  across replicates because it is a confound, not a setting.
- **A room also runs a placebo arm, and it can veto the study.** Some replicates
  show each member statements from a *different* room. A language model is
  agreeable, so convergence may be its disposition rather than these characters;
  if the scrambled arm converges as hard, the deliberation measured nothing, and
  the findings layer *withholds* every influence and counterfactual row rather
  than publishing them with a caveat. A caveat is prose a model may drop; a
  missing evidence id is a citation it cannot make.
- **The room's own estimators were calibrated against rooms with a known
  influence matrix, and two headline numbers came back weaker than they look.**
  A *single* influence weight is never quotable: an individual entry only beats a
  uniform guess above 9 observations per parameter, and the largest room the
  schema allows (9 seats × 8 rounds × 8 replicates) reaches 6.22 — so
  `w_entries_supported` ships False on essentially every run and the network is
  drawn as a pattern rather than labelled. What *does* recover is everything
  derived from the whole matrix: the settled position lands at MAE 0.019 against
  a 0.085 baseline with rank correlation 0.96–0.99, which is why the
  counterfactuals are trustworthy while the individual edges behind them are not.
  Centrality is the weakest: it names the single most central member correctly
  31–38% of the time at default settings against a 20% chance rate, rising to
  49–57% at 8 rounds × 8 replicates. Read it as a ranking with real uncertainty.
- **Two of the permutation tests have combinatorial floors, and the defaults
  moved because of them.** A permutation p cannot go below 2/(arrangements): the
  public/private gap floors at 2/2^seats, so a 5-seat room cannot reach p < 0.05
  however large the gap, and the speaking-order test floors at 2/replicates!, so
  3 replicates floor at 0.333 — that control could never fire. The default is now
  5 replicates (floor 0.017, and 2.5 observations per parameter instead of the
  1.50 that sat exactly on the influence fit's refusal boundary). Replicates are
  the cheap axis: each buys `rounds − 1` fresh observations per member, and
  conformity power goes 48% → 91% from 3 to 8 replicates while doubling the
  rounds changes nothing. Both floors are reported beside their p-values.
- **What the room does NOT measure: preference falsification.** Members state a
  public position and a private one, and the gap was meant to be the headline —
  the Abilene paradox, quantified. Measured on a cast built so dissent would be
  costly, mean |public − private| was **0.013**, and **0.022** after every member
  was also told exactly who outranked them; public and private were *identical*
  in 13 of 16 turns, against a reporting granularity of about 0.05. Both fields
  come from one forward pass by a model that can see it is writing both, and its
  consistency prior beats any social pressure the scenario supplies. Instructing
  members to understate dissent would manufacture the exact phenomenon the
  instrument exists to detect, so the gap is reported, flagged below its
  resolution floor, and refused as a finding.
- **Swarm = fan-out/fan-in.** Twins are isolated; no inter-agent chatter. The
  prompt is layered `[frozen preamble][scenario][persona]` so OpenAI's
  automatic prompt caching absorbs almost all input cost at scale.
- **Findings are harvested, not written.** Every study ends in
  [findings.py](server/app/findings.py): quantitative claims are extracted from
  the study's own statistics by *code*, the whole family is Benjamini-Hochberg
  corrected, and only then does a model write — constrained to **cite evidence
  ids**, with uncited findings dropped before anyone sees them. The model
  supplies judgement about what matters; it is structurally unable to invent a
  measurement. Uncorrected, 40 all-null hypotheses produce a "finding" in 86.8%
  of studies; corrected, 7.2%.
- **The research question never reaches the twins.** `research_question` ranks
  the evidence and aims the answer. Telling a synthetic audience what you hope
  to find produces an audience that finds it — a language model is far more
  agreeable than a person — so the measurements are bit-identical whether or
  not a question was asked. Only the emphasis changes.
- **Bands are simultaneous, not pointwise.** A neuro study draws up to 90
  intervals; pointwise 95% intervals cover the *whole curve* only 50–73% of the
  time. Curve bands are studentized sup-t (measured 97–99%).
- **A "winner" is measured on data that did not pick it.** Selecting the best of
  N variants and then reporting its margin over the baseline on the same sample
  measures the search's luck: under a true null the copy optimizer's lift read
  **+0.087 and was positive in 99.3% of runs**. Both the optimizer and the
  sequence study now re-run the selection inside each split and score on the
  held-out half (null lift −0.0003), and they publish the naive number beside
  the honest one so the size of the correction is visible rather than hidden.
- **Verdicts are gated on intervals, and only on intervals that don't need an
  assumption.** The calibration report's slope is not identified without knowing
  which of predicted/actual is the noisier measurement, and actuals arrive with
  no trial counts — so it brackets the slope over *every* error ratio and asserts
  "compressed" or "exaggerated" only when the whole bracket agrees. Picking one
  assumption instead produced a confident wrong adjective in 44.5% of honest runs
  at n = 40, and the rate *grew* with sample size.
- **Every estimator is calibrated against a null with a known answer, and the
  measurement is in the docstring.** That is how the errors above were found, and
  the same pass caught a change-point detector whose false-positive rate ranged
  from 3% to 63% depending only on how many twins ran (now 3.8–7.0%, flat), a
  power-law fit reading α = 1.58 where theory says exactly 1.5 (now 1.4978 ±
  0.0043, with a goodness-of-fit test that can reject it), and a "95% CI" that
  was the range of two points. See [docs/AUDIT-BACKLOG.md](docs/AUDIT-BACKLOG.md)
  — including what is still open.

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
| `POST /v1/content/fetch` | Paste a **link** of any kind — page, image, video, audio, PDF. Routed by content type (never by file extension), SSRF-guarded, returns a study-ready asset |
| `POST /v1/content/media` | Relays video/audio bytes to the browser, which decodes them — no media decoder ever runs in the API process |
| `POST /v1/studies/price` | Price Sensitivity: demand + revenue curves across candidate prices, revenue-maximizing point |
| `POST /v1/studies/objection` | Objection Radar: ranked objections, dealbreaker rate, sentiment split for a pitch |
| `POST /v1/room/cast` | White Room: describe characters in prose, get a cast back to approve before spending a run |
| `POST /v1/room/stream` | Convene the room. The one study where agents hear each other — see the note below |
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
