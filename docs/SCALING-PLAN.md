# CogniSwarm — Scaling & Cost Plan

Target: from today's 18–30 agents per simulation to **1,000–10,000+ agents per
simulation**, thousands of customers. All figures USD, 2026 pricing.

---

## 1. Unit economics at swarm scale

One twin call ≈ 1,200 tokens in / 250 out.

| Volume | gpt-4o-mini (API) | API + Batch (-50%) | Self-hosted 8B model* |
|---|---|---|---|
| 1 twin call | $0.00033 | $0.00017 | ~$0.00002 |
| 1,000-twin simulation | $0.33 | $0.17 | ~$0.02 |
| 10,000-twin simulation | $3.30 | $1.65 | ~$0.20 |
| 100M calls / month (platform at scale) | ~$33k/mo | ~$16k/mo | ~$2–4k/mo GPU rental |

\* Llama/Qwen-class 8–14B on rented H100s with vLLM, continuous batching.

Cognitive-math engine (DDM, HMM, spectral, …) is **pure CPU — effectively
free at any scale**. The LLM swarm is the only real COGS.

## 2. LLM strategy: API vs local model

**Now (≤ ~$10k/mo LLM spend): stay on API.**
- Batch API halves cost; prompt caching discounts the shared persona prefix.
- OpenAI tier-5 limits (30k RPM on mini models) already allow 1,000-agent
  bursts. Provider choice (OpenAI / Anthropic Haiku / Gemini Flash) matters
  far less than architecture — build a thin provider-abstraction layer so
  switching is a config change, and benchmark twins quarterly.

**At scale (> ~$10–15k/mo sustained): move the twins to a self-hosted open
model. Keep the profiler on a frontier API.**
- Twins are high-volume, low-difficulty (persona roleplay + strict JSON) —
  an 8–14B open model matches mini-class quality here.
- One rented H100 node (~$2–3/hr ≈ $2k/mo) serves ~3–7M twin calls/day.
  Two nodes + failover ≈ $5–6k/mo replaces a $30k+/mo API bill.
- Hidden cost: one ML-infra engineer (~$200k/yr) + eval harness. That's why
  the crossover is ~$10–15k/mo, not earlier.
- Profiling / calibration / anything low-volume & judgment-heavy stays on a
  frontier API (gpt-4o / Claude class) — quality matters more than cost there.
- **Rent GPUs, don't buy.** A 2×H100 server is $70–90k capex; rental keeps
  optionality while models and prices keep dropping.

## 3. Engineering roadmap for 1,000–10,000-agent swarms

Provider-independent work, roughly one quarter for two backend engineers:

1. SQLite → Postgres; telemetry analytics → ClickHouse.
2. In-process job queue → real queue (SQS/Redpanda) + horizontal worker pool
   (today's per-run `SWARM_CONCURRENCY=8` becomes a global, per-tenant cap).
3. SSE aggregation: at 10k agents, stream rolling summary frames
   (~10 frames/sec of counts/means), not one frame per agent — the browser
   viz already renders from aggregates.
4. Per-tenant auth, rate limits, and **cost guardrails** (token budget per
   run/customer — a runaway 100k-twin request must be refused, not billed).
5. EU data region + right-to-erasure across replicas (compliance is the
   product: consented behavioral telemetry).

## 4. Cost by stage

### Stage 1 — Prove it sells (months 0–6, 10–20 customers)
| | |
|---|---|
| 2 VPS + managed Postgres + CDN | ~$150/mo |
| OpenAI (now with 1k-twin sims) | $100–800/mo |
| Tools (domain, Stripe, Sentry, GitHub) | ~$50/mo |
| Privacy legal review (one-time, non-optional) | $5–15k |
| **Run-rate** | **≈ $0.5–1.5k/mo + ~$10k one-time** |

### Stage 2 — Real SaaS (months 6–24, 50–300 customers)
| | |
|---|---|
| Team of ~9 (2 BE, 1 FE, 1 ML/cog-sci, 1 SRE, design, sales, CS, founder) | $1.3–1.7M/yr |
| Cloud (k8s, Postgres, Redis, queue, ClickHouse, observability) | $2–6k/mo |
| LLM API (Batch) | $2–10k/mo |
| SaaS stack (auth, billing, CRM, analytics…) | $1.5–3k/mo |
| SOC 2 Type II + pen test + privacy counsel + insurance | $90–190k/yr |
| **Total** | **≈ $1.6–2.2M/yr** (seed round: $2.5–4M) |

### Stage 3 — Big scale (year 2+, 1,000+ customers, 10k-agent swarms)
| | |
|---|---|
| Team ~22–30 (+data eng, security, sales org, CS, cog-sci advisor) | $4–5.5M/yr |
| Infra (ingest ~10k events/s, Kafka, ClickHouse cluster, EU region) | $15–45k/mo AWS *or* $5–15k/mo bare-metal |
| LLM: self-hosted twins (2–4 H100 nodes) + frontier API for profiling | $8–15k/mo |
| **Total** | **≈ $5.5–8M/yr** (Series A, against $3–10M ARR) |

Global-remote hiring (India / E. Europe) cuts every people line 50–70%.

## 5. Order of operations

1. Postgres + queue + cost guardrails (unlocks 1k-twin sims safely on API).
2. Sell. Nothing below matters until Stage 1 converts.
3. SOC 2 + EU region when the first enterprise deal asks.
4. Self-host twins only when the API bill sustains > $10–15k/mo.
5. Buy hardware never — rent until finance says otherwise.

**Margin picture:** even fully API-based, a $99/mo customer running fifty
1,000-twin simulations costs ~$9 to serve → ~90% gross margin; self-hosting
at scale pushes serving cost toward ~$1.
