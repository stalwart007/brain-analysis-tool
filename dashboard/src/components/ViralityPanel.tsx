"use client";

/**
 * VIRALITY FORECASTER — Galton-Watson branching process over twin share
 * intents. Live: every twin's share verdict pings in; then the cascade tree
 * grows generation by generation from the REAL simulated quantiles, alongside
 * R0 vs the epidemic threshold, PGF extinction probability, and the
 * final-size distribution. Every number is a model output.
 */

import { useRef, useState } from "react";
import { motion } from "framer-motion";
import { CognitiveLoad, GenerationBand, ViralityResult } from "@/lib/api";
import { streamRun } from "@/lib/stream";
import { LoadToggle } from "./LoadToggle";
import FindingsPanel, { ResearchQuestion, type FindingsBlock } from "./FindingsPanel";
import { useAgentCanvas } from "./sim/useAgentCanvas";
import { ChartCursorProvider, useChartCursor, type ChartCursor } from "./charts/cursor";
import ChartFrame from "./charts/ChartFrame";

/* Types come from lib/api — this panel used to declare its own copies, and
   they had drifted: the local ViralityResult omitted six fields and typed
   `criticality.alpha` as required, which is wrong for the `explosive` regime
   where there is no tail fit at all. */
type Gen = GenerationBand;

const SAMPLE = `POV: you open your laptop Monday morning and an AI already triaged your inbox, drafted the 3 replies that matter, and cancelled the meeting nobody wanted. Atlas. Get your week back. (30-sec demo in comments)`;

/* The provider wraps the panel rather than living inside it: a component
   cannot consume a context it renders itself, so the cascade tree and the
   quantile fan would each get a private cursor and stop linking. */
export default function ViralityPanel({ personaCount }: { personaCount: number }) {
  return (
    <ChartCursorProvider>
      <ViralityInner personaCount={personaCount} />
    </ChartCursorProvider>
  );
}

function ViralityInner({ personaCount }: { personaCount: number }) {
  const [content, setContent] = useState("");
  const [fanout, setFanout] = useState(6);
  const [twins, setTwins] = useState(3);
  const [load, setLoad] = useState<CognitiveLoad>("low");
  const [question, setQuestion] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [shares, setShares] = useState<{ share: number; reason: string }[]>([]);
  const [gens, setGens] = useState<Gen[]>([]);
  const [result, setResult] = useState<ViralityResult | null>(null);

  // Generations are their OWN domain — deliberately not "beats". Linking a
  // cascade generation to a narrative beat would be a false equivalence
  // between two unrelated axes that happen to both be small integers.
  const cursor = useChartCursor(
    "generations",
    gens.length,
    "Cascade size by generation",
    (i) => {
      const g = gens[i];
      if (!g) return `generation ${i + 1}`;
      return `generation ${g.g}: median ${Math.round(g.cum_q50)} cumulative, ${Math.round(g.active_q50)} still active`;
    },
    // The SVG carries a 30/340 label gutter on the left and 8/340 of padding
    // on the right, and it is a line chart, so vertices — not slices.
    { inset: [30 / 340, 8 / 340], mapping: "point" }
  );

  async function run() {
    setBusy(true); setError(null); setResult(null); setShares([]); setGens([]);
    cursor.clear();
    try {
      await streamRun(
        "/studies/virality/stream",
        { content, fanout, twins_per_persona: twins, cognitive_load: load, research_question: question },
        (evt) => {
          if (evt.type === "agent") setShares((s) => [...s, { share: evt.share as number, reason: String(evt.reason) }]);
          else if (evt.type === "generation") setGens((g) => [...g, evt as unknown as Gen]);
          else if (evt.type === "done") setResult(evt.result as unknown as ViralityResult);
          else if (evt.type === "error") setError(String(evt.detail));
        }
      );
    } catch (e) {
      setError(e instanceof Error ? e.message : "Forecast failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="card p-6">
      <div className="mb-1 flex items-baseline justify-between">
        <h2 className="font-display text-lg font-medium tracking-tight">Virality Forecaster</h2>
        <span className="text-xs text-muted">will it spread? branching-process epidemiology for content</span>
      </div>
      <p className="lede mb-4">
        Paste a shareable asset. Twins report their honest share probability;
        a Galton-Watson branching process turns that into R₀, take-off odds,
        and simulated cascades.
      </p>

      <textarea
        value={content} onChange={(e) => setContent(e.target.value)} rows={3}
        placeholder={SAMPLE}
        className="w-full resize-y rounded-xl border border-hairline bg-surface-2 p-3.5 text-sm outline-none placeholder:text-muted focus:border-accent/60 focus:ring-2 focus:ring-accent/20"
      />
      <ResearchQuestion
        value={question}
        onChange={setQuestion}
        disabled={busy}
        examples={["e.g. is this actually shareable, or just likeable?"]}
      />

      <div className="mt-3 flex flex-wrap items-center gap-3">
        <button onClick={() => setContent(SAMPLE)}
          className="rounded-lg border border-dashed border-hairline px-3 py-1.5 text-xs text-muted transition hover:border-accent/50 hover:text-ink-2">
          use sample post
        </button>
        <label className="flex items-center gap-2 text-xs text-muted" title="contacts exposed per sharer (k)">
          Fan-out k
          <input type="number" min={1} max={25} value={fanout}
            onChange={(e) => setFanout(parseInt(e.target.value, 10) || 1)}
            className="w-14 rounded-lg border border-hairline bg-surface-2 px-2 py-1.5 text-sm text-ink outline-none tabular-nums" />
        </label>
        <label className="flex items-center gap-2 text-xs text-muted">
          Twins / persona
          <input type="number" min={1} max={20} value={twins}
            onChange={(e) => setTwins(parseInt(e.target.value, 10) || 1)}
            className="w-14 rounded-lg border border-hairline bg-surface-2 px-2 py-1.5 text-sm text-ink outline-none tabular-nums" />
        </label>
        <LoadToggle value={load} onChange={setLoad} id="virality-load" />
        <motion.button whileTap={{ scale: 0.97 }}
          disabled={busy || content.trim().length < 20}
          onClick={run}
          className="ml-auto rounded-xl bg-accent px-5 py-2 font-display text-sm font-semibold text-white transition hover:brightness-110 disabled:opacity-40">
          {busy ? "Forecasting…" : "Run virality forecast"}
        </motion.button>
      </div>

      {busy && (
        <p className="mt-3 font-mono text-[11px] text-accent">
          <span className="pulse-dot mr-1.5 inline-block h-1.5 w-1.5 rounded-full bg-accent align-middle" />
          {shares.length} twins reported · mean share{" "}
          {shares.length ? (shares.reduce((s, x) => s + x.share, 0) / shares.length).toFixed(2) : "—"}
        </p>
      )}
      {error && <p className="mt-3 text-sm text-critical">{error}</p>}

      {(gens.length > 0 || result) && (
        <div className="mt-4 grid gap-4 lg:grid-cols-2">
          <div className="rounded-xl border border-hairline bg-black/40 p-3">
            <div className="flex items-baseline justify-between">
              <span className="hud-label">CASCADE TREE (median run)</span>
              <span className="font-mono text-[9px] text-muted">gen-by-gen from 2,000 simulations</span>
            </div>
            <div className="h-56"><CascadeCanvas gens={gens} supercritical={result?.supercritical ?? true} activeGen={cursor.index === null ? null : gens[cursor.index]?.g ?? null} /></div>
          </div>
          <div className="rounded-xl border border-hairline bg-black/40 p-3">
            <BandChart gens={gens} cursor={cursor} />
            {result?.final_size_bins && <SizeHistogram bins={result.final_size_bins} />}
          </div>
        </div>
      )}

      {result && (
        <div className="mt-4 space-y-3">
        <FindingsPanel findings={result.findings} />
          <div className="grid gap-3 sm:grid-cols-4">
            <Stat label="R₀ (reproduction)" value={result.r0.toFixed(2)}
              sub={result.r0_ci ? `95% CI ${result.r0_ci[0].toFixed(2)}–${result.r0_ci[1].toFixed(2)}` : ""}
              tone={result.supercritical ? "text-good" : "text-critical"} />
            <Stat label="Take-off probability" value={`${Math.round(result.takeoff_probability * 100)}%`}
              sub={`extinction q ${result.extinction_q_hetero.toFixed(3)}${result.extinction_q_ci ? ` [${result.extinction_q_ci[0].toFixed(2)}–${result.extinction_q_ci[1].toFixed(2)}]` : ""}`}
              tone={result.takeoff_probability > 0.3 ? "text-good" : "text-accent-2"} />
            <Stat label="Expected reach / seed" value={result.expected_cascade_size ? result.expected_cascade_size.toFixed(1) : "∞*"}
              sub={result.expected_cascade_size ? "subcritical closed form 1/(1−R₀)" : "*supercritical — unbounded in-model"} />
            <Stat label="Critical fan-out k*" value={result.critical_fanout?.toFixed(1) ?? "—"}
              sub={`your k=${result.fanout} · share p̄ ${result.share_mean.toFixed(3)}`} />
          </div>
          {/* R0 threshold gauge */}
          <div>
            <div className="relative h-2 w-full rounded-full bg-white/[0.06]">
              <div className="absolute h-2 rounded-full bg-accent/70" style={{ width: `${Math.min(100, (result.r0 / 2) * 100)}%` }} />
              <div className="absolute top-[-3px] h-3.5 w-0.5 bg-critical" style={{ left: "50%" }} title="epidemic threshold R₀=1" />
            </div>
            <div className="mt-0.5 flex justify-between font-mono text-[9px] text-muted">
              <span>0 — dies instantly</span><span className="text-critical">R₀=1 epidemic threshold</span><span>2+</span>
            </div>
          </div>
          {/* Cascade-size tail. `explosive` is not a tail shape — it means a
              material share of simulated cascades never terminated, so their
              sizes are lower bounds and there is nothing honest to fit. It is
              also the most important thing that can happen here, so it gets
              the loud treatment rather than a line of grey mono. */}
          {result.criticality?.regime === "explosive" ||
          result.criticality?.regime === "unresolved" ? (
            (() => {
              // Both regimes mean "no honest tail fit", but for opposite
              // reasons, and saying the wrong one is a material misreport:
              // explosive = outgrew the size cap, unresolved = still spreading
              // at the horizon. They used to collapse into one because the
              // censoring causes were summed.
              const explosive = result.criticality?.regime === "explosive";
              const size = result.simulated_size;
              const bound = explosive ? size?.capped_frac : size?.horizon_frac;
              return (
                <div
                  className={
                    explosive
                      ? "border border-critical/40 bg-critical/[0.07] p-3"
                      : "border border-white/15 bg-white/[0.03] p-3"
                  }
                >
                  <div
                    className={`hud-label ${explosive ? "text-critical" : "text-ink-2"}`}
                  >
                    {explosive ? "Unbounded growth" : "Tail unresolved at this horizon"}
                  </div>
                  <p className="mt-1 text-[11px] leading-relaxed text-ink-2">
                    {result.criticality?.interpretation}
                  </p>
                  {size?.median != null && (
                    <p className="mt-1.5 font-mono text-[10px] text-muted">
                      simulated median {size.median.toLocaleString()} · p90{" "}
                      {size.p90?.toLocaleString()}
                      {bound != null && (
                        <>
                          {" · "}
                          {Math.round(bound * 100)}%{" "}
                          {explosive ? "hit the size cap" : "still spreading"}
                        </>
                      )}
                    </p>
                  )}
                </div>
              );
            })()
          ) : (
            result.criticality &&
            (() => {
              const c = result.criticality;
              // α is a parameter of a model that may have been REJECTED. Printing
              // it in the same weight either way is how a fitted number for a
              // wrong model gets read as the measured tail index — so a rejected
              // fit is struck through and labelled, not merely accompanied by a
              // p-value the reader has to interpret unaided.
              const rejected = c.power_law_plausible === false;
              return (
                <p className="font-mono text-[10px] text-ink-2">
                  cascade-size tail:{" "}
                  <b className={rejected ? "text-accent-2" : "text-accent"}>{c.regime}</b>
                  {c.alpha != null && (
                    <>
                      {" "}
                      <span
                        className={rejected ? "text-muted line-through" : undefined}
                        title={
                          rejected
                            ? "fitted, but the power law was rejected by its goodness-of-fit test — not a measured tail index"
                            : undefined
                        }
                      >
                        (α {c.alpha.toFixed(2)}
                        {c.alpha_se != null && ` ± ${c.alpha_se.toFixed(2)}`}
                        {c.xmin != null && `, xₘᵢₙ ${c.xmin.toFixed(1)}`})
                      </span>
                    </>
                  )}
                  {c.p_value != null && (
                    <>
                      {" "}· goodness-of-fit p {c.p_value.toFixed(3)}
                      {c.power_law_plausible != null && (
                        <span className={rejected ? "text-accent-2" : "text-good"}>
                          {rejected ? " · power law rejected" : " · power law holds"}
                        </span>
                      )}
                    </>
                  )}{" "}
                  — {c.interpretation}
                </p>
              );
            })()
          )}
          <p className="font-mono text-[10px] text-muted">{result.method}</p>
          <p className="text-[9px] leading-tight text-muted">{result.disclaimer}</p>
        </div>
      )}
    </div>
  );
}

function Stat({ label, value, sub, tone }: { label: string; value: string; sub?: string; tone?: string }) {
  return (
    <div className="rounded-lg border border-hairline bg-surface-2/60 p-2.5">
      <div className="hud-label">{label}</div>
      <div className={`font-display text-2xl font-semibold tabular-nums ${tone ?? "text-ink"}`}>{value}</div>
      {sub && <div className="font-mono text-[9px] text-muted">{sub}</div>}
    </div>
  );
}

/* radial cascade tree animated from real generation quantiles */
function CascadeCanvas({ gens, supercritical, activeGen }: {
  gens: Gen[];
  supercritical: boolean;
  /** the generation the shared cursor is on, so hovering the quantile fan
   *  picks out the same ring in the tree — this is the whole point of a
   *  linked cursor: two views of one cascade, read together */
  activeGen: number | null;
}) {
  const nodes = useRef<{ x: number; y: number; tx: number; ty: number; g: number; born: number }[]>([]);
  const links = useRef<[number, number][]>([]);
  const consumed = useRef(0);
  const ref = useAgentCanvas((ctx, w, h, t, dt) => {
    ctx.clearRect(0, 0, w, h);
    const cx = w / 2, cy = h / 2;
    const maxR = Math.min(w, h) / 2 - 10;
    const G = Math.max(6, gens.length);
    if (consumed.current === 0 && nodes.current.length === 0) {
      nodes.current.push({ x: cx, y: cy, tx: cx, ty: cy, g: 0, born: t });
    }
    while (consumed.current < gens.length) {
      const gen = gens[consumed.current++];
      const count = Math.min(60, Math.max(gen.active_q50 > 0 ? 1 : 0, Math.round(gen.active_q50)));
      const parents = nodes.current.map((n, i) => ({ n, i })).filter((x) => x.n.g === gen.g - 1);
      const r = (gen.g / G) * maxR;
      for (let j = 0; j < count; j++) {
        const ang = (j / Math.max(1, count)) * Math.PI * 2 + gen.g * 0.6;
        const idx = nodes.current.length;
        nodes.current.push({ x: cx, y: cy, tx: cx + Math.cos(ang) * r, ty: cy + Math.sin(ang) * r, g: gen.g, born: t });
        if (parents.length) links.current.push([parents[j % parents.length].i, idx]);
      }
    }
    ctx.strokeStyle = activeGen === null ? "rgba(120,195,255,0.18)" : "rgba(120,195,255,0.08)";
    for (const [a, b] of links.current) {
      const na = nodes.current[a], nb = nodes.current[b];
      ctx.beginPath(); ctx.moveTo(na.x, na.y); ctx.lineTo(nb.x, nb.y); ctx.stroke();
    }
    // the cursor's generation as a ring, drawn under the nodes
    if (activeGen !== null && activeGen > 0) {
      ctx.beginPath();
      ctx.arc(cx, cy, (activeGen / G) * maxR, 0, Math.PI * 2);
      ctx.strokeStyle = "rgb(var(--region-rgb) / 0.55)";
      ctx.lineWidth = 1.2;
      ctx.stroke();
    }
    for (const n of nodes.current) {
      n.x += (n.tx - n.x) * Math.min(1, dt * 5);
      n.y += (n.ty - n.y) * Math.min(1, dt * 5);
      const age = Math.min(1, (t - n.born) * 2);
      // Dim everything that is not the cursor's generation. Without this the
      // tree is a pretty texture; with it, it answers "which ring is that?"
      const dim = activeGen !== null && n.g !== activeGen ? 0.22 : 1;
      ctx.globalAlpha = dim;
      ctx.fillStyle = n.g === 0 ? "#f2ad1f" : supercritical ? `rgba(22,208,22,${0.35 + age * 0.5})` : `rgba(120,195,255,${0.3 + age * 0.5})`;
      ctx.beginPath(); ctx.arc(n.x, n.y, n.g === 0 ? 4 : 2.2, 0, Math.PI * 2); ctx.fill();
    }
    ctx.globalAlpha = 1;
    if (nodes.current.length > 900) { nodes.current.splice(1, 200); links.current = []; }
    // Repaint on a new generation as well as a cursor move: under reduced
    // motion the loop stops after one frame, which lands before any generation
    // has streamed in, so without this the tree stays empty for good.
  }, `${gens.length}:${activeGen}`);
  return <canvas ref={ref} className="h-full w-full" />;
}

/* log-scale quantile fan of cumulative cascade size */
function BandChart({ gens, cursor }: { gens: Gen[]; cursor: ChartCursor }) {
  if (gens.length < 2) return null;
  const W = 340, H = 120, pad = 30;
  const maxC = Math.max(...gens.map((g) => g.cum_q90), 2);
  const lg = (v: number) => Math.log10(Math.max(1, v));
  const x = (i: number) => pad + (i / (gens.length - 1)) * (W - pad - 8);
  const y = (v: number) => H - 18 - (lg(v) / lg(maxC)) * (H - 30);
  const line = (key: "cum_q10" | "cum_q50" | "cum_q90") => gens.map((g, i) => `${x(i)},${y(g[key])}`).join(" ");
  const i = cursor.index;
  return (
    <ChartFrame
      cursor={cursor}
      title="CASCADE SIZE — QUANTILE FAN"
      height="auto"
      categories={gens.map((g) => `Gen ${g.g}`)}
      series={[
        {
          key: "cum",
          label: "cumulative",
          color: "#4fb6ff",
          values: gens.map((g) => g.cum_q50),
          // The fan IS the interval — showing the median without q10–q90 is
          // the specific misreading this chart exists to prevent.
          intervals: gens.map((g) => [g.cum_q10, g.cum_q90] as [number, number]),
          format: (v) => Math.round(v).toLocaleString(),
        },
        {
          key: "active",
          label: "still active",
          color: "#16d016",
          values: gens.map((g) => g.active_q50),
          format: (v) => Math.round(v).toLocaleString(),
        },
        {
          key: "alive",
          label: "runs alive",
          color: "#f2ad1f",
          values: gens.map((g) => g.alive_frac),
          format: (v) => `${(v * 100).toFixed(0)}%`,
        },
      ]}
      footnote="Median cumulative reach with the 10th–90th percentile fan, over 2,000 simulated cascades."
    >
      <svg viewBox={`0 0 ${W} ${H}`} className="mt-1 w-full">
        {[1, 10, 100, 1000, 10000].filter((v) => v <= maxC * 1.5).map((v) => (
          <g key={v}>
            <line x1={pad} y1={y(v)} x2={W - 8} y2={y(v)} stroke="rgba(120,165,230,0.08)" />
            <text x={pad - 3} y={y(v) + 2.5} fontSize="6.5" fill="#5f6b7d" textAnchor="end" fontFamily="ui-monospace, monospace">{v}</text>
          </g>
        ))}
        <path d={`M ${line("cum_q90").split(" ").join(" L ")} L ${[...gens].reverse().map((g, ri) => `${x(gens.length - 1 - ri)},${y(g.cum_q10)}`).join(" L ")} Z`}
          fill="rgba(79,182,255,0.12)" />
        <polyline points={line("cum_q50")} fill="none" stroke="#4fb6ff" strokeWidth="1.6" />
        {/* the cursor's generation, marked on all three quantiles at once, so
            hovering reads the spread and not just the median */}
        {i !== null && gens[i] && (
          <g>
            <line x1={x(i)} y1={y(gens[i].cum_q10)} x2={x(i)} y2={y(gens[i].cum_q90)}
              stroke={cursor.isPinned(i) ? "rgb(var(--region-rgb))" : "rgba(255,255,255,0.5)"} strokeWidth="0.8" />
            {(["cum_q10", "cum_q90"] as const).map((k) => (
              <circle key={k} cx={x(i)} cy={y(gens[i][k])} r="1.4" fill="rgba(255,255,255,0.55)" />
            ))}
            <circle cx={x(i)} cy={y(gens[i].cum_q50)} r="2.6"
              fill={cursor.isPinned(i) ? "rgb(var(--region-rgb))" : "#4fb6ff"}
              stroke="rgba(0,0,0,0.6)" strokeWidth="0.8" />
          </g>
        )}
        <text x={W - 8} y={H - 6} fontSize="6.5" fill="#5f6b7d" textAnchor="end" fontFamily="ui-monospace, monospace">generation → · size (log)</text>
      </svg>
    </ChartFrame>
  );
}

/* log-binned final-size histogram — the heavy-tail fingerprint */
function SizeHistogram({ bins }: { bins: [number, number, number][] }) {
  // Its own domain. Final-size bins are not generations — bin 3 and generation
  // 3 have nothing to do with each other, and linking them would invent a
  // relationship the model never claims.
  const cursor = useChartCursor(
    "cascade-size",
    bins.length,
    "Final cascade size distribution",
    (i) => {
      const b = bins[i];
      return b ? `sizes ${b[0]} to ${b[1]}: ${b[2]} of 2,000 cascades` : `bin ${i + 1}`;
    }
  );
  if (!bins.length) return null;
  const maxC = Math.max(...bins.map((b) => b[2]));
  const total = bins.reduce((sum, b) => sum + b[2], 0) || 1;
  return (
    <div className="mt-2">
      <ChartFrame
        cursor={cursor}
        title="FINAL SIZE — LOG₂ BINS"
        height={48}
        categories={bins.map(([lo, hi]) => `${lo}–${hi}`)}
        series={[
          {
            key: "count",
            label: "cascades",
            color: "rgb(var(--region-rgb))",
            values: bins.map((b) => b[2]),
            format: (v) => `${v.toLocaleString()} (${((v / total) * 100).toFixed(1)}%)`,
          },
        ]}
        footnote="Height is log-scaled; a fat right tail is the signature of a supercritical process."
      >
        <div className="flex h-full items-end gap-0.5">
          {bins.map(([, , c], i) => (
            <div key={i} className="flex-1 rounded-t-sm transition-[opacity,background-color] duration-150"
              style={{
                height: `${8 + (Math.log10(1 + c) / Math.log10(1 + maxC)) * 88}%`,
                background: cursor.isPinned(i) ? "rgb(var(--region-rgb))" : "var(--accent, #4fb6ff)",
                // Dimming the rest is what makes one bar readable; the audit
                // found every chart here rendered all bars identically lit.
                opacity: cursor.index === null ? 0.75 : cursor.isMarked(i) ? 1 : 0.28,
              }} />
          ))}
        </div>
      </ChartFrame>
    </div>
  );
}
