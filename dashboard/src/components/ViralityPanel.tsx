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
import { useAgentCanvas } from "./sim/useAgentCanvas";

/* Types come from lib/api — this panel used to declare its own copies, and
   they had drifted: the local ViralityResult omitted six fields and typed
   `criticality.alpha` as required, which is wrong for the `explosive` regime
   where there is no tail fit at all. */
type Gen = GenerationBand;

const SAMPLE = `POV: you open your laptop Monday morning and an AI already triaged your inbox, drafted the 3 replies that matter, and cancelled the meeting nobody wanted. Atlas. Get your week back. (30-sec demo in comments)`;

export default function ViralityPanel({ personaCount }: { personaCount: number }) {
  const [content, setContent] = useState("");
  const [fanout, setFanout] = useState(6);
  const [twins, setTwins] = useState(3);
  const [load, setLoad] = useState<CognitiveLoad>("low");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [shares, setShares] = useState<{ share: number; reason: string }[]>([]);
  const [gens, setGens] = useState<Gen[]>([]);
  const [result, setResult] = useState<ViralityResult | null>(null);

  async function run() {
    setBusy(true); setError(null); setResult(null); setShares([]); setGens([]);
    try {
      await streamRun(
        "/studies/virality/stream",
        { content, fanout, twins_per_persona: twins, cognitive_load: load },
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
            <div className="h-56"><CascadeCanvas gens={gens} supercritical={result?.supercritical ?? true} /></div>
          </div>
          <div className="rounded-xl border border-hairline bg-black/40 p-3">
            <span className="hud-label">CASCADE SIZE — QUANTILE FAN</span>
            <BandChart gens={gens} />
            {result?.final_size_bins && <SizeHistogram bins={result.final_size_bins} />}
          </div>
        </div>
      )}

      {result && (
        <div className="mt-4 space-y-3">
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
            result.criticality && (
              <p className="font-mono text-[10px] text-ink-2">
                cascade-size tail:{" "}
                <b className="text-accent">{result.criticality.regime}</b>
                {result.criticality.alpha != null && (
                  <>
                    {" "}(α {result.criticality.alpha.toFixed(2)}
                    {result.criticality.alpha_se != null &&
                      ` ± ${result.criticality.alpha_se.toFixed(2)}`}
                    {result.criticality.xmin != null &&
                      `, xₘᵢₙ ${result.criticality.xmin.toFixed(1)}`}
                    )
                  </>
                )}
                {result.criticality.p_value != null &&
                  ` · p ${result.criticality.p_value.toFixed(3)}`}{" "}
                — {result.criticality.interpretation}
              </p>
            )
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
function CascadeCanvas({ gens, supercritical }: { gens: Gen[]; supercritical: boolean }) {
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
    ctx.strokeStyle = "rgba(120,195,255,0.18)";
    for (const [a, b] of links.current) {
      const na = nodes.current[a], nb = nodes.current[b];
      ctx.beginPath(); ctx.moveTo(na.x, na.y); ctx.lineTo(nb.x, nb.y); ctx.stroke();
    }
    for (const n of nodes.current) {
      n.x += (n.tx - n.x) * Math.min(1, dt * 5);
      n.y += (n.ty - n.y) * Math.min(1, dt * 5);
      const age = Math.min(1, (t - n.born) * 2);
      ctx.fillStyle = n.g === 0 ? "#f2ad1f" : supercritical ? `rgba(22,208,22,${0.35 + age * 0.5})` : `rgba(120,195,255,${0.3 + age * 0.5})`;
      ctx.beginPath(); ctx.arc(n.x, n.y, n.g === 0 ? 4 : 2.2, 0, Math.PI * 2); ctx.fill();
    }
    if (nodes.current.length > 900) { nodes.current.splice(1, 200); links.current = []; }
  });
  return <canvas ref={ref} className="h-full w-full" />;
}

/* log-scale quantile fan of cumulative cascade size */
function BandChart({ gens }: { gens: Gen[] }) {
  if (gens.length < 2) return null;
  const W = 340, H = 120, pad = 30;
  const maxC = Math.max(...gens.map((g) => g.cum_q90), 2);
  const lg = (v: number) => Math.log10(Math.max(1, v));
  const x = (i: number) => pad + (i / (gens.length - 1)) * (W - pad - 8);
  const y = (v: number) => H - 18 - (lg(v) / lg(maxC)) * (H - 30);
  const line = (key: "cum_q10" | "cum_q50" | "cum_q90") => gens.map((g, i) => `${x(i)},${y(g[key])}`).join(" ");
  return (
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
      <text x={W - 8} y={H - 6} fontSize="6.5" fill="#5f6b7d" textAnchor="end" fontFamily="ui-monospace, monospace">generation → · size (log)</text>
    </svg>
  );
}

/* log-binned final-size histogram — the heavy-tail fingerprint */
function SizeHistogram({ bins }: { bins: [number, number, number][] }) {
  if (!bins.length) return null;
  const maxC = Math.max(...bins.map((b) => b[2]));
  return (
    <div className="mt-2">
      <div className="flex h-12 items-end gap-0.5">
        {bins.map(([lo, , c], i) => (
          <div key={i} className="flex-1 rounded-t-sm bg-accent"
            style={{ height: `${8 + (Math.log10(1 + c) / Math.log10(1 + maxC)) * 88}%`, opacity: 0.75 }}
            title={`size ${lo}–${lo * 2}: ${c} of 2,000 cascades`} />
        ))}
      </div>
      <div className="mt-0.5 font-mono text-[8px] text-muted">final cascade size (log₂ bins) · height log-scaled</div>
    </div>
  );
}
