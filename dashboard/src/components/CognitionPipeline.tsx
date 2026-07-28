"use client";

/**
 * THE COMPUTATION, VISUALIZED. When a session is profiled, the backend streams
 * every modeling stage over SSE and this component renders the actual work:
 * the EM algorithm converging, BIC choosing the state count, the drift-
 * diffusion process re-simulated from *fitted* parameters, the session's
 * decoded cognitive-state timeline, the transition structure, spectral bands,
 * foraging regime. Nothing decorative — every pixel is a model output.
 */

import { useEffect, useRef, useState } from "react";
import {
  CognitionProfile,
  DDMResult,
  HMMResult,
  InfoDynamics,
  PersonaSeed,
} from "@/lib/api";
import { streamRun, StreamEvent } from "@/lib/stream";
import { useAgentCanvas } from "./sim/useAgentCanvas";

const STAGES = [
  ["ddm", "DRIFT-DIFFUSION"],
  ["hmm", "HIDDEN MARKOV"],
  ["information", "INFO DYNAMICS"],
  ["spectral", "SPECTRAL"],
  ["foraging", "FORAGING"],
  ["habituation", "HABITUATION"],
  ["signal", "LLM · SIGNAL"],
  ["persona", "LLM · PERSONA"],
] as const;

const STATE_COLORS = ["#3f8ff0", "#f2ad1f", "#f0605f", "#16d016", "#b07ff0"];

interface ConsoleLine {
  stage: string;
  text: string;
}

export default function CognitionPipeline({
  sessionId,
  stored,
  onDone,
}: {
  sessionId: string;
  stored?: CognitionProfile | null; // render a saved profile without streaming
  onDone?: () => void;
}) {
  const [status, setStatus] = useState<Record<string, string>>({});
  const [lines, setLines] = useState<ConsoleLine[]>([]);
  const [profile, setProfile] = useState<CognitionProfile | null>(stored ?? null);
  const [persona, setPersona] = useState<PersonaSeed | null>(null);
  const [error, setError] = useState<string | null>(null);
  const started = useRef(false);

  useEffect(() => {
    if (stored || started.current) return;
    started.current = true;
    const log = (stage: string, text: string) =>
      setLines((ls) => [...ls.slice(-40), { stage, text }]);

    streamRun(`/sessions/${sessionId}/profile/stream`, {}, (evt: StreamEvent) => {
      if (evt.type === "cognition") {
        const stage = String(evt.stage);
        if (stage === "complete") {
          setProfile(evt.profile as unknown as CognitionProfile);
          return;
        }
        const st = String(evt.status);
        setStatus((s) => ({ ...s, [stage]: st }));
        if (evt.detail) log(stage, String(evt.detail));
        if (st === "converged" && evt.ll_trace) {
          const t = evt.ll_trace as number[];
          log(stage, `EM log-likelihood ${t[0].toFixed(1)} → ${t[t.length - 1].toFixed(1)}`);
          log(stage, `BIC by K: ${Object.entries(evt.bic_by_k as Record<string, number>).map(([k, b]) => `K=${k}:${b.toFixed(0)}`).join("  ")}`);
        }
        if (st === "done") {
          setStatus((s) => ({ ...s, [stage]: evt.result ? "done" : "skipped" }));
          if (!evt.result) log(stage, "insufficient data — model skipped (honest degradation)");
        }
      } else if (evt.type === "llm") {
        const stage = String(evt.stage);
        setStatus((s) => ({ ...s, [stage]: String(evt.status) }));
        if (evt.detail) log(stage, String(evt.detail));
      } else if (evt.type === "done") {
        const r = evt.result as { persona: PersonaSeed; cognition: CognitionProfile };
        setPersona(r.persona);
        setProfile(r.cognition);
        setStatus((s) => ({ ...s, signal: "done", persona: "done" }));
        onDone?.();
      } else if (evt.type === "error") {
        setError(String(evt.detail));
      }
    });
  }, [sessionId, stored, onDone]);

  return (
    <div className="space-y-4 p-4">
      {/* stage rail */}
      <div className="flex flex-wrap gap-2">
        {STAGES.map(([key, label]) => {
          const st = stored ? "done" : (status[key] ?? "pending");
          const tone =
            st === "done"
              ? "border-good/40 text-good"
              : st === "skipped"
                ? "border-hairline text-muted line-through"
                : st === "pending"
                  ? "border-hairline text-muted"
                  : "border-accent/60 text-accent";
          return (
            <span
              key={key}
              className={`rounded-full border px-2.5 py-1 font-mono text-[10px] tracking-wider ${tone}`}
            >
              {st !== "pending" && st !== "done" && st !== "skipped" && (
                <span className="pulse-dot mr-1 inline-block h-1.5 w-1.5 rounded-full bg-accent align-middle" />
              )}
              {label}
            </span>
          );
        })}
      </div>

      {/* live computation console */}
      {!stored && lines.length > 0 && (
        <div className="max-h-36 overflow-y-auto rounded-lg border border-hairline bg-black/50 p-2 font-mono text-[10px] leading-relaxed">
          {lines.map((l, i) => (
            <div key={i}>
              <span className="text-accent/70">[{l.stage}]</span>{" "}
              <span className="text-ink-2">{l.text}</span>
            </div>
          ))}
        </div>
      )}
      {error && <p className="text-sm text-critical">{error}</p>}

      {profile && (
        <div className="grid gap-4 lg:grid-cols-2">
          {profile.ddm && <DDMPanel ddm={profile.ddm} />}
          {profile.hmm && <HMMPanel hmm={profile.hmm} />}
          {profile.information && <InfoPanel info={profile.information} />}
          <MetricsPanel profile={profile} />
        </div>
      )}

      {persona && (
        <p className="text-sm">
          <span className="hud-label mr-2">PERSONA SEEDED</span>
          <span className="font-display font-semibold text-accent">{persona.label}</span>
        </p>
      )}
      <p className="text-[10px] leading-tight text-muted">
        Behavioral model parameters estimated from consented interaction telemetry —
        not clinical, psychometric, or neuroimaging measures.
      </p>
    </div>
  );
}

/* ── drift-diffusion: fitted parameters re-simulated as live particles ── */

function DDMPanel({ ddm }: { ddm: DDMResult }) {
  const particles = useRef<{ x: number; y: number; done: boolean; hit: number }[]>([]);
  const ref = useAgentCanvas((ctx, w, h, t, dt) => {
    ctx.clearRect(0, 0, w, h);
    const pad = 14;
    const midY = h / 2;
    const bTop = pad + 8;
    const bBot = h - pad - 8;
    // boundaries scaled by fitted a; drift by fitted v (sign flips target)
    ctx.strokeStyle = "rgba(22,208,22,0.5)";
    ctx.beginPath(); ctx.moveTo(pad, bTop); ctx.lineTo(w - pad, bTop); ctx.stroke();
    ctx.strokeStyle = "rgba(240,96,95,0.5)";
    ctx.beginPath(); ctx.moveTo(pad, bBot); ctx.lineTo(w - pad, bBot); ctx.stroke();
    ctx.fillStyle = "#5f6b7d";
    ctx.font = "8px ui-monospace, monospace";
    ctx.textAlign = "left";
    ctx.fillText("engage  +a/2", pad, bTop - 3);
    ctx.fillText("abandon −a/2", pad, bBot + 9);

    if (particles.current.length < 26 && Math.random() < 0.25) {
      particles.current.push({ x: pad, y: midY, done: false, hit: 0 });
    }
    const vScale = ddm.drift_rate * 55; // px/s drift from the fitted rate
    for (const p of particles.current) {
      if (!p.done) {
        p.x += 26 * dt;
        p.y -= vScale * dt + (Math.random() - 0.5) * 4.2; // drift + diffusion
        if (p.y <= bTop || p.y >= bBot || p.x > w - pad) {
          p.done = true;
          p.hit = p.y <= bTop ? 1 : p.y >= bBot ? -1 : 0;
        }
      }
      ctx.fillStyle = p.done
        ? p.hit > 0
          ? "rgba(22,208,22,0.8)"
          : p.hit < 0
            ? "rgba(240,96,95,0.8)"
            : "rgba(120,165,230,0.4)"
        : "rgba(120,195,255,0.85)";
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.done ? 1.6 : 2.2, 0, Math.PI * 2);
      ctx.fill();
    }
    if (particles.current.filter((p) => p.done).length > 40) {
      particles.current = particles.current.filter((p) => !p.done).concat(
        particles.current.filter((p) => p.done).slice(-20)
      );
    }
  });
  return (
    <div className="rounded-xl border border-hairline bg-surface-2/60 p-3">
      <div className="mb-1 flex items-baseline justify-between">
        <span className="hud-label">DRIFT-DIFFUSION DECISION MODEL</span>
        <span className="font-mono text-[10px] text-muted">{ddm.model}</span>
      </div>
      <div className="h-32"><canvas ref={ref} className="h-full w-full" /></div>
      <div className="mt-2 flex flex-wrap gap-4 font-mono text-[10px] tabular-nums text-ink-2">
        <span>drift v <b className="text-accent">{ddm.drift_rate}</b></span>
        <span>boundary a <b className="text-accent">{ddm.boundary_separation}</b></span>
        <span>non-decision Ter <b className="text-accent">{ddm.non_decision_time_s}s</b></span>
      </div>
      <p className="mt-1 text-[11px] leading-tight text-muted">{ddm.interpretation}</p>
    </div>
  );
}

/* ── HMM: decoded cognitive-state ribbon + EM convergence + BIC ───────── */

function HMMPanel({ hmm }: { hmm: HMMResult }) {
  const ll = hmm.ll_trace;
  const llMin = Math.min(...ll);
  const llMax = Math.max(...ll);
  const pts = ll
    .map((v, i) => `${(i / Math.max(1, ll.length - 1)) * 100},${34 - ((v - llMin) / (llMax - llMin + 1e-9)) * 30}`)
    .join(" ");
  // compress the state path into runs for rendering
  const runs: { state: number; len: number }[] = [];
  for (const s of hmm.state_path) {
    const last = runs[runs.length - 1];
    if (last && last.state === s) last.len++;
    else runs.push({ state: s, len: 1 });
  }
  const total = hmm.state_path.length;
  return (
    <div className="rounded-xl border border-hairline bg-surface-2/60 p-3">
      <div className="mb-1 flex items-baseline justify-between">
        <span className="hud-label">DECODED COGNITIVE-STATE TIMELINE</span>
        <span className="font-mono text-[10px] text-muted">
          Baum-Welch EM · Viterbi · BIC picks K={hmm.n_states}
        </span>
      </div>
      {/* the session, decoded */}
      <div className="flex h-7 w-full overflow-hidden rounded-md">
        {runs.map((r, i) => (
          <div
            key={i}
            title={hmm.states[r.state]?.label}
            style={{
              width: `${(r.len / total) * 100}%`,
              background: STATE_COLORS[r.state % STATE_COLORS.length],
              opacity: 0.75,
            }}
          />
        ))}
      </div>
      <div className="mt-1.5 flex flex-wrap gap-3">
        {hmm.states.map((s, i) => (
          <span key={i} className="flex items-center gap-1 font-mono text-[10px] text-ink-2">
            <span
              className="inline-block h-2 w-2 rounded-sm"
              style={{ background: STATE_COLORS[i % STATE_COLORS.length] }}
            />
            {s.label}
          </span>
        ))}
      </div>
      <div className="mt-3 grid grid-cols-2 gap-3">
        <div>
          <div className="font-mono text-[9px] uppercase tracking-wider text-muted">
            EM convergence ({hmm.em_iterations} iters)
          </div>
          <svg viewBox="0 0 100 36" className="mt-1 h-9 w-full">
            <polyline points={pts} fill="none" stroke="#4fb6ff" strokeWidth="1.5" />
          </svg>
        </div>
        <div>
          <div className="font-mono text-[9px] uppercase tracking-wider text-muted">
            BIC by state count
          </div>
          <div className="mt-1 space-y-1">
            {Object.entries(hmm.bic_by_k).map(([k, b]) => {
              const vals = Object.values(hmm.bic_by_k);
              const best = Math.min(...vals);
              const worst = Math.max(...vals);
              const frac = 1 - (b - best) / (worst - best + 1e-9);
              return (
                <div key={k} className="flex items-center gap-1.5 font-mono text-[9px]">
                  <span className={b === best ? "text-good" : "text-muted"}>K={k}</span>
                  <div className="h-1.5 flex-1 rounded bg-white/[0.06]">
                    <div
                      className="h-1.5 rounded"
                      style={{
                        width: `${20 + frac * 80}%`,
                        background: b === best ? "#16d016" : "#3f8ff0",
                        opacity: 0.7,
                      }}
                    />
                  </div>
                  <span className="tabular-nums text-muted">{b.toFixed(0)}</span>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}

/* ── information dynamics: transition heatmap + entropy metrics ───────── */

const SYMBOLS = ["scroll", "dwell", "click", "hesit", "rage", "back", "zone"];

function InfoPanel({ info }: { info: InfoDynamics }) {
  const P = info.transition;
  return (
    <div className="rounded-xl border border-hairline bg-surface-2/60 p-3">
      <div className="mb-1 flex items-baseline justify-between">
        <span className="hud-label">BEHAVIORAL TRANSITION STRUCTURE</span>
        <span className="font-mono text-[10px] text-muted">first-order Markov, Laplace-smoothed</span>
      </div>
      {P && (
        <div
          className="grid gap-px"
          style={{ gridTemplateColumns: `auto repeat(${P.length}, minmax(0,1fr))` }}
        >
          <span />
          {SYMBOLS.slice(0, P.length).map((s) => (
            <span key={s} className="text-center font-mono text-[8px] text-muted">{s}</span>
          ))}
          {P.map((row, i) => {
            const rowMax = Math.max(...row, 1e-9);
            return [
              <span key={`l${i}`} className="pr-1 text-right font-mono text-[8px] leading-4 text-muted">
                {SYMBOLS[i]}
              </span>,
              ...row.map((p, j) => (
                <div
                  key={`${i}-${j}`}
                  title={`P(${SYMBOLS[i]}→${SYMBOLS[j]}) = ${p.toFixed(3)}`}
                  className="h-4 rounded-[2px]"
                  style={{ background: `rgba(79,182,255,${0.06 + (p / rowMax) * 0.75})` }}
                />
              )),
            ];
          })}
        </div>
      )}
      <div className="mt-2 flex flex-wrap gap-4 font-mono text-[10px] tabular-nums text-ink-2">
        {info.entropy_rate_bits != null && (
          <span>entropy rate <b className="text-accent">{info.entropy_rate_bits}</b> bits</span>
        )}
        {info.predictability != null && (
          <span>predictability <b className="text-accent">{Math.round(info.predictability * 100)}%</b></span>
        )}
        {info.lz76_complexity != null && (
          <span>LZ76 <b className="text-accent">{info.lz76_complexity}</b></span>
        )}
        {info.kinematic_sample_entropy != null && (
          <span>SampEn <b className="text-accent">{info.kinematic_sample_entropy}</b></span>
        )}
      </div>
    </div>
  );
}

/* ── remaining model outputs in one readout ───────────────────────────── */

/** The three Lomb-Scargle band summaries, in ascending frequency. Bands are
 *  bandwidth-normalised server-side, so these are comparable to each other. */
const SPECTRAL_BANDS = [
  { key: "drift_band", color: "#3f8ff0", opacity: 0.6 },
  { key: "rhythm_band", color: "#4fb6ff", opacity: 0.8 },
  { key: "burst_band", color: "#f2ad1f", opacity: 0.8 },
] as const;

function MetricsPanel({ profile }: { profile: CognitionProfile }) {
  const { spectral, foraging, habituation } = profile;
  return (
    <div className="rounded-xl border border-hairline bg-surface-2/60 p-3">
      <span className="hud-label">KINEMATIC & TEMPORAL DYNAMICS</span>
      <div className="mt-2 space-y-3 text-[11px]">
        {spectral && (
          <div>
            <div className="mb-1 font-mono text-[9px] uppercase tracking-wider text-muted">
              {spectral.method}
              {spectral.peak_frequency_hz != null && ` · peak ${spectral.peak_frequency_hz} Hz`}
            </div>
            {/* A band is null when its frequencies are unresolvable in this
                record's span — the grid is now bounded by 1/span and Nyquist.
                Rendering null as a 0%-wide bar would claim "measured, and
                empty" for something that was never measurable. */}
            <div className="flex h-4 w-full overflow-hidden rounded">
              {SPECTRAL_BANDS.map(({ key, color, opacity }) => {
                const v = spectral[key];
                return v == null ? null : (
                  <div
                    key={key}
                    title={`${key.replace("_band", "")} ${v}`}
                    style={{ width: `${v * 100}%`, background: color, opacity }}
                  />
                );
              })}
            </div>
            <div className="mt-0.5 flex gap-3 font-mono text-[9px] text-muted">
              {SPECTRAL_BANDS.map(({ key }) => {
                const v = spectral[key];
                const label = key.replace("_band", "");
                return (
                  <span key={key} className={v == null ? "opacity-60" : undefined}>
                    {label} {v == null ? "unresolvable" : `${Math.round(v * 100)}%`}
                  </span>
                );
              })}
            </div>
          </div>
        )}
        {foraging && (
          <div>
            <span className="font-mono text-[10px] text-ink-2">
              foraging regime{" "}
              <b className={foraging.regime === "levy_like" ? "text-accent-2" : "text-accent"}>
                {foraging.regime === "levy_like" ? "Lévy-like" : "Brownian-like"}
              </b>{" "}
              · α {foraging.alpha}
            </span>
            <p className="text-[10px] leading-tight text-muted">{foraging.interpretation}</p>
          </div>
        )}
        {habituation && (
          <div className="font-mono text-[10px] text-ink-2">
            habituation{" "}
            {habituation.habituating && habituation.half_life_s ? (
              <>
                <b className="text-accent">t½ = {habituation.half_life_s}s</b>
                <span className="text-muted"> · λ {habituation.decay_rate_per_s}/s · R² {habituation.r2}</span>
              </>
            ) : (
              <b className="text-good">sustained — no decay detected</b>
            )}
          </div>
        )}
        {profile.models_skipped.length > 0 && (
          <p className="text-[9px] text-muted">
            skipped (insufficient data): {profile.models_skipped.join(", ")}
          </p>
        )}
      </div>
    </div>
  );
}
