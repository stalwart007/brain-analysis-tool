"use client";

/**
 * NEURO-IMPACT STUDY — feed content (script / ad / article), the swarm
 * experiences it beat by beat, and the analysis stack answers "what does this
 * do to an audience": attention synchrony (ISC), structural drop-off points
 * (change-point detection), what memory will keep (peak-end + serial
 * position), the emotional trajectory (circumplex), and a functional brain
 * atlas per beat. Every number on screen is a model output.
 */

import { useMemo, useState } from "react";
import { motion } from "framer-motion";
import { CognitiveLoad } from "@/lib/api";
import { streamRun } from "@/lib/stream";
import { LoadToggle } from "./LoadToggle";
import BrainMap, { SystemScores } from "./BrainMap";
import { AssetInput, assetReady, KINDS, type AssetKind, type ContentAsset } from "./AssetInput";

interface RetentionStep {
  beat: number;
  entered: number;
  dropped: number;
  /** null once the risk set is empty — UNDEFINED, not zero. */
  hazard: number | null;
  survival: number;
}

interface ContentResult {
  segments: string[];
  /** temporal | sequential | spatial — decides which statistics below exist. */
  axis?: string | null;
  beat_source?: string | null;
  beat_notes?: string | null;
  modality?: string;
  /** statistic -> why it was not computed for this modality. */
  withheld?: Record<string, string>;
  retention?: {
    threshold: number;
    steps: RetentionStep[];
    completion_rate: number;
    worst_beat: number | null;
  } | null;
  twin_count: number;
  curves: Record<string, number[]>;
  curve_cis: Record<string, ([number, number] | null)[]>;
  isc: { overall: number; grip: string; leave_one_out: (number | null)[]; method: string } | null;
  change_points: number[] | null;
  memory: {
    remembered_affect: number;
    peak_index: number;
    peak_index_distribution: number[];
    peak_agreement: number;
    encoding_strength: number;
    n_twins: number;
    model: string;
  } | null;
  trajectory: { dynamism: number; net_valence_shift: number; quadrant_occupancy: Record<string, number>; model: string } | null;
  systems: SystemScores;
  per_segment_systems: SystemScores[];
  share_intent_mean: number;
  gists: string[];
  verdict: string;
  disclaimer: string;
}

const SAMPLE = `Scene 1 (0:00): Black screen. A phone buzzes at 3:12 AM. A founder stares at a wall of unread Slack messages.
Scene 2 (0:08): Quick cuts — 12 browser tabs, 4 dashboards, a calendar with no gaps. Voiceover: "Your tools were supposed to help."
Scene 3 (0:18): Everything freezes. One clean interface slides in: Atlas. The clutter dissolves into a single feed.
Scene 4 (0:26): The founder types one sentence; Atlas drafts the plan, books the meetings, files the tickets.
Scene 5 (0:34): Morning. The founder is asleep. Atlas's status: "11 hours of busywork handled."
Scene 6 (0:40): Logo. "Atlas — get your week back." Free 14-day trial.`;

export default function ContentImpactPanel({ personaCount }: { personaCount: number }) {
  const [kind, setKind] = useState<AssetKind>("text");
  const [asset, setAsset] = useState<ContentAsset>({ kind: "text" });
  const [twins, setTwins] = useState(3);
  const [load, setLoad] = useState<CognitiveLoad>("low");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [stage, setStage] = useState<string | null>(null);
  const [segments, setSegments] = useState<string[]>([]);
  const [traces, setTraces] = useState<number[][]>([]); // live attention curves
  const [result, setResult] = useState<ContentResult | null>(null);
  const [beat, setBeat] = useState<number | null>(null); // brain-map scrubber

  async function run() {
    setBusy(true); setError(null); setResult(null);
    setTraces([]); setSegments([]); setBeat(null); setStage("segmenting content…");
    try {
      await streamRun(
        "/studies/content/stream",
        {
          asset: { ...asset, kind },
          twins_per_persona: twins,
          cognitive_load: load,
        },
        (evt) => {
          if (evt.type === "stage") setStage(String(evt.detail));
          else if (evt.type === "start") {
            setSegments(evt.segments as string[]);
            setStage(`${(evt.segments as string[]).length} beats — audience experiencing content…`);
          } else if (evt.type === "agent") setTraces((t) => [...t, evt.attention as number[]]);
          else if (evt.type === "done") setResult(evt.result as unknown as ContentResult);
          else if (evt.type === "error") setError(String(evt.detail));
        }
      );
    } catch (e) {
      setError(e instanceof Error ? e.message : "Study failed");
    } finally {
      setBusy(false); setStage(null);
    }
  }

  return (
    <div className="card p-6">
      <div className="mb-1 flex items-baseline justify-between">
        <h2 className="font-display text-lg font-medium tracking-tight">Neuro-Impact Study</h2>
        <span className="text-xs text-muted">what the content does to an audience, beat by beat</span>
      </div>
      <p className="lede mb-4">
        Any digital content — a script, a static ad, a video, a podcast, a
        landing page, a deck. The swarm experiences it beat by beat; you get
        attention synchrony, drop-off, what memory keeps, the emotional arc,
        and a functional brain atlas. Statistics that the format cannot support
        are withheld with their reason rather than computed anyway.
      </p>

      {/* Modality picker. Each kind reaches the study through a different
          adapter and lands on a different AXIS, which is what decides whether
          peak-end, change points and retention are meaningful at all. */}
      <div
        role="radiogroup"
        aria-label="Content type"
        className="mb-3 flex flex-wrap gap-1.5"
      >
        {KINDS.map((k) => (
          <button
            key={k.id}
            role="radio"
            aria-checked={kind === k.id}
            title={k.hint}
            disabled={busy}
            onClick={() => {
              setKind(k.id);
              setAsset({ kind: k.id });
              setResult(null);
            }}
            className={`rounded-lg border px-3 py-1.5 text-xs transition ${
              kind === k.id
                ? "border-accent/60 bg-accent/[0.10] text-ink"
                : "border-hairline text-muted hover:border-accent/40 hover:text-ink-2"
            }`}
          >
            {k.label}
          </button>
        ))}
      </div>

      <AssetInput kind={kind} asset={asset} onChange={setAsset} disabled={busy} />

      <div className="mt-3 flex flex-wrap items-center gap-3">
        {kind === "text" && (
          <button
            onClick={() => setAsset({ kind: "text", text: SAMPLE })}
            className="rounded-lg border border-dashed border-hairline px-3 py-1.5 text-xs text-muted transition hover:border-accent/50 hover:text-ink-2"
          >
            use sample script
          </button>
        )}
        <label className="flex items-center gap-2 text-xs text-muted">
          Twins / persona
          <input
            type="number" min={1} max={20} value={twins}
            onChange={(e) => setTwins(parseInt(e.target.value, 10) || 1)}
            className="w-16 rounded-lg border border-hairline bg-surface-2 px-2 py-1.5 text-sm text-ink outline-none tabular-nums"
          />
        </label>
        <LoadToggle value={load} onChange={setLoad} id="content-load" />
        <motion.button
          whileTap={{ scale: 0.97 }}
          disabled={busy || !!assetReady(kind, asset)}
          onClick={run}
          title={assetReady(kind, asset) ?? undefined}
          className="ml-auto rounded-xl bg-accent px-5 py-2 font-display text-sm font-semibold text-white transition hover:brightness-110 disabled:opacity-40"
        >
          {busy ? "Screening…" : "Run neuro-impact study"}
        </motion.button>
      </div>

      {stage && (
        <p className="mt-3 font-mono text-[11px] text-accent">
          <span className="pulse-dot mr-1.5 inline-block h-1.5 w-1.5 rounded-full bg-accent align-middle" />
          {stage} {traces.length > 0 && `· ${traces.length} twins reported`}
        </p>
      )}
      {error && <p className="mt-3 text-sm text-critical">{error}</p>}

      {/* live ghost traces: every twin's attention curve as it arrives */}
      {(busy || result) && traces.length > 0 && segments.length > 1 && (
        <AttentionField traces={traces} result={result} segments={segments} />
      )}

      {result && (
        <div className="mt-5 space-y-5">
          <p className="verdict rounded-xl border border-accent/25 bg-accent/5 px-3 py-2 text-ink-2">
            <span className="hud-label mr-2">VERDICT</span>{result.verdict}
          </p>

          {/* How the beats were derived, and what this format cannot support.
              An absent statistic and an inapplicable one look identical in a
              UI, and only one of them is a finding — so the reasons are shown
              rather than the rows silently missing. */}
          {(result.axis || result.beat_notes) && (
            <p className="mt-2 font-mono text-[10px] leading-relaxed text-muted">
              {result.modality && <>{result.modality} · </>}
              {result.axis && <>{result.axis} axis · </>}
              {result.beat_source === "paragraph_fallback" ? (
                <span className="text-critical">
                  beats are a mechanical split, not model-identified structure
                </span>
              ) : (
                <>beats via {result.beat_source}</>
              )}
              {result.beat_notes && <> · {result.beat_notes}</>}
            </p>
          )}
          {result.withheld && Object.keys(result.withheld).length > 0 && (
            <div className="mt-2 rounded-xl border border-hairline bg-surface-2/40 px-3 py-2">
              <span className="hud-label text-muted">NOT COMPUTED FOR THIS FORMAT</span>
              <ul className="mt-1 space-y-1">
                {Object.entries(result.withheld).map(([stat, reason]) => (
                  <li key={stat} className="text-[10px] leading-relaxed text-muted">
                    <b className="text-ink-2">{stat.replace(/_/g, " ")}</b> — {reason}
                  </li>
                ))}
              </ul>
            </div>
          )}

          <div className="grid gap-5 lg:grid-cols-2">
            {/* brain atlas with beat scrubber */}
            <div className="rounded-xl border border-hairline bg-surface-2/60 p-3">
              <div className="mb-2 flex flex-wrap items-center gap-1.5">
                <span className="hud-label mr-1">FUNCTIONAL BRAIN ATLAS</span>
                <button
                  onClick={() => setBeat(null)}
                  className={`rounded-full border px-2 py-0.5 font-mono text-[9px] ${beat === null ? "border-accent text-accent" : "border-hairline text-muted"}`}
                >
                  OVERALL
                </button>
                {result.segments.map((_, i) => (
                  <button
                    key={i}
                    onClick={() => setBeat(i)}
                    className={`rounded-full border px-2 py-0.5 font-mono text-[9px] ${beat === i ? "border-accent text-accent" : "border-hairline text-muted"}`}
                  >
                    B{i + 1}
                  </button>
                ))}
              </div>
              <BrainMap
                scores={beat === null ? result.systems : result.per_segment_systems[beat]}
                caption={
                  beat === null
                    ? undefined
                    : `Beat ${beat + 1}: ${result.segments[beat].slice(0, 70)}`
                }
              />
            </div>

            <div className="space-y-4">
              {/* ISC */}
              {result.isc && (
                <div className="rounded-xl border border-hairline bg-surface-2/60 p-3">
                  <div className="flex items-baseline justify-between">
                    <span className="hud-label">ATTENTION SYNCHRONY (ISC)</span>
                    {/* Colour follows the backend's `grip`, which is now gated on a
                        permutation test. Re-deriving it here from raw r cuts was a
                        third threshold ladder that could contradict the label
                        printed beside it. */}
                    <span className={`font-display text-xl font-semibold ${result.isc.grip === "locked-in" ? "text-good" : result.isc.grip === "engaged" ? "text-accent" : "text-critical"}`}>
                      {result.isc.overall.toFixed(2)} · {result.isc.grip}
                    </span>
                  </div>
                  <div className="mt-2 flex h-6 items-end gap-0.5">
                    {result.isc.leave_one_out.map((r, i) => (
                      <div key={i} title={`twin ${i + 1}: r=${r ?? "—"}`}
                        className="flex-1 rounded-t-sm bg-accent"
                        style={{ height: `${Math.max(6, ((r ?? 0) + 1) * 50)}%`, opacity: 0.35 + ((r ?? 0) + 1) * 0.3 }}
                      />
                    ))}
                  </div>
                  <p className="mt-1 text-[9px] text-muted">{result.isc.method} · per-twin leave-one-out synchrony</p>
                </div>
              )}

              {/* memory — the peak-index DISTRIBUTION, not a softmax.
                  Peak-end is a within-subject phenomenon, so this is where
                  each twin individually peaked. One tall bar = the audience
                  shares a peak; several = the audience is split, which the old
                  mean-curve version silently averaged away. */}
              {result.memory && (
                <div className="rounded-xl border border-hairline bg-surface-2/60 p-3">
                  <span className="hud-label">WHERE EACH TWIN PEAKED</span>
                  <div className="mt-2 flex items-end gap-1">
                    {result.memory.peak_index_distribution.map((count, i) => {
                      const top = Math.max(...result.memory!.peak_index_distribution, 1);
                      return (
                        <div key={i} className="flex-1 text-center">
                          <div
                            className="mx-auto w-full rounded-t-sm"
                            style={{
                              height: `${4 + (count / top) * 94}px`,
                              background: i === result.memory!.peak_index ? "#f2ad1f" : "#3f8ff0",
                              opacity: count ? 0.85 : 0.18,
                            }}
                            title={`${count} of ${result.memory!.n_twins} twins peaked on beat ${i + 1}`}
                          />
                          <span className="font-mono text-[8px] text-muted">B{i + 1}</span>
                        </div>
                      );
                    })}
                  </div>
                  <p className="mt-1 text-[9px] text-muted">
                    peak agreement {(result.memory.peak_agreement * 100).toFixed(0)}%
                    {result.memory.peak_agreement < 0.5 && (
                      <span className="text-critical">
                        {" "}— the audience does not share a peak
                      </span>
                    )}
                    {" "}· remembered affect {result.memory.remembered_affect > 0 ? "+" : ""}
                    {result.memory.remembered_affect.toFixed(2)} · encoding{" "}
                    {result.memory.encoding_strength.toFixed(2)}
                  </p>
                </div>
              )}

              {/* retention — the commercially load-bearing number for anything
                  with a running order: where do people stop. */}
              {result.retention && result.retention.steps.length > 0 && (
                <div className="rounded-xl border border-hairline bg-surface-2/60 p-3">
                  <span className="hud-label">RETENTION</span>
                  <div className="mt-2 flex items-end gap-1">
                    {result.retention.steps.map((st) => (
                      <div key={st.beat} className="flex-1 text-center">
                        <div
                          className="mx-auto w-full rounded-t-sm"
                          style={{
                            height: `${4 + st.survival * 94}px`,
                            // An undefined hazard (empty risk set) is drawn
                            // hollow, never as a zero-hazard bar — "nobody left
                            // here" and "nobody was left to leave" are
                            // different findings.
                            background: st.hazard === null ? "transparent" : "#4fb477",
                            border: st.hazard === null ? "1px dashed rgba(255,255,255,0.25)" : "none",
                            opacity: 0.85,
                          }}
                          title={
                            st.hazard === null
                              ? `beat ${st.beat + 1}: nobody left at risk — hazard undefined`
                              : `beat ${st.beat + 1}: ${st.dropped} of ${st.entered} dropped (hazard ${st.hazard})`
                          }
                        />
                        <span className="font-mono text-[8px] text-muted">B{st.beat + 1}</span>
                      </div>
                    ))}
                  </div>
                  <p className="mt-1 text-[9px] text-muted">
                    {(result.retention.completion_rate * 100).toFixed(0)}% reached the end
                    {result.retention.worst_beat !== null && (
                      <> · steepest drop at beat {result.retention.worst_beat + 1}</>
                    )}
                  </p>
                </div>
              )}

              {/* trajectory */}
              {result.trajectory && <Circumplex result={result} />}
            </div>
          </div>

          <div className="flex flex-wrap gap-6 font-mono text-[11px] text-ink-2">
            <span>share intent <b className="text-accent">{Math.round(result.share_intent_mean * 100)}%</b></span>
            <span>emotional dynamism <b className="text-accent">{result.trajectory?.dynamism.toFixed(2)}</b></span>
            <span>{result.twin_count} twins · load {load}</span>
          </div>

          {result.gists.length > 0 && (
            <details className="text-xs text-muted">
              <summary className="cursor-pointer font-mono text-[10px] uppercase tracking-wider hover:text-ink-2">what twins took away</summary>
              <ul className="mt-1.5 space-y-1 border-l border-hairline pl-3">
                {result.gists.map((g, i) => <li key={i} className="text-[11px] text-ink-2">{g}</li>)}
              </ul>
            </details>
          )}
          <p className="text-[9px] leading-tight text-muted">{result.disclaimer}</p>
        </div>
      )}
    </div>
  );
}

/* ── attention field: ghost curve per twin + mean + CI + change points ── */

function AttentionField({
  traces, result, segments,
}: {
  traces: number[][];
  result: ContentResult | null;
  segments: string[];
}) {
  const W = 720, H = 150, padX = 34, padY = 16;
  const n = segments.length;
  const x = (i: number) => padX + (i / Math.max(1, n - 1)) * (W - padX * 2);
  const y = (v: number) => H - padY - v * (H - padY * 2);
  const mean = useMemo(() => {
    if (result) return result.curves.attention;
    if (!traces.length) return [];
    return Array.from({ length: n }, (_, i) =>
      traces.reduce((s, t) => s + (t[i] ?? 0), 0) / traces.length
    );
  }, [traces, result, n]);
  const cis = result?.curve_cis.attention;
  const band = cis && cis.every(Boolean)
    ? `M ${cis.map((c, i) => `${x(i)},${y((c as [number, number])[1])}`).join(" L ")} L ${[...cis].reverse().map((c, ri) => `${x(cis.length - 1 - ri)},${y((c as [number, number])[0])}`).join(" L ")} Z`
    : null;
  return (
    <div className="mt-4 rounded-xl border border-hairline bg-black/40 p-3">
      <div className="flex items-baseline justify-between">
        <span className="hud-label">AUDIENCE ATTENTION FIELD</span>
        <span className="font-mono text-[9px] text-muted">
          one trace per twin · bold = population mean · dashed = detected change points
        </span>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} className="mt-1 w-full">
        {[0, 0.5, 1].map((v) => (
          <g key={v}>
            <line x1={padX} y1={y(v)} x2={W - padX} y2={y(v)} stroke="rgba(120,165,230,0.10)" />
            <text x={padX - 6} y={y(v) + 2.5} fontSize="7" fill="#5f6b7d" textAnchor="end" fontFamily="ui-monospace, monospace">{v}</text>
          </g>
        ))}
        {band && <path d={band} fill="rgba(79,182,255,0.12)" />}
        {traces.map((t, k) => (
          <polyline key={k} fill="none" stroke="rgba(120,195,255,0.16)" strokeWidth="1"
            points={t.map((v, i) => `${x(i)},${y(Math.max(0, Math.min(1, v)))}`).join(" ")} />
        ))}
        {mean.length > 1 && (
          <polyline fill="none" stroke="#4fb6ff" strokeWidth="2"
            points={mean.map((v, i) => `${x(i)},${y(v)}`).join(" ")} />
        )}
        {result?.change_points?.map((cp) => (
          <g key={cp}>
            <line x1={x(cp)} y1={padY} x2={x(cp)} y2={H - padY} stroke="#f0605f" strokeWidth="1" strokeDasharray="3 3" />
            <text x={x(cp) + 3} y={padY + 7} fontSize="7" fill="#f0605f" fontFamily="ui-monospace, monospace">regime shift</text>
          </g>
        ))}
        {segments.map((_, i) => (
          <text key={i} x={x(i)} y={H - 3} fontSize="7" fill="#5f6b7d" textAnchor="middle" fontFamily="ui-monospace, monospace">B{i + 1}</text>
        ))}
      </svg>
    </div>
  );
}

/* ── circumplex: the emotional arc through valence × arousal space ────── */

function Circumplex({ result }: { result: ContentResult }) {
  const S = 170, pad = 20;
  const vx = (v: number) => pad + ((v + 1) / 2) * (S - pad * 2);
  const ay = (a: number) => S - pad - a * (S - pad * 2);
  const V = result.curves.valence, A = result.curves.arousal;
  return (
    <div className="rounded-xl border border-hairline bg-surface-2/60 p-3">
      <span className="hud-label">EMOTIONAL TRAJECTORY (CIRCUMPLEX)</span>
      <div className="mt-2 flex gap-3">
        <svg viewBox={`0 0 ${S} ${S}`} className="h-40 w-40 shrink-0">
          <rect x={pad} y={pad} width={S - pad * 2} height={S - pad * 2} fill="none" stroke="rgba(120,165,230,0.15)" />
          <line x1={S / 2} y1={pad} x2={S / 2} y2={S - pad} stroke="rgba(120,165,230,0.12)" />
          <line x1={pad} y1={S / 2} x2={S - pad} y2={S / 2} stroke="rgba(120,165,230,0.12)" />
          <text x={S - pad} y={12} fontSize="6.5" fill="#5f6b7d" textAnchor="end" fontFamily="ui-monospace, monospace">arousal ↑</text>
          <text x={S - pad} y={S - 4} fontSize="6.5" fill="#5f6b7d" textAnchor="end" fontFamily="ui-monospace, monospace">valence →</text>
          <polyline fill="none" stroke="rgba(240,120,200,0.7)" strokeWidth="1.5"
            points={V.map((v, i) => `${vx(v)},${ay(A[i])}`).join(" ")} />
          {V.map((v, i) => (
            <g key={i}>
              <circle cx={vx(v)} cy={ay(A[i])} r={i === 0 ? 2 : 2.6} fill={i === V.length - 1 ? "#b07ff0" : "#f078c8"} opacity={0.5 + (i / V.length) * 0.5} />
              <text x={vx(v) + 4} y={ay(A[i]) - 3} fontSize="6" fill="#8fa3c0" fontFamily="ui-monospace, monospace">B{i + 1}</text>
            </g>
          ))}
        </svg>
        <div className="space-y-1 font-mono text-[10px] text-ink-2">
          <div>net valence shift <b className="text-accent">{result.trajectory!.net_valence_shift > 0 ? "+" : ""}{result.trajectory!.net_valence_shift.toFixed(2)}</b></div>
          <div>dynamism <b className="text-accent">{result.trajectory!.dynamism.toFixed(2)}</b></div>
          <div className="pt-1 text-[9px] text-muted">audience time share:</div>
          {Object.entries(result.trajectory!.quadrant_occupancy).map(([q, f]) => (
            <div key={q} className="flex items-center gap-1.5">
              <span className="w-16 text-muted">{q}</span>
              <div className="h-1.5 w-20 rounded bg-white/[0.07]">
                <div className="h-1.5 rounded bg-accent" style={{ width: `${f * 100}%`, opacity: 0.8 }} />
              </div>
              <span className="text-muted">{Math.round(f * 100)}%</span>
            </div>
          ))}
          <p className="pt-1 text-[9px] text-muted">{result.trajectory!.model}</p>
        </div>
      </div>
    </div>
  );
}
