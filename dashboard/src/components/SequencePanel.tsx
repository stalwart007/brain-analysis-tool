"use client";

/**
 * MESSAGE SEQUENCE OPTIMISER — order effects.
 *
 * N! orderings is intractable, so the backend estimates a pairwise precedence
 * matrix from a sampled subset and solves the linear-ordering problem
 * heuristically. Two things have to be visible for the result to be readable:
 *
 *   1. The **net advantage matrix** A[i][j] = P[i][j] − P[j][i]. This is the
 *      antisymmetric quantity the ordering is actually solved on — what j gains
 *      from following i, minus what i gives up. Rendering it as a heatmap makes
 *      the recommendation auditable rather than an oracle.
 *   2. **Position effects separated from content effects.** A message that only
 *      wins because it went last is not a strong message, and conflating the
 *      two is the classic way order-effect studies mislead.
 */

import { useState } from "react";
import { motion } from "framer-motion";
import { CognitiveLoad } from "@/lib/api";
import { streamRun } from "@/lib/stream";
import { LoadToggle } from "./LoadToggle";
import SimStage from "./sim/SimStage";

interface SequenceResult {
  twin_count: number;
  failed_walks: number;
  messages: string[];
  submitted_ordering: number[];
  recommended_ordering: number[];
  recommended_from: string;
  recommended_messages: string[];
  objective: number;
  submitted_objective: number;
  objective_gain: number;
  objective_gain_ci: [number, number] | null;
  gain_supported: boolean;
  objective_units: string;
  net_advantage?: number[][];
  precedence?: number[][];
  position_effects?: number[];
  content_effects?: number[];
  primacy?: number | null;
  recency?: number | null;
  exploration?: {
    orderings_designed: number;
    orderings_walked: number;
    sample_space: number;
    [k: string]: unknown;
  };
  search?: { heuristic: string; optimal: boolean; note?: string };
  method?: string;
  disclaimer?: string;
}

const SAMPLE = [
  "Your inbox is out of control. Atlas reads it overnight so you don't have to.",
  "Atlas drafted 3 replies while you slept. Approve them in one tap.",
  "Nothing leaves your device. Atlas runs locally — your mail is never uploaded.",
  "Teams using Atlas get back 9 hours a week. Start free for 14 days.",
];

/** Net advantage heatmap: row i, column j = how much j gains by following i. */
function AdvantageMatrix({ m, labels }: { m: number[][]; labels: string[] }) {
  const peak = Math.max(1e-9, ...m.flat().map(Math.abs));
  const n = m.length;
  const cell = 34;
  const pad = 22;
  return (
    <svg viewBox={`0 0 ${pad + n * cell + 8} ${pad + n * cell + 8}`} className="w-full max-w-sm">
      {m.map((row, i) =>
        row.map((v, j) => {
          if (i === j) {
            return (
              <rect key={`${i}-${j}`} x={pad + j * cell} y={pad + i * cell}
                width={cell - 2} height={cell - 2} fill="rgba(223,217,217,0.04)" />
            );
          }
          // green = j genuinely benefits from following i; red = it costs
          const t = v / peak;
          const color = t >= 0 ? "22,208,22" : "240,96,95";
          return (
            <rect key={`${i}-${j}`} x={pad + j * cell} y={pad + i * cell}
              width={cell - 2} height={cell - 2}
              fill={`rgb(${color} / ${Math.min(0.88, Math.abs(t) * 0.85 + 0.06)})`}>
              <title>{`${labels[j]} after ${labels[i]}: ${v >= 0 ? "+" : ""}${v.toFixed(3)}`}</title>
            </rect>
          );
        })
      )}
      {labels.map((l, i) => (
        <g key={i}>
          <text x={pad - 5} y={pad + i * cell + cell / 2} fontSize="9" fill="#a8a3a3"
            textAnchor="end" dominantBaseline="middle" fontFamily="ui-monospace, monospace">{l}</text>
          <text x={pad + i * cell + (cell - 2) / 2} y={pad - 7} fontSize="9" fill="#a8a3a3"
            textAnchor="middle" fontFamily="ui-monospace, monospace">{l}</text>
        </g>
      ))}
      <text x={pad} y={pad + n * cell + 6} fontSize="7" fill="#6f6c6c" fontFamily="ui-monospace, monospace">
        row = precedes · column = follows
      </text>
    </svg>
  );
}

export default function SequencePanel({ personaCount }: { personaCount: number }) {
  const [text, setText] = useState("");
  const [orderings, setOrderings] = useState(6);
  const [twins, setTwins] = useState(2);
  const [load, setLoad] = useState<CognitiveLoad>("low");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<SequenceResult | null>(null);
  const [walked, setWalked] = useState(0);
  const [designed, setDesigned] = useState(0);
  const [totalWalks, setTotalWalks] = useState<number | null>(null);

  const messages = text.split("\n").map((s) => s.trim()).filter(Boolean);

  async function run() {
    setBusy(true);
    setError(null);
    setResult(null);
    setWalked(0);
    setDesigned(0);
    try {
      await streamRun(
        "/studies/sequence/stream",
        {
          messages,
          orderings_sampled: orderings,
          twins_per_persona: twins,
          cognitive_load: load,
        },
        (evt) => {
          // `ordering` is the DESIGN announcement — the server emits every
          // ordering up front, before a single twin runs. Counting those made
          // the progress label jump straight to its final value at t=0 and then
          // sit still for the whole run, which read as a hang.
          //
          // `agent` is the actual per-walk completion, so that is what
          // progress means here. `agent_failed` counts too: a twin that failed
          // is a walk that finished, and excluding it would leave the counter
          // permanently short of the total.
          if (evt.type === "start") setTotalWalks(Number(evt.total ?? 0) || null);
          else if (evt.type === "ordering") setDesigned((n) => n + 1);
          else if (evt.type === "agent" || evt.type === "agent_failed")
            setWalked((n) => n + 1);
          else if (evt.type === "done") setResult(evt.result as unknown as SequenceResult);
          else if (evt.type === "error") setError(String(evt.detail));
        }
      );
    } catch (e) {
      setError(e instanceof Error ? e.message : "Sequence run failed");
    } finally {
      setBusy(false);
    }
  }

  const labels = (result?.messages ?? messages).map((_, i) => `M${i + 1}`);

  return (
    <div className="card p-6" data-live={busy ? "1" : "0"}>
      <div className="mb-3 flex items-baseline justify-between">
        <h2>Sequence optimizer</h2>
        <span className="text-xs text-muted">order effects · which order lands</span>
      </div>
      <p className="lede mb-4">
        One message per line, in the order you plan to send them. Twins walk sampled
        orderings; the backend estimates a pairwise precedence matrix and solves the
        linear-ordering problem — and reports position effects separately, so a
        message that only wins by going last isn&apos;t mistaken for a strong one.
      </p>

      <SimStage
        show={busy || !!result}
        busy={busy}
        // `walked` counts WALKS (personas × twins) and `designed` counts
        // ORDERINGS, so pairing them counted to 12 against a denominator of 6
        // at the defaults. sequence.py's `start` frame already carries both
        // numbers; use `total` for the walk denominator and report the two
        // quantities separately rather than pretending they are one ratio.
        runningLabel={
          totalWalks
            ? `${walked} of ${totalWalks} walks · ${designed} orderings`
            : designed
              ? `${walked} walks · ${designed} orderings`
              : "designing orderings"
        }
        doneLabel="ordering solved"
        height={230}
      >
        {result?.net_advantage ? (
          <div className="flex h-full items-center justify-center p-2">
            <AdvantageMatrix m={result.net_advantage} labels={labels} />
          </div>
        ) : (
          <div className="flex h-full flex-col items-center justify-center gap-2">
            <span className="spike-train">
              {Array.from({ length: 12 }, (_, i) => <i key={i} />)}
            </span>
            <span className="hud-label">estimating precedence</span>
          </div>
        )}
      </SimStage>

      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        rows={5}
        placeholder={"One message per line, in your planned order…\n\nMessage 1\nMessage 2\nMessage 3"}
        className="w-full resize-y rounded-xl border border-hairline bg-surface-2 p-3.5 font-mono text-xs outline-none placeholder:text-muted focus:border-accent/60"
      />
      <button
        onClick={() => setText(SAMPLE.join("\n"))}
        className="mt-2 border border-dashed border-hairline px-2.5 py-1 font-mono text-[10px] uppercase tracking-wider text-muted hover:text-ink-2"
      >
        use sample sequence
      </button>

      <div className="mt-3 flex flex-wrap items-center gap-3 text-xs text-muted">
        <span className="soma">{messages.length} messages</span>
        <label className="flex items-center gap-2">
          Orderings
          <input type="number" min={2} max={24} value={orderings}
            onChange={(e) => setOrderings(Math.min(24, Math.max(2, +e.target.value || 2)))}
            className="w-14 border border-hairline bg-surface-2 px-2 py-1.5 text-sm tabular-nums text-ink outline-none" />
        </label>
        <label className="flex items-center gap-2">
          Twins
          <input type="number" min={1} max={10} value={twins}
            onChange={(e) => setTwins(Math.min(10, Math.max(1, +e.target.value || 1)))}
            className="w-14 border border-hairline bg-surface-2 px-2 py-1.5 text-sm tabular-nums text-ink outline-none" />
        </label>
        <LoadToggle value={load} onChange={setLoad} id="sequence-load" />
        <motion.button
          whileTap={{ scale: 0.97 }}
          disabled={busy || messages.length < 2 || personaCount === 0}
          onClick={run}
          className="ml-auto bg-accent px-5 py-2 disabled:opacity-40"
        >
          {busy ? "Walking…" : "Solve order"}
        </motion.button>
      </div>

      {error && <p className="mt-3 text-sm text-critical">{error}</p>}

      {result && (
        <div className="mt-5 space-y-4">
          <div
            className={`border p-3 ${
              result.gain_supported
                ? "border-good/40 bg-good/[0.07]"
                : "border-accent-2/40 bg-accent-2/[0.07]"
            }`}
          >
            <div className="hud-label">
              {result.gain_supported ? "Reordering supported" : "Reordering not supported"}
            </div>
            <p className="verdict mt-1 text-ink-2">
              {result.gain_supported
                ? `Recommended order gains ${result.objective_gain.toFixed(3)} ${result.objective_units} over your submitted order.`
                : `The ${result.objective_gain.toFixed(3)} gain is inside the noise — the interval straddles zero, so keep your current order.`}
            </p>
            {result.objective_gain_ci && (
              <p className="mt-1 font-mono text-[10px] text-muted">
                95% CI {result.objective_gain_ci[0].toFixed(3)} – {result.objective_gain_ci[1].toFixed(3)}
                {result.search && ` · ${result.search.heuristic}${result.search.optimal ? "" : " (heuristic, not proven optimal)"}`}
              </p>
            )}
          </div>

          <div>
            <div className="hud-label mb-2">Recommended order</div>
            <ol className="space-y-1.5">
              {result.recommended_ordering.map((idx, pos) => (
                <li key={pos} className="flex items-baseline gap-3 text-sm">
                  <span className="soma shrink-0">
                    {pos + 1} ← M{idx + 1}
                  </span>
                  <span className="truncate text-ink-2">{result.messages[idx]}</span>
                </li>
              ))}
            </ol>
          </div>

          {result.position_effects && result.content_effects && (
            <div className="grid gap-4 sm:grid-cols-2">
              <div>
                <div className="hud-label mb-1">Content effect</div>
                <p className="mb-2 text-[10px] leading-tight text-muted">
                  Message strength, independent of where it sat.
                </p>
                {result.content_effects.map((v, i) => (
                  <div key={i} className="mb-1 flex items-center gap-2 font-mono text-[10px]">
                    <span className="w-7 text-muted">M{i + 1}</span>
                    <span className="relative h-1 flex-1 bg-white/[0.06]">
                      <span className="absolute inset-y-0 left-1/2 w-px bg-white/20" />
                      <span
                        className="absolute inset-y-0"
                        style={{
                          left: v >= 0 ? "50%" : `${50 + v * 100}%`,
                          width: `${Math.min(50, Math.abs(v) * 100)}%`,
                          background: v >= 0 ? "#16d016" : "#f0605f",
                        }}
                      />
                    </span>
                    <span className="w-10 tabular-nums text-ink-2">{v >= 0 ? "+" : ""}{v.toFixed(2)}</span>
                  </div>
                ))}
              </div>
              <div>
                <div className="hud-label mb-1">Position effect</div>
                <p className="mb-2 text-[10px] leading-tight text-muted">
                  What the slot is worth regardless of what fills it.
                  {result.primacy != null && ` Primacy ${result.primacy.toFixed(2)}.`}
                  {result.recency != null && ` Recency ${result.recency.toFixed(2)}.`}
                </p>
                {result.position_effects.map((v, i) => (
                  <div key={i} className="mb-1 flex items-center gap-2 font-mono text-[10px]">
                    <span className="w-7 text-muted">#{i + 1}</span>
                    <span className="relative h-1 flex-1 bg-white/[0.06]">
                      <span className="absolute inset-y-0 left-1/2 w-px bg-white/20" />
                      <span
                        className="absolute inset-y-0"
                        style={{
                          left: v >= 0 ? "50%" : `${50 + v * 100}%`,
                          width: `${Math.min(50, Math.abs(v) * 100)}%`,
                          background: "rgb(var(--region-rgb))",
                        }}
                      />
                    </span>
                    <span className="w-10 tabular-nums text-ink-2">{v >= 0 ? "+" : ""}{v.toFixed(2)}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {result.exploration && (
            <p className="font-mono text-[10px] text-muted">
              explored {result.exploration.orderings_walked} of{" "}
              {result.exploration.sample_space.toLocaleString()} possible orderings
              {result.failed_walks > 0 && ` · ${result.failed_walks} walks failed`}
            </p>
          )}
          {result.disclaimer && (
            <p className="text-[9px] leading-tight text-muted">{result.disclaimer}</p>
          )}
        </div>
      )}
    </div>
  );
}
