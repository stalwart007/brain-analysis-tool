"use client";

import { useCallback, useEffect, useState } from "react";
import { motion } from "framer-motion";
import { api, CalibrationReport, MetricCalibration, SwarmRunRow } from "@/lib/api";

export default function CalibrationPanel() {
  const [runs, setRuns] = useState<SwarmRunRow[]>([]);
  const [report, setReport] = useState<CalibrationReport | null>(null);
  const [drafts, setDrafts] = useState<Record<string, string>>({});
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(() => {
    api.swarmRuns("swarm").then(setRuns).catch(() => {});
    api.validationReport().then(setReport).catch(() => {});
  }, []);

  useEffect(refresh, [refresh]);

  async function save(runId: string) {
    const raw = drafts[runId];
    const value = parseFloat(raw);
    if (isNaN(value) || value < 0 || value > 1) {
      setError("Actual intent must be a number between 0 and 1.");
      return;
    }
    setError(null);
    try {
      await api.recordActuals(runId, { intent: value });
      refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to record");
    }
  }

  return (
    <div className="card p-6">
      <div className="mb-1 flex items-baseline justify-between">
        <h2 className="font-display text-lg font-medium tracking-tight">Calibration</h2>
        <span className="text-xs text-muted">the honesty loop</span>
      </div>
      <p className="mb-4 text-sm text-muted">
        When a stimulus ships for real, record the observed intent (CTR-derived,
        completion rate, …) against its run. Error metrics tell you how far the
        swarm is from reality — and whether to trust it.
      </p>

      {/* validation.py computes eleven fields and this panel used to render
          three of them — runs_with_actuals, intent.mae, intent.bias — as bare
          numbers, dropping every interval, the slope/r pair, the verdict
          string, and the entire engagement column. That is the exact opposite
          of what the module's own docstring argues: MAE and bias cannot
          separate a compressed predictor from a noisy one, and slope and r are
          what tell them apart. The verdict leads, because it is the sentence a
          buyer would actually quote. */}
      {report?.intent?.verdict && (
        <p className="mb-4 border-l-2 border-accent/50 pl-3 text-[13px] leading-relaxed text-ink">
          {report.intent.verdict}
        </p>
      )}

      <div className="mb-5 flex flex-wrap gap-x-8 gap-y-4">
        <Stat label="Runs with actuals" value={report ? String(report.runs_with_actuals) : "—"} />
        <MetricColumn label="Intent" m={report?.intent ?? null} />
        <MetricColumn label="Engagement" m={report?.engagement ?? null} />
      </div>

      {error && <p className="mb-3 text-sm text-critical">{error}</p>}

      {runs.length === 0 ? (
        <p className="text-sm text-muted">No swarm runs yet.</p>
      ) : (
        <div className="max-h-64 overflow-auto">
          <table className="w-full text-sm">
            <thead className="sticky top-0 bg-surface text-left text-xs uppercase tracking-wider text-muted">
              <tr>
                <th className="pb-2 pr-2 font-medium">Run</th>
                <th className="pb-2 pr-2 font-medium">Predicted intent</th>
                <th className="pb-2 pr-2 font-medium">Actual</th>
                <th className="pb-2" />
              </tr>
            </thead>
            <tbody>
              {runs.map((r) => (
                <tr key={r.id} className="border-t border-hairline/60">
                  <td className="py-2 pr-2">
                    <code className="text-xs text-ink-2">{r.id}</code>
                    <div className="max-w-52 truncate text-xs text-muted">
                      {r.request.scenario}
                    </div>
                  </td>
                  <td className="py-2 pr-2 tabular-nums">
                    {r.result.mean_intent?.toFixed(2)}
                  </td>
                  <td className="py-2 pr-2">
                    {r.actuals?.intent != null ? (
                      <span className="tabular-nums text-good">
                        {r.actuals.intent.toFixed(2)}
                      </span>
                    ) : (
                      <input
                        type="number"
                        min={0}
                        max={1}
                        step={0.01}
                        placeholder="0.00"
                        value={drafts[r.id] ?? ""}
                        onChange={(e) =>
                          setDrafts((d) => ({ ...d, [r.id]: e.target.value }))
                        }
                        className="w-20 rounded-lg border border-hairline bg-surface-2 px-2 py-1 text-sm outline-none tabular-nums focus:border-accent/60"
                      />
                    )}
                  </td>
                  <td className="py-2 text-right">
                    {r.actuals?.intent == null && (
                      <motion.button
                        whileTap={{ scale: 0.96 }}
                        onClick={() => save(r.id)}
                        // `!drafts[r.id]` is falsy for the string "0", so an
                        // observed intent of exactly zero — a stimulus nobody
                        // acted on, which is a real and important outcome —
                        // could never be recorded, even though save() accepts
                        // it and the API validates 0 as in range.
                        disabled={(drafts[r.id] ?? "").trim() === ""}
                        className="rounded-lg border border-hairline px-3 py-1 text-xs text-ink-2 transition hover:border-accent/50 hover:text-ink disabled:opacity-40"
                      >
                        Record
                      </motion.button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function Stat({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div>
      <div className="text-xs uppercase tracking-wider text-muted">{label}</div>
      <div className="font-display text-2xl font-medium tracking-tight">{value}</div>
      {hint && <div className="text-xs text-muted">{hint}</div>}
    </div>
  );
}

/** An interval, rendered so that "we couldn't form one" and "it is very tight"
 *  never look the same. A zero-width interval is the quantisation of twin
 *  scores showing through, not precision — bootstrap_ci returns (0.5, 0.5) for
 *  a unanimous sample at any n — and a null one means the sample was too small
 *  to resample at all. Both used to degrade silently to a bare number. */
function Interval({ ci }: { ci?: [number, number] | null }) {
  if (!ci) return <span className="text-muted"> · interval n/a</span>;
  const [lo, hi] = ci;
  if (hi - lo <= 0) {
    return (
      <span className="text-muted" title="Every observation was identical, so the bootstrap has no sampling spread to report — this is not a tight interval.">
        {" "}· no observed spread
      </span>
    );
  }
  return (
    <span className="text-muted">
      {" "}
      · [{lo.toFixed(3)}, {hi.toFixed(3)}]
    </span>
  );
}

/**
 * The reliability diagram: where along the range the predictor drifts.
 *
 * A slope is one number for the whole range, so it cannot distinguish "wrong
 * everywhere by the same amount" from "fine in the middle, collapses at the
 * top" — and those call for opposite responses. The isotonic (PAV) fit is the
 * shape of the error as a function of what was predicted, which is the question
 * a calibration panel is actually being asked.
 *
 * The identity line is the reference: on it, predicted equals observed. The
 * curve sagging BELOW means over-prediction in that band, above means under.
 * Deviation is filled against the diagonal rather than left for the eye to
 * measure, because the gap IS the miscalibration term reported beside it.
 */
function ReliabilityDiagram({ knots }: { knots: [number, number][] }) {
  if (knots.length < 2) return null;
  const S = 100;
  const x = (v: number) => v * S;
  const y = (v: number) => S - v * S;
  const pts = knots.map(([p, a]) => `${x(p)},${y(a)}`).join(" ");
  // Closing the isotonic path back along the diagonal makes the enclosed area
  // the total miscalibration — the fill is the quantity, not a decoration.
  const area =
    `${x(knots[0][0])},${y(knots[0][0])} ` +
    knots.map(([p, a]) => `${x(p)},${y(a)}`).join(" ") +
    ` ${x(knots[knots.length - 1][0])},${y(knots[knots.length - 1][0])}`;

  return (
    <svg viewBox={`-2 -2 ${S + 4} ${S + 4}`} className="h-28 w-28 shrink-0" aria-hidden>
      <rect x={0} y={0} width={S} height={S} fill="rgb(var(--region-rgb) / 0.04)" />
      <line x1={0} y1={S} x2={S} y2={0} stroke="rgba(223,217,217,0.28)" strokeWidth={0.8} strokeDasharray="3 2" />
      <polygon points={area} fill="rgb(var(--region-rgb) / 0.18)" />
      <polyline points={pts} fill="none" stroke="rgb(var(--region-rgb))" strokeWidth={1.6} />
      {knots.map(([p, a], i) => (
        <circle key={i} cx={x(p)} cy={y(a)} r={1.6} fill="rgb(var(--region-rgb))" />
      ))}
    </svg>
  );
}

/**
 * CORP decomposition as one bar.
 *
 * Brier = miscalibration − discrimination + uncertainty, and the two terms
 * answer different questions: MCB is how wrong the predictions are, DSC is how
 * much signal they carry at all. A predictor can be perfectly calibrated and
 * still worthless (predict the base rate every time: MCB ≈ 0, DSC ≈ 0), which
 * is the failure a single score hides and the reason both are drawn.
 */
function CorpBar({ d }: { d: { miscalibration: number; discrimination: number; uncertainty: number } }) {
  const total = Math.max(d.miscalibration + d.discrimination, 1e-9);
  const parts = [
    { k: "MCB", v: d.miscalibration, c: "var(--color-critical)", t: "miscalibration — how far off the predictions are" },
    { k: "DSC", v: d.discrimination, c: "var(--color-good)", t: "discrimination — how much the predictions separate outcomes" },
  ];
  return (
    <div className="mt-1">
      <div className="flex h-1.5 w-full overflow-hidden">
        {parts.map((p) => (
          <span
            key={p.k}
            title={p.t}
            style={{ width: `${(p.v / total) * 100}%`, background: p.c }}
          />
        ))}
      </div>
      <div className="mt-0.5 font-mono text-[9px] text-muted">
        MCB {d.miscalibration.toFixed(3)} · DSC {d.discrimination.toFixed(3)} · UNC{" "}
        {d.uncertainty.toFixed(3)}
      </div>
    </div>
  );
}

function MetricColumn({ label, m }: { label: string; m: MetricCalibration | null }) {
  if (!m) {
    return (
      <div>
        <div className="text-xs uppercase tracking-wider text-muted">{label}</div>
        <div className="font-display text-2xl font-medium tracking-tight text-muted">—</div>
        <div className="text-xs text-muted">no actuals recorded</div>
      </div>
    );
  }
  return (
    <div className="min-w-52">
      <div className="text-xs uppercase tracking-wider text-muted">
        {label} <span className="normal-case">· n={m.n}</span>
      </div>
      <div className="font-display text-2xl font-medium tracking-tight">
        {m.mae.toFixed(3)}
        <span className="ml-1 text-xs font-normal text-muted">MAE</span>
      </div>
      <div className="mt-1 flex items-start gap-3">
        <div className="min-w-0 flex-1 space-y-0.5 font-mono text-[10px] leading-relaxed text-ink-2">
          <div>
            mae<Interval ci={m.mae_ci} />
          </div>
          <div>
            bias {m.bias > 0 ? "+" : ""}
            {m.bias.toFixed(3)}
            <Interval ci={m.bias_ci} />
          </div>
          <div>
            {/* The pair the docstring says is load-bearing: slope near 1 means
                the predictor tracks reality, slope near 0 means it is compressed
                toward its own mean regardless of how good the MAE looks. */}
            slope {m.slope != null ? m.slope.toFixed(2) : "n/a"}
            <Interval ci={m.slope_ci} />
          </div>
          {/* The gate, shown because the verdict above depends on it. The point
              estimate needs an assumption about which axis is noisier; this
              range holds without one, so it is the honest width. */}
          {m.slope_bracket_ci && (
            <div
              className="text-muted"
              title="Slopes consistent with the data under any assumption about which axis carries more measurement error. The verdict is asserted only when this whole range sits on one side of 1."
            >
              identified [{m.slope_bracket_ci[0].toFixed(2)},{" "}
              {m.slope_bracket_ci[1].toFixed(2)}]
              {m.slope_bracket_ci[0] <= 1 && m.slope_bracket_ci[1] >= 1 && (
                <span className="text-good"> · spans 1</span>
              )}
            </div>
          )}
          <div>
            r {m.r != null ? m.r.toFixed(2) : "n/a"}
            <Interval ci={m.r_ci} />
          </div>
          <div>
            {m.rmse != null && `rmse ${m.rmse.toFixed(3)}`}
            {m.slope_r2 != null && ` · R² ${m.slope_r2.toFixed(2)}`}
          </div>
          {m.brier_decomposition && <CorpBar d={m.brier_decomposition} />}
        </div>
        {m.reliability_curve && m.reliability_curve.length > 1 && (
          <div>
            <ReliabilityDiagram knots={m.reliability_curve} />
            <p className="mt-0.5 w-28 font-mono text-[8px] leading-tight text-muted">
              observed vs predicted · dashed = perfect
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
