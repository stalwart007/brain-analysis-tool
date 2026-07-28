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
      <div className="mt-1 space-y-0.5 font-mono text-[10px] leading-relaxed text-ink-2">
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
          {m.slope_r2 != null && ` (R² ${m.slope_r2.toFixed(2)})`}
          {m.r != null && ` · r ${m.r.toFixed(2)}`}
          {m.rmse != null && ` · rmse ${m.rmse.toFixed(3)}`}
        </div>
      </div>
    </div>
  );
}
