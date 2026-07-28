"use client";

import { useCallback, useEffect, useState } from "react";
import { motion } from "framer-motion";
import { api, CalibrationReport, SwarmRunRow } from "@/lib/api";

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

      <div className="mb-5 flex gap-8">
        <Stat label="Runs with actuals" value={report ? String(report.runs_with_actuals) : "—"} />
        <Stat
          label="Intent MAE"
          value={report?.intent ? report.intent.mae.toFixed(3) : "—"}
          hint={report?.intent ? `n=${report.intent.n}` : undefined}
        />
        <Stat
          label="Intent bias"
          value={
            report?.intent
              ? `${report.intent.bias > 0 ? "+" : ""}${report.intent.bias.toFixed(3)}`
              : "—"
          }
          hint={report?.intent ? (report.intent.bias > 0 ? "over-predicts" : "under-predicts") : undefined}
        />
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
                        disabled={!drafts[r.id]}
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
