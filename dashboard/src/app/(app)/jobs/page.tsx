"use client";

import { useCallback, useEffect, useState } from "react";
import { motion } from "framer-motion";
import { api, JobRow } from "@/lib/api";
import Reveal from "@/components/Reveal";
import { LoadToggle } from "@/components/LoadToggle";
import type { CognitiveLoad } from "@/lib/api";

const STATUS_STYLE: Record<string, string> = {
  queued: "text-muted",
  running: "text-accent",
  done: "text-good",
  error: "text-critical",
};

export default function JobsPage() {
  const [jobs, setJobs] = useState<JobRow[]>([]);
  const [scenario, setScenario] = useState("");
  const [twins, setTwins] = useState(20);
  const [load, setLoad] = useState<CognitiveLoad>("low");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const refresh = useCallback(() => {
    api.jobs().then(setJobs).catch(() => {});
  }, []);

  useEffect(() => {
    refresh();
    // poll while anything is in flight
    const t = setInterval(refresh, 4000);
    return () => clearInterval(t);
  }, [refresh]);

  async function submit() {
    setBusy(true);
    setError(null);
    try {
      await api.createJob("batch_swarm", {
        scenario,
        twins_per_persona: twins,
        cognitive_load: load,
      });
      setScenario("");
      refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to queue job");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="mx-auto max-w-4xl px-6 py-10">
      <Reveal>
        <div className="flex items-baseline gap-3">
          <span className="tag text-muted">Sector BTCH</span>
          <span className="h-px flex-1 bg-hairline" />
          <span className="panel-index">04</span>
        </div>
        <h1 className="display reg mt-2 text-[clamp(2.4rem,6vw,4.4rem)] text-bone">Batch Jobs</h1>
        <p className="mt-1 max-w-xl text-sm text-muted">
          Large non-interactive swarms run through the OpenAI Batch API at half
          the token cost. Jobs are durable — they survive a server restart and
          resume automatically.
        </p>
      </Reveal>

      <Reveal index={1} className="mt-6 block">
        <div className="card p-6">
          <h2 className="mb-3 font-display text-lg font-medium tracking-tight">
            Queue a batch swarm
          </h2>
          <textarea
            value={scenario}
            onChange={(e) => setScenario(e.target.value)}
            rows={3}
            placeholder="Stimulus to run against a large twin sample…"
            className="w-full resize-y rounded-xl border border-hairline bg-surface-2 p-3.5 text-sm outline-none placeholder:text-muted focus:border-accent/60 focus:ring-2 focus:ring-accent/20"
          />
          <div className="mt-3 flex flex-wrap items-center gap-3">
            <label className="flex items-center gap-2 text-xs text-muted">
              Twins / persona
              <input
                type="number"
                min={1}
                max={200}
                value={twins}
                onChange={(e) => setTwins(parseInt(e.target.value, 10) || 1)}
                className="w-20 rounded-lg border border-hairline bg-surface-2 px-2 py-1.5 text-sm text-ink outline-none tabular-nums"
              />
            </label>
            <LoadToggle value={load} onChange={setLoad} id="jobs-load" />
            <motion.button
              whileTap={{ scale: 0.97 }}
              disabled={busy || !scenario.trim()}
              onClick={submit}
              className="ml-auto rounded-xl bg-accent px-5 py-2 font-display text-sm font-semibold text-white transition hover:brightness-110 disabled:opacity-40"
            >
              {busy ? "Queuing…" : "Queue job"}
            </motion.button>
          </div>
          {error && <p className="mt-3 text-sm text-critical">{error}</p>}
        </div>
      </Reveal>

      <Reveal index={2} className="mt-6 block">
        <div className="card p-6">
          <h2 className="mb-4 font-display text-lg font-medium tracking-tight">History</h2>
          {jobs.length === 0 ? (
            <p className="text-sm text-muted">No jobs yet.</p>
          ) : (
            <table className="w-full text-sm">
              <thead className="text-left text-xs uppercase tracking-wider text-muted">
                <tr>
                  <th className="pb-2 pr-2 font-medium">Job</th>
                  <th className="pb-2 pr-2 font-medium">Kind</th>
                  <th className="pb-2 pr-2 font-medium">Status</th>
                  <th className="pb-2 font-medium">Result / error</th>
                </tr>
              </thead>
              <tbody>
                {jobs.map((j) => (
                  <tr key={j.id} className="border-t border-hairline/60">
                    <td className="py-2 pr-2">
                      <code className="text-xs text-ink-2">{j.id}</code>
                    </td>
                    <td className="py-2 pr-2 text-xs text-muted">{j.kind}</td>
                    <td className="py-2 pr-2">
                      <span className={`text-xs font-medium ${STATUS_STYLE[j.status]}`}>
                        {j.status === "running" && "◍ "}
                        {j.status}
                      </span>
                    </td>
                    <td className="py-2 text-xs text-muted">
                      {j.run_id ? (
                        <code className="text-ink-2">run {j.run_id}</code>
                      ) : j.error ? (
                        <span className="text-critical">{j.error}</span>
                      ) : (
                        "—"
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </Reveal>
    </div>
  );
}
