"use client";

import { useState } from "react";
import dynamic from "next/dynamic";
import { AnimatePresence, motion } from "framer-motion";
import {
  Bar,
  BarChart,
  CartesianGrid,
  LabelList,
  ResponsiveContainer,
  XAxis,
  YAxis,
} from "recharts";
import { SwarmAggregate } from "@/lib/api";
import { streamSwarm } from "@/lib/stream";
import { useEntranceEnabled } from "./Reveal";
import { CIStat, ConsensusBadge, SegmentsPanel } from "./StatReadouts";

// 3D network is lazy — kept out of the initial dashboard bundle.
const SwarmNetwork = dynamic(() => import("./SwarmNetwork"), { ssr: false });

const LOADS = ["low", "medium", "high"] as const;
const ACTION_ORDER = ["convert", "continue", "hesitate", "abandon"];
const ACTION_DOT: Record<string, string> = {
  convert: "#0ca30c",
  continue: "#3987e5",
  hesitate: "#eda100",
  abandon: "#e66767",
};

export default function SwarmPanel({
  personaCount,
  onRan,
}: {
  personaCount: number;
  onRan: () => void;
}) {
  const entrance = useEntranceEnabled();
  const [scenario, setScenario] = useState("");
  const [twins, setTwins] = useState(3);
  const [load, setLoad] = useState<(typeof LOADS)[number]>("low");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<SwarmAggregate | null>(null);
  // live-streaming state: verdicts grows one entry per agent that reports in.
  const [live, setLive] = useState<{ total: number; verdicts: string[] } | null>(null);

  async function run() {
    setBusy(true);
    setError(null);
    setResult(null);
    setLive({ total: personaCount * twins, verdicts: [] });
    try {
      await streamSwarm({ scenario, twins_per_persona: twins, cognitive_load: load }, (evt) => {
        if (evt.type === "start") {
          setLive({ total: evt.total, verdicts: [] });
        } else if (evt.type === "agent") {
          setLive((s) => (s ? { ...s, verdicts: [...s.verdicts, evt.action] } : s));
        } else if (evt.type === "done") {
          setResult(evt.result);
          onRan();
        } else if (evt.type === "error") {
          setError(evt.detail);
        }
      });
    } catch (e) {
      setError(e instanceof Error ? e.message : "Swarm run failed");
    } finally {
      setBusy(false);
    }
  }

  // verdicts fed to the 3D graph: live during the stream, full once done.
  const liveVerdicts = result
    ? (result.reactions as { action: string }[]).map((r) => r.action)
    : live?.verdicts;
  const totalCount = result ? result.twin_count : live?.total ?? personaCount * twins;
  const arrived = result ? result.twin_count : live?.verdicts.length ?? 0;

  const bars = result
    ? ACTION_ORDER.map((action) => ({
        action,
        count: result.action_distribution[action] ?? 0,
      }))
    : [];

  return (
    <div className="card flex h-full flex-col p-6">
      <div className="mb-4 flex items-baseline justify-between">
        <h2 className="font-display text-lg font-medium tracking-tight">Forecast response</h2>
        <span className="text-xs text-muted">
          {personaCount > 0
            ? `${personaCount} persona${personaCount === 1 ? "" : "s"} seeded`
            : "no telemetry yet — audience will be inferred from your stimulus"}
        </span>
      </div>

      {/* live swarm network — appears while running and stays to show verdicts */}
      <AnimatePresence>
        {(busy || result) && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 300 }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ type: "spring", stiffness: 200, damping: 28 }}
            className="mb-4 overflow-hidden rounded-xl border border-hairline bg-black/30"
          >
            <div className="relative h-[300px]">
              <SwarmNetwork count={totalCount} verdicts={liveVerdicts} />
              <div className="pointer-events-none absolute left-3 top-3 flex items-center gap-2 text-xs">
                <span
                  className={`neon-dot inline-block h-2 w-2 rounded-full ${
                    busy ? "pulse-dot bg-accent" : "bg-good"
                  }`}
                />
                <span className="text-ink-2">
                  {busy ? "agents reporting in…" : "verdicts in"}
                </span>
                <span className="font-mono text-[11px] tabular-nums text-neon">
                  {arrived}/{totalCount}
                </span>
              </div>
              {result && (
                <div className="pointer-events-none absolute bottom-3 left-3 flex flex-wrap gap-x-3 gap-y-1 text-[11px]">
                  {ACTION_ORDER.map((a) => (
                    <span key={a} className="flex items-center gap-1 text-muted">
                      <span
                        className="inline-block h-2 w-2 rounded-full"
                        style={{ background: ACTION_DOT[a] }}
                      />
                      {a}
                    </span>
                  ))}
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <textarea
        value={scenario}
        onChange={(e) => setScenario(e.target.value)}
        placeholder="Paste the stimulus — an ad variant, a timestamped transcript, or a described UI flow…"
        className="min-h-24 w-full resize-y rounded-xl border border-hairline bg-surface-2 p-3.5 text-sm outline-none transition placeholder:text-muted focus:border-accent/60 focus:ring-2 focus:ring-accent/20"
      />

      <div className="mt-3 flex flex-wrap items-center gap-3">
        <label className="flex items-center gap-2 text-xs text-muted">
          Twins / persona
          <input
            type="number"
            min={1}
            max={50}
            value={twins}
            onChange={(e) => setTwins(parseInt(e.target.value, 10) || 1)}
            className="w-16 rounded-lg border border-hairline bg-surface-2 px-2 py-1.5 text-sm text-ink outline-none tabular-nums"
          />
        </label>

        <div className="flex overflow-hidden rounded-lg border border-hairline text-xs">
          {LOADS.map((l) => (
            <button
              key={l}
              onClick={() => setLoad(l)}
              className={`relative px-3 py-1.5 capitalize transition ${
                load === l ? "text-ink" : "text-muted hover:text-ink-2"
              }`}
            >
              {load === l && (
                <motion.span
                  layoutId="load-pill"
                  className="absolute inset-0 bg-white/[0.07]"
                  transition={{ type: "spring", stiffness: 400, damping: 34 }}
                />
              )}
              <span className="relative">{l}</span>
            </button>
          ))}
        </div>
        <span className="text-xs text-muted">cognitive load</span>

        <motion.button
          whileTap={{ scale: 0.97 }}
          disabled={busy || !scenario.trim()}
          onClick={run}
          className="ml-auto rounded-xl bg-accent px-5 py-2 font-display text-sm font-semibold text-white transition hover:brightness-110 disabled:opacity-40"
        >
          {busy ? "Forecasting…" : "Forecast response"}
        </motion.button>
      </div>

      {error && <p className="mt-3 text-sm text-critical">{error}</p>}

      <AnimatePresence>
        {result && (
          <motion.div
            initial={entrance ? { opacity: 0, y: 12 } : false}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className="mt-5 grid gap-4 sm:grid-cols-[auto_1fr]"
          >
            <div className="space-y-3">
              <CIStat
                label="Engagement"
                value={result.mean_engagement}
                ci={result.engagement_ci}
              />
              <CIStat label="Intent" value={result.mean_intent} ci={result.intent_ci} accent />
              <ConsensusBadge
                consensus={result.consensus}
                polarization={result.polarization}
                bimodality={result.bimodality}
              />
              <p className="max-w-44 text-xs text-muted">
                {result.twin_count} twins · load {result.cognitive_load}
              </p>
              {/* A silent shortfall is the whole bug: every mean and interval
                  above is computed over the survivors, and a non-random
                  subsample is not the sample the reader thinks they are
                  looking at. Say so where the numbers are. */}
              {!!result.twins_failed && (
                <p
                  className="max-w-44 text-xs text-critical"
                  title="These twins errored or returned unusable output. The statistics above are computed over the survivors only."
                >
                  {result.twins_failed} of {result.twins_requested} failed —
                  figures cover survivors
                </p>
              )}
            </div>

            <div className="min-h-44">
              <ResponsiveContainer width="100%" height={180}>
                <BarChart data={bars} layout="vertical" margin={{ left: 8, right: 28 }}>
                  <CartesianGrid horizontal={false} stroke="#2c2c2a" />
                  <XAxis type="number" hide />
                  <YAxis
                    type="category"
                    dataKey="action"
                    width={72}
                    tickLine={false}
                    axisLine={false}
                    tick={{ fill: "#898781", fontSize: 12 }}
                  />
                  <Bar
                    dataKey="count"
                    fill="#3987e5"
                    radius={[0, 4, 4, 0]}
                    barSize={16}
                    isAnimationActive={false}
                  >
                    <LabelList
                      dataKey="count"
                      position="right"
                      style={{ fill: "#c3c2b7", fontSize: 12 }}
                    />
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            {result.segments && (
              <div className="sm:col-span-2">
                <SegmentsPanel segments={result.segments} />
              </div>
            )}

            {(result.top_dropoff_points.length > 0 || result.top_frictions.length > 0) && (
              <div className="sm:col-span-2">
                {result.top_dropoff_points.length > 0 && (
                  <>
                    <h3 className="mb-1 text-xs font-medium uppercase tracking-wider text-muted">
                      Drop-off points
                    </h3>
                    <ul className="mb-3 space-y-1 text-sm text-ink-2">
                      {result.top_dropoff_points.map((d) => (
                        <li key={d}>· {d}</li>
                      ))}
                    </ul>
                  </>
                )}
                <h3 className="mb-1 text-xs font-medium uppercase tracking-wider text-muted">
                  Top frictions
                </h3>
                <ul className="space-y-1 text-sm text-ink-2">
                  {result.top_frictions.map((f) => (
                    <li key={f}>· {f}</li>
                  ))}
                </ul>
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

