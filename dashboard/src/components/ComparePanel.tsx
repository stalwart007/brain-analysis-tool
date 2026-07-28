"use client";

import { useRef, useState } from "react";
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
import dynamic from "next/dynamic";
import { BayesianResult, CognitiveLoad, CompareResult } from "@/lib/api";
import { streamRun } from "@/lib/stream";
import { LoadToggle } from "./LoadToggle";
import { useEntranceEnabled } from "./Reveal";
import SimStage from "./sim/SimStage";
import type { ArenaArrival } from "./sim/ArenaSim";
import { BayesianPanel } from "./StatReadouts";

const ArenaSim = dynamic(() => import("./sim/ArenaSim"), { ssr: false });

interface Variant {
  name: string;
  scenario: string;
}

interface EarlyStop {
  spent: number;
  budget: number;
  saved: number;
  reason: string;
}

export default function ComparePanel({ personaCount }: { personaCount: number }) {
  const entrance = useEntranceEnabled();
  const [variants, setVariants] = useState<Variant[]>([
    { name: "A (control)", scenario: "" },
    { name: "B", scenario: "" },
  ]);
  const [twins, setTwins] = useState(3);
  const [load, setLoad] = useState<CognitiveLoad>("low");
  const [adaptive, setAdaptive] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<CompareResult | null>(null);
  // live stream state — each entry is one real agent that landed
  const [arrivals, setArrivals] = useState<ArenaArrival[]>([]);
  const [total, setTotal] = useState(0);
  const [liveBayes, setLiveBayes] = useState<BayesianResult | null>(null);
  const [waveNote, setWaveNote] = useState<string | null>(null);
  const [earlyStop, setEarlyStop] = useState<EarlyStop | null>(null);
  const laneOf = useRef<Map<string, number>>(new Map());

  const canRun =
    !busy && personaCount > 0 && variants.every((v) => v.scenario.trim().length > 0);

  function update(i: number, patch: Partial<Variant>) {
    setVariants((vs) => vs.map((v, j) => (j === i ? { ...v, ...patch } : v)));
  }

  function addVariant() {
    const letter = String.fromCharCode(65 + variants.length);
    setVariants((vs) => [...vs, { name: letter, scenario: "" }]);
  }

  async function run() {
    setBusy(true);
    setError(null);
    setResult(null);
    setArrivals([]);
    setLiveBayes(null);
    setWaveNote(null);
    setEarlyStop(null);
    laneOf.current = new Map(variants.map((v, i) => [v.name, i]));
    try {
      await streamRun(
        "/swarm/compare/stream",
        { variants, twins_per_persona: twins, cognitive_load: load, adaptive },
        (evt) => {
          if (evt.type === "start") setTotal(evt.total as number);
          else if (evt.type === "agent") {
            const lane = laneOf.current.get(evt.variant as string) ?? 0;
            setArrivals((a) => [...a, { lane, convert: evt.action === "convert" }]);
          } else if (evt.type === "wave") {
            const alloc = evt.allocation as Record<string, number>;
            const parts = Object.entries(alloc)
              .filter(([, c]) => c > 0)
              .map(([n, c]) => `${n}:${c}`)
              .join(" ");
            setWaveNote(`wave ${(evt.index as number) + 1} — bandit routed ${parts}`);
          } else if (evt.type === "posterior") {
            setLiveBayes(evt.bayesian as BayesianResult);
          } else if (evt.type === "early_stop") {
            setEarlyStop(evt as unknown as EarlyStop);
          } else if (evt.type === "done") {
            setResult(evt.result as unknown as CompareResult);
          } else if (evt.type === "error") {
            setError(String(evt.detail));
          }
        }
      );
    } catch (e) {
      setError(e instanceof Error ? e.message : "Compare failed");
    } finally {
      setBusy(false);
    }
  }

  const bars =
    result?.ranking.map((r) => ({
      name: r.name,
      intent: r.mean_intent,
      winner: r.name === result.winner,
    })) ?? [];

  return (
    <div className="card p-6">
      <div className="mb-1 flex items-baseline justify-between">
        <h2 className="font-display text-lg font-medium tracking-tight">
          Zero-shot A/B
        </h2>
        <span className="text-xs text-muted">
          every variant faces the same swarm, same state
        </span>
      </div>
      <p className="lede mb-4">
        Paste 2–8 stimulus variants (ad copy, hooks, layouts described in words).
        The first is the control for lift.
      </p>

      <SimStage
        show={busy || !!result}
        busy={busy}
        runningLabel={
          adaptive ? "thompson bandit allocating agents…" : "agents entering the arena…"
        }
        doneLabel={earlyStop ? "arena settled — stopped early" : "arena settled"}
        progress={
          busy
            ? `${arrivals.length}${total ? `/${total}` : ""} agents`
            : result
              ? `${arrivals.length || result.twin_count_per_variant * variants.length} agents`
              : undefined
        }
      >
        <ArenaSim
          variants={variants.map((v) => v.name)}
          arrivals={arrivals}
          lanes={
            result
              ? variants.map((v) => {
                  const r = result.ranking.find((x) => x.name === v.name);
                  return {
                    name: v.name,
                    intent: r?.mean_intent ?? 0.5,
                    winner: v.name === result.winner,
                  };
                })
              : undefined
          }
        />
      </SimStage>

      {/* live bandit telemetry while the stream runs */}
      {busy && (waveNote || liveBayes) && (
        <div className="mb-4 flex flex-wrap items-center gap-3 rounded-xl border border-hairline bg-surface-2/60 px-3 py-2 font-mono text-[11px]">
          {waveNote && <span className="text-accent/90">{waveNote}</span>}
          {liveBayes?.winner && (
            <span className="text-ink-2">
              live posterior: {liveBayes.winner} P(best){" "}
              {Math.round(
                (liveBayes.arms.find((a) => a.name === liveBayes.winner)?.p_best ?? 0) * 100
              )}
              %
            </span>
          )}
        </div>
      )}

      <div className="space-y-3">
        {variants.map((v, i) => (
          <div key={i} className="flex gap-2">
            <input
              value={v.name}
              onChange={(e) => update(i, { name: e.target.value })}
              className="w-28 shrink-0 rounded-xl border border-hairline bg-surface-2 px-3 py-2 text-sm outline-none focus:border-accent/60"
              aria-label={`Variant ${i + 1} name`}
            />
            <textarea
              value={v.scenario}
              onChange={(e) => update(i, { scenario: e.target.value })}
              placeholder={i === 0 ? "Control stimulus…" : "Variant stimulus…"}
              rows={2}
              className="min-h-10 flex-1 resize-y rounded-xl border border-hairline bg-surface-2 px-3 py-2 text-sm outline-none placeholder:text-muted focus:border-accent/60"
            />
            {variants.length > 2 && (
              <button
                onClick={() => setVariants((vs) => vs.filter((_, j) => j !== i))}
                className="self-start rounded-lg px-2 py-2 text-muted transition hover:text-critical"
                aria-label={`Remove variant ${v.name}`}
              >
                ✕
              </button>
            )}
          </div>
        ))}
      </div>

      <div className="mt-3 flex flex-wrap items-center gap-3">
        {variants.length < 8 && (
          <button
            onClick={addVariant}
            className="rounded-lg border border-dashed border-hairline px-3 py-1.5 text-xs text-muted transition hover:border-accent/50 hover:text-ink-2"
          >
            + variant
          </button>
        )}
        <label className="flex items-center gap-2 text-xs text-muted">
          Twins / persona
          <input
            type="number"
            min={1}
            max={20}
            value={twins}
            onChange={(e) => setTwins(parseInt(e.target.value, 10) || 1)}
            className="w-16 rounded-lg border border-hairline bg-surface-2 px-2 py-1.5 text-sm text-ink outline-none tabular-nums"
          />
        </label>
        <LoadToggle value={load} onChange={setLoad} id="compare-load" />
        <label
          className="flex cursor-pointer items-center gap-2 text-xs text-muted"
          title="Thompson-sampling bandit: waves of agents are routed toward the variant most likely to be winning, and the run stops early once the posterior is conclusive."
        >
          <input
            type="checkbox"
            checked={adaptive}
            onChange={(e) => setAdaptive(e.target.checked)}
            className="accent-[#3987e5]"
          />
          adaptive (bandit)
        </label>
        <motion.button
          whileTap={{ scale: 0.97 }}
          disabled={!canRun}
          onClick={run}
          className="ml-auto rounded-xl bg-accent px-5 py-2 font-display text-sm font-semibold text-white transition hover:brightness-110 disabled:opacity-40"
        >
          {busy ? "Comparing…" : "Run compare"}
        </motion.button>
      </div>

      {error && <p className="mt-3 text-sm text-critical">{error}</p>}

      <AnimatePresence>
        {result && (
          <motion.div
            initial={entrance ? { opacity: 0, y: 12 } : false}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className="mt-6"
          >
            <p className="mb-3 text-sm">
              Winner:{" "}
              <span className="font-display font-semibold text-accent">
                {result.winner}
              </span>
              {result.ranking[0]?.lift_vs_control_pct != null && (
                <span className="text-ink-2">
                  {" "}
                  · {result.ranking[0].lift_vs_control_pct > 0 ? "+" : ""}
                  {result.ranking[0].lift_vs_control_pct}% intent vs control
                </span>
              )}
            </p>

            {earlyStop && (
              <div className="mb-3 rounded-xl border border-good/30 bg-good/5 px-3 py-2 text-xs">
                <span className="font-display font-semibold text-good">
                  OPTIMAL STOPPING
                </span>{" "}
                <span className="text-ink-2">
                  spent {earlyStop.spent} of {earlyStop.budget} agents — saved{" "}
                  {earlyStop.saved} ({Math.round((earlyStop.saved / earlyStop.budget) * 100)}
                  % of budget) because {earlyStop.reason}
                </span>
              </div>
            )}

            {/* the naive lift above is only meaningful if the posterior agrees */}
            {result.bayesian && (
              <div className="mb-4">
                <BayesianPanel bayes={result.bayesian} />
              </div>
            )}

            <ResponsiveContainer width="100%" height={Math.max(140, bars.length * 44)}>
              <BarChart data={bars} layout="vertical" margin={{ left: 8, right: 40 }}>
                <CartesianGrid horizontal={false} stroke="#2c2c2a" />
                <XAxis type="number" domain={[0, 1]} hide />
                <YAxis
                  type="category"
                  dataKey="name"
                  width={110}
                  tickLine={false}
                  axisLine={false}
                  tick={{ fill: "#c3c2b7", fontSize: 12 }}
                />
                <Bar dataKey="intent" fill="#3987e5" radius={[0, 4, 4, 0]} barSize={18} isAnimationActive={false}>
                  <LabelList
                    dataKey="intent"
                    position="right"
                    formatter={(v: number) => v.toFixed(2)}
                    style={{ fill: "#c3c2b7", fontSize: 12 }}
                  />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
