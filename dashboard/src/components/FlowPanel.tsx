"use client";

import { useState } from "react";
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
import { CognitiveLoad, WalkAggregate } from "@/lib/api";
import { streamRun } from "@/lib/stream";
import { LoadToggle } from "./LoadToggle";
import FindingsPanel, { ResearchQuestion, type FindingsBlock } from "./FindingsPanel";
import { useEntranceEnabled } from "./Reveal";
import SimStage from "./sim/SimStage";
import type { GateEvent } from "./sim/FunnelSim";
import { SurvivalPanel } from "./StatReadouts";

const FunnelSim = dynamic(() => import("./sim/FunnelSim"), { ssr: false });

export default function FlowPanel({ personaCount }: { personaCount: number }) {
  const entrance = useEntranceEnabled();
  const [stepsText, setStepsText] = useState("");
  const [twins, setTwins] = useState(2);
  const [load, setLoad] = useState<CognitiveLoad>("high");
  const [question, setQuestion] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<WalkAggregate | null>(null);
  // live stream state — every gate decision of every twin, as it happens
  const [gateEvents, setGateEvents] = useState<GateEvent[]>([]);
  const [twinsDone, setTwinsDone] = useState(0);
  const [total, setTotal] = useState(0);

  const steps = stepsText
    .split("\n")
    .map((s) => s.trim())
    .filter(Boolean);

  async function run() {
    setBusy(true);
    setError(null);
    setResult(null);
    setGateEvents([]);
    setTwinsDone(0);
    try {
      await streamRun(
        "/swarm/walk/stream",
        { steps, twins_per_persona: twins, cognitive_load: load, research_question: question },
        (evt) => {
          if (evt.type === "start") setTotal(evt.total as number);
          else if (evt.type === "step") {
            setGateEvents((g) => [
              ...g,
              { gate: evt.step_index as number, action: evt.action as string },
            ]);
          } else if (evt.type === "twin_done") setTwinsDone((n) => n + 1);
          else if (evt.type === "done") setResult(evt.result as unknown as WalkAggregate);
          else if (evt.type === "error") setError(String(evt.detail));
        }
      );
    } catch (e) {
      setError(e instanceof Error ? e.message : "Walkthrough failed");
    } finally {
      setBusy(false);
    }
  }

  const funnel =
    result?.steps.map((s) => ({
      name: `${s.step_index + 1}. ${s.step.slice(0, 28)}${s.step.length > 28 ? "…" : ""}`,
      entered: s.entered,
      abandoned: s.abandoned,
    })) ?? [];

  return (
    <div className="card p-6">
      <div className="mb-1 flex items-baseline justify-between">
        <h2 className="font-display text-lg font-medium tracking-tight">
          Flow walkthrough
        </h2>
        <span className="text-xs text-muted">the context throttle in action</span>
      </div>
      <p className="lede mb-4">
        Describe a flow, one step per line (screen, section, moment). Twins walk it
        step-by-step; under high load their memory of earlier steps shrinks to{" "}
        <span className="text-ink-2">2 steps</span> — the 3 AM user.
      </p>

      <SimStage
        show={busy || !!result}
        busy={busy}
        runningLabel="agents walking the flow…"
        doneLabel="funnel complete"
        progress={
          busy
            ? `${twinsDone}${total ? `/${total}` : ""} twins · ${gateEvents.length} gate decisions`
            : result
              ? `${gateEvents.length} gate decisions observed`
              : undefined
        }
      >
        <FunnelSim steps={steps} events={gateEvents} />
      </SimStage>

      <textarea
        value={stepsText}
        onChange={(e) => setStepsText(e.target.value)}
        rows={5}
        placeholder={
          "Landing hero: 'Ship AI agents in minutes', one CTA\nSignup form: email + password + company size dropdown\nEmail verification interstitial\nOnboarding: pick a template from 12 options\nDashboard, empty state with 'Create first agent'"
        }
        className="w-full resize-y rounded-xl border border-hairline bg-surface-2 p-3.5 font-mono text-xs leading-relaxed outline-none placeholder:text-muted focus:border-accent/60 focus:ring-2 focus:ring-accent/20"
      />

      <ResearchQuestion
        value={question}
        onChange={setQuestion}
        disabled={busy}
        examples={["e.g. which step is the real drop-off, and why there?"]}
      />

      <div className="mt-3 flex flex-wrap items-center gap-3">
        <span className="text-xs text-muted tabular-nums">{steps.length} steps</span>
        <label className="flex items-center gap-2 text-xs text-muted">
          Twins / persona
          <input
            type="number"
            min={1}
            max={10}
            value={twins}
            onChange={(e) => setTwins(parseInt(e.target.value, 10) || 1)}
            className="w-16 rounded-lg border border-hairline bg-surface-2 px-2 py-1.5 text-sm text-ink outline-none tabular-nums"
          />
        </label>
        <LoadToggle value={load} onChange={setLoad} id="flow-load" />
        <motion.button
          whileTap={{ scale: 0.97 }}
          disabled={busy || steps.length < 2}
          onClick={run}
          className="ml-auto rounded-xl bg-accent px-5 py-2 font-display text-sm font-semibold text-white transition hover:brightness-110 disabled:opacity-40"
        >
          {busy ? "Walking…" : "Run walkthrough"}
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
          <FindingsPanel findings={result.findings} />
            <div className="mb-4 flex gap-8">
              <div>
                <div className="text-xs uppercase tracking-wider text-muted">Completion</div>
                <div className="font-display text-3xl font-medium tracking-tight">
                  {(result.completion_rate * 100).toFixed(0)}%
                </div>
              </div>
              <div>
                <div className="text-xs uppercase tracking-wider text-muted">Conversion</div>
                <div className="font-display text-3xl font-medium tracking-tight text-accent">
                  {(result.conversion_rate * 100).toFixed(0)}%
                </div>
              </div>
              <div className="self-end text-xs text-muted">
                {result.twin_count} twins · load {result.cognitive_load} · memory window{" "}
                {result.context_window_steps}
              </div>
            </div>

            <ResponsiveContainer width="100%" height={Math.max(160, funnel.length * 40)}>
              <BarChart data={funnel} layout="vertical" margin={{ left: 8, right: 40 }}>
                <CartesianGrid horizontal={false} stroke="#2c2c2a" />
                <XAxis type="number" allowDecimals={false} hide />
                <YAxis
                  type="category"
                  dataKey="name"
                  width={210}
                  tickLine={false}
                  axisLine={false}
                  tick={{ fill: "#898781", fontSize: 11 }}
                />
                <Bar dataKey="entered" fill="#3987e5" radius={[0, 4, 4, 0]} barSize={14} isAnimationActive={false}>
                  <LabelList
                    dataKey="abandoned"
                    position="right"
                    formatter={(v: number) => (v > 0 ? `−${v}` : "")}
                    style={{ fill: "#d95926", fontSize: 12 }}
                  />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            <p className="mt-1 text-xs text-muted">
              Bars: twins reaching each step. Orange −n: abandons at that step.
            </p>

            {result.survival && (
              <div className="mt-4">
                <SurvivalPanel survival={result.survival} steps={result.steps} />
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
