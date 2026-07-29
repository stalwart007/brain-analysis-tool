"use client";

import { useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import dynamic from "next/dynamic";
import { CognitiveLoad, ObjectionResult } from "@/lib/api";
import { streamRun } from "@/lib/stream";
import { LoadToggle } from "./LoadToggle";
import { useEntranceEnabled } from "./Reveal";
import SimStage from "./sim/SimStage";
import type { SonarPing } from "./sim/SonarSim";

const SonarSim = dynamic(() => import("./sim/SonarSim"), { ssr: false });

const SENTIMENT_COLOR: Record<string, string> = {
  positive: "#12c512",
  neutral: "#898781",
  negative: "#e66767",
};

export default function ObjectionPanel({ personaCount }: { personaCount: number }) {
  const entrance = useEntranceEnabled();
  const [pitch, setPitch] = useState("");
  const [twins, setTwins] = useState(3);
  const [load, setLoad] = useState<CognitiveLoad>("low");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ObjectionResult | null>(null);
  // live stream state — one ping per streamed twin verdict
  const [pings, setPings] = useState<SonarPing[]>([]);
  const [phase, setPhase] = useState<string | null>(null);
  const [streamTotal, setStreamTotal] = useState(0);

  async function run() {
    setBusy(true);
    setError(null);
    setResult(null);
    setPings([]);
    setPhase(null);
    try {
      await streamRun(
        "/studies/objection/stream",
        { pitch, twins_per_persona: twins, cognitive_load: load },
        (evt) => {
          if (evt.type === "start") setStreamTotal(evt.total as number);
          else if (evt.type === "agent") {
            setPings((p) => [
              ...p,
              { dealbreaker: !!evt.dealbreaker, sentiment: String(evt.sentiment) },
            ]);
          } else if (evt.type === "clustering") setPhase("clustering themes…");
          else if (evt.type === "done") setResult(evt.result as unknown as ObjectionResult);
          else if (evt.type === "error") setError(String(evt.detail));
        }
      );
    } catch (e) {
      setError(e instanceof Error ? e.message : "Objection scan failed");
    } finally {
      setBusy(false);
      setPhase(null);
    }
  }

  const total = result ? result.twin_count : 0;
  const sentiments = result ? Object.entries(result.sentiment_distribution) : [];

  return (
    <div className="card p-6">
      <div className="mb-1 flex items-baseline justify-between">
        <h2 className="font-display text-lg font-medium tracking-tight">Objection Radar</h2>
        <span className="text-xs text-muted">stress-test a pitch or claim</span>
      </div>
      <p className="lede mb-4">
        Paste a pitch, headline, or claim. Each twin surfaces its single biggest
        objection and whether it&apos;s a dealbreaker — you get the ranked doubts to
        address before you ship the copy.
      </p>

      <SimStage
        show={busy || !!result}
        busy={busy}
        runningLabel={phase ?? "sweeping for objections…"}
        doneLabel={result?.theme_map ? "semantic map (kernel PCA)" : "objection map"}
        progress={
          busy
            ? `${pings.length}${streamTotal ? `/${streamTotal}` : ""} verdicts`
            : result
              ? `${result.twin_count} twins · ${result.top_objections.length} themes`
              : undefined
        }
        height={280}
      >
        <SonarSim
          active={busy}
          pings={pings}
          blips={
            result
              ? result.top_objections.map((o, i) => {
                  const mapped = result.theme_map?.find(
                    (m) => m.label === o.objection.slice(0, 80)
                  );
                  return {
                    count: o.count,
                    dealbreaker: o.dealbreaker_count > 0,
                    x: mapped?.x ?? result.theme_map?.[i]?.x,
                    y: mapped?.y ?? result.theme_map?.[i]?.y,
                    label: o.objection,
                  };
                })
              : undefined
          }
        />
      </SimStage>

      <textarea
        value={pitch}
        onChange={(e) => setPitch(e.target.value)}
        rows={3}
        placeholder="e.g. 'Cancel anytime. The only AI assistant that never sends your data to the cloud — everything runs on your device. Just $12/mo.'"
        className="w-full resize-y rounded-xl border border-hairline bg-surface-2 p-3.5 text-sm outline-none placeholder:text-muted focus:border-accent/60 focus:ring-2 focus:ring-accent/20"
      />

      <div className="mt-3 flex flex-wrap items-center gap-3">
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
        <LoadToggle value={load} onChange={setLoad} id="objection-load" />
        <motion.button
          whileTap={{ scale: 0.97 }}
          disabled={busy || !pitch.trim()}
          onClick={run}
          className="ml-auto rounded-xl bg-accent px-5 py-2 font-display text-sm font-semibold text-white transition hover:brightness-110 disabled:opacity-40"
        >
          {busy ? "Scanning…" : "Run objection scan"}
        </motion.button>
      </div>

      {error && <p className="mt-3 text-sm text-critical">{error}</p>}

      <AnimatePresence>
        {result && (
          <motion.div
            initial={entrance ? { opacity: 0, y: 12 } : false}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className="mt-6 space-y-5"
          >
            <div className="flex flex-wrap gap-8">
              <div>
                <div className="text-xs uppercase tracking-wider text-muted">Dealbreaker rate</div>
                <div className="font-display text-3xl font-medium tracking-tight text-critical">
                  {(result.dealbreaker_rate * 100).toFixed(0)}%
                </div>
              </div>
              <div>
                <div className="text-xs uppercase tracking-wider text-muted">Mean recommend</div>
                <div className="font-display text-3xl font-medium tracking-tight text-accent">
                  {result.mean_recommend.toFixed(2)}
                </div>
              </div>
              <div className="min-w-40 flex-1">
                <div className="mb-1 text-xs uppercase tracking-wider text-muted">Sentiment</div>
                {/* single stacked bar, direct-labeled — no legend needed */}
                <div className="flex h-6 overflow-hidden rounded-lg">
                  {sentiments.map(([k, v]) => (
                    <div
                      key={k}
                      title={`${k}: ${v}`}
                      style={{ width: `${(v / total) * 100}%`, background: SENTIMENT_COLOR[k] ?? "#898781" }}
                    />
                  ))}
                </div>
                <div className="mt-1 flex gap-3 text-xs text-muted">
                  {sentiments.map(([k, v]) => (
                    <span key={k} className="flex items-center gap-1">
                      <span className="inline-block h-2 w-2 rounded-full" style={{ background: SENTIMENT_COLOR[k] ?? "#898781" }} />
                      {k} {v}
                    </span>
                  ))}
                </div>
              </div>
            </div>

            <div>
              <div className="mb-2 flex items-center gap-2">
                <span className="hud-label">Objection themes (⚑ = dealbreakers)</span>
                {result.clustering_method === "semantic" && (
                  <span className="border border-hairline px-2 py-0.5 font-mono text-[10px] uppercase tracking-wider text-ink-2">
                    semantically clustered
                  </span>
                )}
              </div>
              <ul className="space-y-2">
                {result.top_objections.map((o, i) => (
                  <li key={i} className="flex items-start gap-3 text-sm">
                    <span className="mt-0.5 inline-flex h-5 min-w-5 items-center justify-center rounded-full bg-white/[0.06] px-1.5 text-xs tabular-nums text-ink-2">
                      {o.count}
                    </span>
                    <span className="flex-1">
                      <span className="text-ink-2">{o.objection}</span>
                      {o.variants && o.variants.length > 0 && (
                        <details className="mt-0.5">
                          <summary className="cursor-pointer font-mono text-[10px] text-muted hover:text-ink-2">
                            +{o.variants.length} merged phrasing
                            {o.variants.length === 1 ? "" : "s"}
                          </summary>
                          <ul className="mt-1 space-y-0.5 border-l border-hairline pl-2">
                            {o.variants.map((v, k) => (
                              <li key={k} className="text-[11px] leading-tight text-muted">
                                {v}
                              </li>
                            ))}
                          </ul>
                        </details>
                      )}
                    </span>
                    {o.dealbreaker_count > 0 && (
                      <span className="shrink-0 text-xs text-critical">⚑ {o.dealbreaker_count}</span>
                    )}
                  </li>
                ))}
                {result.top_objections.length === 0 && (
                  <li className="text-sm text-good">No substantive objections surfaced — well received.</li>
                )}
              </ul>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
