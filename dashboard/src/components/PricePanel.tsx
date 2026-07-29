"use client";

import { useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import dynamic from "next/dynamic";
import { CognitiveLoad, PriceSensitivityResult } from "@/lib/api";
import { streamRun } from "@/lib/stream";
import { LoadToggle } from "./LoadToggle";
import { useEntranceEnabled } from "./Reveal";
import SimStage from "./sim/SimStage";
import { ElasticityPanel } from "./StatReadouts";

const DemandSim = dynamic(() => import("./sim/DemandSim"), { ssr: false });

export default function PricePanel({ personaCount }: { personaCount: number }) {
  const entrance = useEntranceEnabled();
  const [product, setProduct] = useState("");
  const [pricesText, setPricesText] = useState("19, 29, 49, 79, 99");
  const [twins, setTwins] = useState(3);
  const [load, setLoad] = useState<CognitiveLoad>("low");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PriceSensitivityResult | null>(null);
  // live stream state — each streamed buyer's full intent-per-price curve
  const [curves, setCurves] = useState<number[][]>([]);
  const [runningDemand, setRunningDemand] = useState<number[] | undefined>();
  const [total, setTotal] = useState(0);

  const prices = pricesText
    .split(",")
    .map((p) => parseFloat(p.trim()))
    .filter((p) => !isNaN(p) && p > 0);

  async function run() {
    setBusy(true);
    setError(null);
    setResult(null);
    setCurves([]);
    setRunningDemand(undefined);
    try {
      await streamRun(
        "/studies/price/stream",
        { product, prices, twins_per_persona: twins, cognitive_load: load },
        (evt) => {
          if (evt.type === "start") setTotal(evt.total as number);
          else if (evt.type === "agent") {
            setCurves((c) => [...c, evt.intents as number[]]);
            setRunningDemand(evt.running_demand as number[]);
          } else if (evt.type === "done") {
            setResult(evt.result as unknown as PriceSensitivityResult);
          } else if (evt.type === "error") setError(String(evt.detail));
        }
      );
    } catch (e) {
      setError(e instanceof Error ? e.message : "Price study failed");
    } finally {
      setBusy(false);
    }
  }

  const data =
    result?.points.map((p) => ({
      price: `$${p.price}`,
      demand: p.demand,
      revenue: p.revenue_index,
      best: p.price === result.recommended_price,
    })) ?? [];

  return (
    <div className="card p-6">
      <div className="mb-1 flex items-baseline justify-between">
        <h2 className="font-display text-lg font-medium tracking-tight">Price Sensitivity Lab</h2>
        <span className="text-xs text-muted">find the revenue-maximizing price</span>
      </div>
      <p className="lede mb-4">
        Describe a product and list candidate prices. Each twin rates how likely
        they&apos;d buy at every price; we build the demand &amp; revenue curves and
        pick the best point.
      </p>

      <SimStage
        show={busy || !!result}
        busy={busy}
        runningLabel="buyers testing each price…"
        doneLabel="demand mapped"
        progress={
          busy
            ? `${curves.length}${total ? `/${total}` : ""} buyers`
            : result
              ? `${result.twin_count} buyers × ${prices.length} prices`
              : undefined
        }
      >
        <DemandSim
          prices={prices}
          curves={curves}
          runningDemand={runningDemand}
          cols={
            result
              ? result.points.map((p) => ({
                  price: p.price,
                  demand: p.demand,
                  best: p.price === result.recommended_price,
                }))
              : undefined
          }
        />
      </SimStage>

      <textarea
        value={product}
        onChange={(e) => setProduct(e.target.value)}
        rows={2}
        placeholder="e.g. A minimalist AI note-taking app with offline sync and unlimited notes, billed monthly."
        className="w-full resize-y rounded-xl border border-hairline bg-surface-2 p-3.5 text-sm outline-none placeholder:text-muted focus:border-accent/60 focus:ring-2 focus:ring-accent/20"
      />
      <label className="mt-3 block">
        <span className="mb-1 block text-xs text-muted">Candidate prices (comma-separated)</span>
        <input
          value={pricesText}
          onChange={(e) => setPricesText(e.target.value)}
          className="w-full rounded-xl border border-hairline bg-surface-2 px-3 py-2 text-sm outline-none tabular-nums focus:border-accent/60"
        />
      </label>

      <div className="mt-3 flex flex-wrap items-center gap-3">
        <span className="text-xs text-muted tabular-nums">{prices.length} prices</span>
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
        <LoadToggle value={load} onChange={setLoad} id="price-load" />
        <motion.button
          whileTap={{ scale: 0.97 }}
          disabled={busy || !product.trim() || prices.length < 2}
          onClick={run}
          className="ml-auto rounded-xl bg-accent px-5 py-2 font-display text-sm font-semibold text-white transition hover:brightness-110 disabled:opacity-40"
        >
          {busy ? "Modeling…" : "Run price study"}
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
            <p className="mb-4 text-sm">
              Revenue-maximizing price:{" "}
              <span className="font-display text-lg font-semibold text-accent">
                ${result.recommended_price}
              </span>
              <span className="text-muted"> · {result.twin_count} twins</span>
            </p>
            <div className="grid gap-4 sm:grid-cols-2">
              <div>
                <div className="mb-1 text-xs uppercase tracking-wider text-muted">
                  Demand vs price
                </div>
                <ResponsiveContainer width="100%" height={170}>
                  <LineChart data={data} margin={{ top: 6, right: 10, left: -20, bottom: 0 }}>
                    <CartesianGrid vertical={false} stroke="#2c2c2a" />
                    <XAxis dataKey="price" tickLine={false} axisLine={{ stroke: "#383835" }} tick={{ fill: "#898781", fontSize: 11 }} />
                    <YAxis domain={[0, 1]} tickLine={false} axisLine={false} tick={{ fill: "#898781", fontSize: 11 }} />
                    <Tooltip contentStyle={tooltipStyle} formatter={(v: number) => v.toFixed(2)} />
                    <Line type="monotone" dataKey="demand" stroke="#3987e5" strokeWidth={2} dot={{ r: 3, fill: "#3987e5" }} isAnimationActive={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
              <div>
                <div className="mb-1 text-xs uppercase tracking-wider text-muted">
                  Revenue index (best = 1.0)
                </div>
                <ResponsiveContainer width="100%" height={170}>
                  <BarChart data={data} margin={{ top: 6, right: 10, left: -20, bottom: 0 }}>
                    <CartesianGrid vertical={false} stroke="#2c2c2a" />
                    <XAxis dataKey="price" tickLine={false} axisLine={{ stroke: "#383835" }} tick={{ fill: "#898781", fontSize: 11 }} />
                    <YAxis domain={[0, 1]} tickLine={false} axisLine={false} tick={{ fill: "#898781", fontSize: 11 }} />
                    <Tooltip contentStyle={tooltipStyle} formatter={(v: number) => v.toFixed(2)} cursor={{ fill: "rgba(255,255,255,0.04)" }} />
                    <Bar dataKey="revenue" radius={[4, 4, 0, 0]} isAnimationActive={false}>
                      {data.map((d, i) => (
                        <Cell key={i} fill={d.best ? "#12c512" : "#3987e5"} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            {result.econometrics && (
              <div className="mt-4">
                <ElasticityPanel econ={result.econometrics} />
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

const tooltipStyle = {
  background: "#1a1a19",
  border: "1px solid rgba(255,255,255,0.1)",
  borderRadius: 12,
  fontSize: 12,
  color: "#c3c2b7",
};
