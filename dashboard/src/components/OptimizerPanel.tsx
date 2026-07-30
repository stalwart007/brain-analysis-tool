"use client";

/**
 * COPY OPTIMIZER — the first study that WRITES rather than scores.
 *
 * Every other panel measures copy you already wrote. This one runs an
 * evolutionary loop: score the population against the swarm, breed the next
 * generation from the elites, allocate twin budget by Thompson sampling so it
 * concentrates on the promising variants instead of spreading evenly.
 *
 * The visualization is the lineage. A ranked list would hide the only thing
 * that makes this an optimiser rather than a leaderboard — that variants have
 * PARENTS, and fitness climbs (or fails to) across generations. So the panel
 * draws the family tree, live, with each node's fitness as its height.
 */

import { useMemo, useState } from "react";
import { motion } from "framer-motion";
import { CognitiveLoad } from "@/lib/api";
import { useStreamRun } from "@/lib/useStreamRun";
import { LoadToggle } from "./LoadToggle";
import FindingsPanel, { ResearchQuestion, type FindingsBlock } from "./FindingsPanel";
import SimStage from "./sim/SimStage";
import { keyboardOnlyProps, useChartCursor } from "./charts/cursor";

interface PopVariant {
  id: string;
  generation: number;
  text: string;
  parents: string[];
  operation: string;
  rationale?: string;
  trials: number;
  acts: number;
  mean_intent: number;
  /** Empirical-Bayes shrunk mean — the RANKING quantity. A two-trial arm's raw
   *  mean is mostly noise, so ordering on it puts luck at the top: under a true
   *  null the raw ordering crowned a challenger 98.7% of the time. Absent on
   *  runs recorded before shrinkage existed, hence optional. */
  shrunk_intent?: number;
  act_rate: number;
}

interface GenStat {
  generation: number;
  best_intent?: number;
  mean_intent?: number;
  [k: string]: unknown;
}

interface OptimizerResult {
  findings?: FindingsBlock | null;
  twin_count: number;
  failed_evaluations: number;
  generations_planned: number;
  generations: GenStat[];
  population: PopVariant[];
  seed_variant_id: string;
  best_variant_id: string;
  best_text: string;
  convergence: string;
  verdict: string;
  /** The in-sample margin: the champion's lead measured on the very sample that
   *  crowned it. Kept for continuity and shown only as the thing the honest
   *  number is corrected AGAINST — never on its own. */
  intent_lift_vs_seed: number | null;
  intent_lift_vs_seed_pct: number | null;
  /** The reportable lift: selection re-run inside each split, evaluated on the
   *  held-out half. Under a true null the naive number reads +0.087 and is
   *  positive in 99.3% of runs; this one reads −0.0003. */
  selection_adjusted_lift?: number | null;
  selection_adjusted_lift_ci?: [number, number] | null;
  selection_adjusted_lift_pct?: number | null;
  selection_bias?: number | null;
  p_best_beats_seed?: number | null;
  lineage_edges: [string, string][];
  method: string;
  disclaimer: string;
}

/** Rank on the shrunk mean where the run provides one, falling back to the raw
 *  mean for runs recorded before shrinkage existed. */
const rankValue = (v: PopVariant) => v.shrunk_intent ?? v.mean_intent;

/* Convergence is the honest part: a search's winner is upward-biased by
   construction, so "improved" is only claimed when the intervals separate. */
const VERDICT_TONE: Record<string, string> = {
  improved: "border-good/40 bg-good/[0.07] text-good",
  not_supported: "border-accent-2/40 bg-accent-2/[0.07] text-accent-2",
  seed_wins: "border-critical/40 bg-critical/[0.07] text-critical",
  no_baseline: "border-hairline bg-white/[0.03] text-ink-2",
};

/** The family tree. Generations run left→right, fitness maps to height. */
function Lineage({ result }: { result: OptimizerResult }) {
  const { nodes, edges, W, H } = useMemo(() => {
    const gens = Math.max(1, result.generations_planned);
    const byGen = new Map<number, PopVariant[]>();
    for (const v of result.population) {
      const g = byGen.get(v.generation) ?? [];
      g.push(v);
      byGen.set(v.generation, g);
    }
    const W = 560;
    const H = 190;
    const padX = 34;
    const padY = 18;
    const pos = new Map<string, { x: number; y: number; v: PopVariant }>();
    for (const [g, list] of byGen) {
      const x = padX + (gens <= 1 ? 0.5 : g / (gens - 1 || 1)) * (W - padX * 2);
      list
        .slice()
        .sort((a, b) => b.mean_intent - a.mean_intent)
        .forEach((v) => {
          // fitness IS the vertical axis — a node's height is its mean intent,
          // so a rising tree is a working optimisation and a flat one is not
          const y = H - padY - v.mean_intent * (H - padY * 2);
          pos.set(v.id, { x, y, v });
        });
    }
    const edges = result.lineage_edges
      .map(([from, to]) => ({ a: pos.get(from), b: pos.get(to), from, to }))
      .filter((e) => e.a && e.b) as {
        a: { x: number; y: number };
        b: { x: number; y: number };
        from: string;
        to: string;
      }[];
    return { nodes: [...pos.values()], edges, W, H };
  }, [result]);

  // A graph, not an axis — so the index cursor is driven per node rather than
  // by pointer geometry, and the nodes carry their own hit areas. It still
  // buys hover, pin, keyboard traversal and Escape from the same primitive.
  const cursor = useChartCursor(
    "optimizer-lineage",
    nodes.length,
    "Variant lineage",
    (i) => {
      const v = nodes[i]?.v;
      if (!v) return `variant ${i + 1}`;
      return `generation ${v.generation}, ${v.operation}, mean intent ${v.mean_intent.toFixed(2)} over ${v.trials} trials`;
    }
  );
  const activeId = cursor.index === null ? null : nodes[cursor.index]?.v.id ?? null;
  // A node's parents, so hovering shows what it descended FROM — the only
  // question a lineage tree is really asked.
  const activeParents = new Set(
    cursor.index === null ? [] : nodes[cursor.index]?.v.parents ?? []
  );
  const active = cursor.index === null ? null : nodes[cursor.index]?.v ?? null;

  return (
    <>
    <svg {...keyboardOnlyProps(cursor)}
      onPointerLeave={() => cursor.setHovered(null)}
      viewBox={`0 0 ${W} ${H}`}
      className="w-full focus-visible:ring-1 focus-visible:ring-accent/60">
      {[0, 0.25, 0.5, 0.75, 1].map((v) => (
        <g key={v}>
          <line
            x1={26} x2={W - 8}
            y1={H - 18 - v * (H - 36)} y2={H - 18 - v * (H - 36)}
            stroke="rgba(223,217,217,0.07)"
          />
          <text x={22} y={H - 15 - v * (H - 36)} fontSize="7" fill="#6f6c6c" textAnchor="end" fontFamily="ui-monospace, monospace">
            {v}
          </text>
        </g>
      ))}
      {/* descent edges drawn as curves — a straight line reads as a chart
          series rather than an inheritance */}
      {edges.map((e, i) => {
        // an edge touching the cursor's node is its ancestry or its descent
        const lit = activeId !== null && (e.from === activeId || e.to === activeId);
        return (
          <path
            key={i}
            d={`M ${e.a.x},${e.a.y} C ${(e.a.x + e.b.x) / 2},${e.a.y} ${(e.a.x + e.b.x) / 2},${e.b.y} ${e.b.x},${e.b.y}`}
            fill="none"
            stroke={lit ? "rgb(var(--region-rgb) / 0.9)" : "rgb(var(--region-rgb) / 0.34)"}
            strokeWidth={lit ? 1.8 : 1}
            opacity={activeId && !lit ? 0.28 : 1}
            className="pointer-events-none"
          />
        );
      })}
      {nodes.map(({ x, y, v }, i) => {
        const isBest = v.id === result.best_variant_id;
        const isSeed = v.id === result.seed_variant_id;
        const lit = cursor.isMarked(i);
        const kin = activeParents.has(v.id);
        return (
          <g key={v.id}>
            {/* A 3.5px dot is not a hit target. This invisible disc is what
                makes the tree hoverable at all — Fitts's law, not decoration. */}
            <circle
              cx={x} cy={y} r={11} fill="transparent" className="cursor-pointer"
              onPointerEnter={() => cursor.setHovered(i)}
              onPointerDown={() => cursor.togglePin(i)}
            />
            <circle
              cx={x} cy={y}
              r={isBest ? 5.5 : lit ? 5 : 3.5}
              fill={isBest ? "#16d016" : isSeed ? "#dfd9d9" : "rgb(var(--region-rgb))"}
              stroke={cursor.isPinned(i) ? "#dfd9d9" : kin ? "rgba(255,255,255,0.7)" : "none"}
              strokeWidth={cursor.isPinned(i) ? 1.6 : 1}
              opacity={
                activeId && !lit && !kin ? 0.22 : v.trials > 0 ? 0.95 : 0.3
              }
              className="pointer-events-none"
            />
            {isBest && <circle cx={x} cy={y} r={9} fill="none" stroke="#16d016" strokeWidth="1" opacity="0.5" className="pointer-events-none" />}
            {isSeed && (
              <text x={x} y={y - 10} fontSize="7" fill="#dfd9d9" textAnchor="middle" fontFamily="ui-monospace, monospace" className="pointer-events-none">
                seed
              </text>
            )}
          </g>
        );
      })}
      <text x={W - 8} y={12} fontSize="7" fill="#6f6c6c" textAnchor="end" fontFamily="ui-monospace, monospace">
        generation →   ↑ mean intent
      </text>
    </svg>

    {/* The variant text itself. A dot on a tree cannot tell you what the copy
        SAYS, which is the one thing a copy optimiser is asked.

        THE HEIGHT IS RESERVED, NOT GROWN, and this was the worst offender in
        the codebase: `min-h-14` reserved 56px for a block whose focused state
        renders a whole copy variant — routinely 3-5 wrapped lines at 11px —
        plus an optional rationale. Every hover over the lineage tree therefore
        shoved the convergence verdict, the champion, the population list and
        everything below it down the page, and snapped it all back the moment
        the pointer left. The variant text is clamped to three lines for the
        same reason: an unbounded block of prose inside a fixed strip is a row
        whose height is a function of the data. Three rather than two because
        reading the copy is the point of this panel; the full text is always
        available in the population list below. */}
    <div className="mt-1.5 min-h-[6.25rem] rounded-lg border border-hairline/60 bg-black/20 px-2.5 py-1.5">
      {active ? (
        <>
          {/* tabular figures across the whole header: intent, trials and act
              rate all change as the cursor moves between nodes, and
              proportional digits shift every clause that follows them. */}
          <div className="flex flex-wrap items-baseline gap-x-2 font-mono text-[9px] tabular-nums text-muted">
            <span className="text-ink-2">gen {active.generation}</span>
            <span>· {active.operation}</span>
            <span>
              · intent{" "}
              <span className="text-accent">{active.mean_intent.toFixed(3)}</span>
            </span>
            {/* A mean over few trials is the classic optimiser trap, so the
                denominator travels with it rather than being a hover away. */}
            <span>· {active.trials} trial{active.trials === 1 ? "" : "s"}</span>
            <span>· act rate {(active.act_rate * 100).toFixed(0)}%</span>
            {active.id === result.best_variant_id && <span className="text-good">· best</span>}
            {cursor.pinned !== null && (
              <button onClick={cursor.clear} className="ml-auto text-accent transition hover:text-ink">
                pinned ✕
              </button>
            )}
          </div>
          <p className="mt-1 line-clamp-3 text-[11px] leading-snug text-ink-2">{active.text}</p>
          {/* Unconditional. An OPTIONAL row is a row whose presence moves the
              page: this line existed for mutated variants and not for the seed,
              so crossing between two nodes jumped everything below it by its
              own height. Saying the seed had no rationale is also the more
              honest reading than silently omitting the row. */}
          <p className="mt-0.5 line-clamp-1 text-[10px] leading-snug text-muted">
            {active.rationale || "no rationale recorded — this is the seed variant"}
          </p>
        </>
      ) : (
        <p className="font-mono text-[10px] text-muted">
          Hover a node to read its variant — arrow keys work too, Enter pins.
        </p>
      )}
    </div>
    </>
  );
}

export default function OptimizerPanel({ personaCount }: { personaCount: number }) {
  const streamRun = useStreamRun();
  const [brief, setBrief] = useState("");
  const [seed, setSeed] = useState("");
  const [pop, setPop] = useState(4);
  const [gens, setGens] = useState(2);
  const [twins, setTwins] = useState(2);
  const [load, setLoad] = useState<CognitiveLoad>("low");
  const [question, setQuestion] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<OptimizerResult | null>(null);
  const [gen, setGen] = useState(0);
  const [scored, setScored] = useState(0);
  const [total, setTotal] = useState<number | null>(null);
  const [mutations, setMutations] = useState<string[]>([]);

  async function run() {
    setBusy(true);
    setError(null);
    setResult(null);
    setGen(0);
    setScored(0);
    setMutations([]);
    try {
      await streamRun(
        "/studies/optimize/stream",
        {
          brief,
          seed_variant: seed,
          population_size: pop,
          generations: gens,
          twins_per_persona: twins,
          cognitive_load: load, research_question: question,
        },
        (evt) => {
          // The server sends `index`, not `generation` (optimizer.py emits
          // {"type":"generation","index":g,…}). Reading the wrong field meant
          // `Number(undefined ?? 0)` every time, so the counter in the running
          // label was pinned at 0 for the whole run. It is 0-based on the wire;
          // display it 1-based.
          if (evt.type === "start") setTotal(Number(evt.total ?? 0) || null);
          else if (evt.type === "generation") setGen(Number(evt.index ?? 0) + 1);
          else if (evt.type === "variant_scored") setScored((n) => n + 1);
          else if (evt.type === "mutation")
            setMutations((m) => [
              ...m.slice(-5),
              // Not every mutation frame is a successful breed. The generator
              // also emits {status:"skipped", detail:…} when a whole generation
              // failed to score, and those carry no `operation` — so falling
              // back to the literal "mutation" rendered a failure as if it were
              // a normal crossover.
              evt.status
                ? `${String(evt.status)}: ${String(evt.detail ?? "")}`.trim()
                : String(evt.operation ?? "mutation"),
            ]);
          else if (evt.type === "done") setResult(evt.result as unknown as OptimizerResult);
          else if (evt.type === "error") setError(String(evt.detail));
        }
      );
    } catch (e) {
      setError(e instanceof Error ? e.message : "Optimizer run failed");
    } finally {
      setBusy(false);
    }
  }

  // The evaluation budget is `personas × twins_per_persona` PER GENERATION,
  // split across the whole population within that generation (optimizer.py
  // decrements a single `remaining` across variants) — so the population size
  // is not a multiplier. Including it made the only cost readout in the product
  // 4× high at the default pop=4 and 8× at pop=8, which pushes users to shrink
  // the population out of unwarranted fear. The server emits this exact number
  // as `start.total`; `total` below replaces the estimate once it arrives.
  const cost = total ?? gens * personaCount * twins;

  return (
    <div className="card p-6" data-live={busy ? "1" : "0"}>
      <div className="mb-3 flex items-baseline justify-between">
        <h2>Copy optimizer</h2>
        <span className="text-xs text-muted">evolutionary search · writes, not scores</span>
      </div>
      <p className="lede mb-4">
        Score the population against the swarm, breed the next generation from the
        elites, and concentrate twin budget on the promising variants by Thompson
        sampling. A win is only claimed when the champion&apos;s credible interval
        clears the seed&apos;s.
      </p>

      <SimStage
        show={busy || !!result}
        busy={busy}
        runningLabel={`generation ${gen} · ${scored} variants scored`}
        doneLabel="lineage resolved"
        height={210}
      >
        {result ? (
          <div className="p-2">
            <Lineage result={result} />
          </div>
        ) : (
          <div className="flex h-full flex-col items-center justify-center gap-2">
            <span className="spike-train">
              {Array.from({ length: 12 }, (_, i) => <i key={i} />)}
            </span>
            <span className="hud-label">
              {mutations.length > 0 ? mutations[mutations.length - 1] : "seeding population"}
            </span>
          </div>
        )}
      </SimStage>

      <textarea
        value={brief}
        onChange={(e) => setBrief(e.target.value)}
        placeholder="The brief — what this copy has to achieve, for whom, and the constraints it must respect…"
        className="min-h-20 w-full resize-y rounded-xl border border-hairline bg-surface-2 p-3.5 text-sm outline-none placeholder:text-muted focus:border-accent/60"
      />
      <textarea
        value={seed}
        onChange={(e) => setSeed(e.target.value)}
        placeholder="Your current best copy — the seed, and the baseline every generation is measured against…"
        className="mt-2 min-h-20 w-full resize-y rounded-xl border border-hairline bg-surface-2 p-3.5 text-sm outline-none placeholder:text-muted focus:border-accent/60"
      />

      <ResearchQuestion
        value={question}
        onChange={setQuestion}
        disabled={busy}
        examples={["e.g. did the optimiser find a real win, or fit noise?"]}
      />

      <div className="mt-3 flex flex-wrap items-center gap-3 text-xs text-muted">
        <label className="flex items-center gap-2">
          Population
          <input type="number" min={2} max={8} value={pop}
            onChange={(e) => setPop(Math.min(8, Math.max(2, +e.target.value || 2)))}
            className="w-14 border border-hairline bg-surface-2 px-2 py-1.5 text-sm tabular-nums text-ink outline-none" />
        </label>
        <label className="flex items-center gap-2">
          Generations
          <input type="number" min={1} max={5} value={gens}
            onChange={(e) => setGens(Math.min(5, Math.max(1, +e.target.value || 1)))}
            className="w-14 border border-hairline bg-surface-2 px-2 py-1.5 text-sm tabular-nums text-ink outline-none" />
        </label>
        <label className="flex items-center gap-2">
          Twins
          <input type="number" min={1} max={10} value={twins}
            onChange={(e) => setTwins(Math.min(10, Math.max(1, +e.target.value || 1)))}
            className="w-14 border border-hairline bg-surface-2 px-2 py-1.5 text-sm tabular-nums text-ink outline-none" />
        </label>
        <LoadToggle value={load} onChange={setLoad} id="optimize-load" />
        {/* This is the most expensive study in the product; showing the bill
            before the click is the difference between a tool and a trap. */}
        <span className="soma" title="population × generations × personas × twins">
          ≈{cost} twin calls
        </span>
        <motion.button
          whileTap={{ scale: 0.97 }}
          disabled={busy || !brief.trim() || !seed.trim()}
          onClick={run}
          className="ml-auto bg-accent px-5 py-2 disabled:opacity-40"
        >
          {busy ? "Evolving…" : "Optimize"}
        </motion.button>
      </div>

      {error && <p className="mt-3 text-sm text-critical">{error}</p>}

      {result && (
        <div className="mt-5 space-y-4">
        <FindingsPanel findings={result.findings} />
          <div className={`border p-3 ${VERDICT_TONE[result.convergence] ?? VERDICT_TONE.no_baseline}`}>
            <div className="hud-label">Convergence · {result.convergence.replace(/_/g, " ")}</div>
            <p className="verdict mt-1">{result.verdict}</p>
            {/* The lift is shown ONLY when the run supports one. A search's
                winner is upward-biased by construction, so the in-sample margin
                is positive in ~99% of runs even when every variant is identical
                — quoting it under "not supported" is quoting the bias. */}
            {result.convergence !== "not_supported" &&
              result.selection_adjusted_lift != null && (
                <p className="mt-1 font-mono text-[10px] opacity-80">
                  {result.selection_adjusted_lift > 0 ? "+" : ""}
                  {result.selection_adjusted_lift.toFixed(3)} mean intent vs seed
                  {result.selection_adjusted_lift_ci && (
                    <> · 95% CI [{result.selection_adjusted_lift_ci[0].toFixed(3)},{" "}
                    {result.selection_adjusted_lift_ci[1].toFixed(3)}]</>
                  )}
                  {result.p_best_beats_seed != null && (
                    <> · P(beats seed) {result.p_best_beats_seed.toFixed(2)}</>
                  )}
                </p>
              )}
            {/* The correction itself, as a receipt: how much of the apparent
                win was the search fitting noise. Mirrors SequencePanel. */}
            {result.selection_bias != null && result.intent_lift_vs_seed != null && (
              <p className="mt-0.5 font-mono text-[9px] opacity-60">
                in-sample margin {result.intent_lift_vs_seed > 0 ? "+" : ""}
                {result.intent_lift_vs_seed.toFixed(3)} · winner&apos;s curse removed{" "}
                {result.selection_bias.toFixed(3)}
              </p>
            )}
          </div>

          <div>
            <div className="hud-label mb-1">Champion</div>
            <p className="border-l-2 border-good/60 bg-white/[0.02] p-3 text-sm text-ink">
              {result.best_text}
            </p>
          </div>

          <div>
            <div className="hud-label mb-2">Population</div>
            {/* Ordered by the SHRUNK mean, matching the server's ranking. Sorting
                on the raw mean here re-created the exact bug the server fixed:
                a two-trial arm that got lucky rode to the top of the list, and
                the top row is the one anyone reads. The bar is drawn at the
                shrunk value with a tick at the raw one, so how far a thin arm
                was pulled toward the population is visible rather than
                asserted. */}
            <div className="space-y-1.5">
              {result.population
                .slice()
                .sort((a, b) => rankValue(b) - rankValue(a))
                .map((v) => (
                  <div key={v.id} className="flex items-baseline gap-3 text-xs">
                    <span className="w-10 shrink-0 font-mono text-[10px] text-muted">
                      g{v.generation}
                    </span>
                    <span className="relative h-1 w-20 shrink-0 bg-white/[0.06]">
                      <span
                        className="block h-1"
                        style={{
                          width: `${rankValue(v) * 100}%`,
                          background: v.id === result.best_variant_id ? "#16d016" : "rgb(var(--region-rgb))",
                        }}
                      />
                      {v.shrunk_intent != null && v.trials > 0 && (
                        <span
                          aria-hidden
                          title={`raw mean ${v.mean_intent.toFixed(2)} over ${v.trials} trials`}
                          className="absolute top-[-2px] h-2 w-px bg-white/50"
                          style={{ left: `${Math.min(100, v.mean_intent * 100)}%` }}
                        />
                      )}
                    </span>
                    <span className="w-24 shrink-0 font-mono text-[10px] tabular-nums text-muted">
                      {v.trials > 0 ? `${v.mean_intent.toFixed(2)} · n=${v.trials}` : "unevaluated"}
                    </span>
                    <span className="truncate text-ink-2">{v.text}</span>
                  </div>
                ))}
            </div>
          </div>

          <p className="font-mono text-[10px] text-muted">{result.method}</p>
          <p className="text-[9px] leading-tight text-muted">{result.disclaimer}</p>
        </div>
      )}
    </div>
  );
}
