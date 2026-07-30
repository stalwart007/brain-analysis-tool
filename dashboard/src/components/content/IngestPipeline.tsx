"use client";

/**
 * WHAT INGEST IS ACTUALLY DOING, while it does it.
 *
 * Pasting a link used to produce a spinner and then, some seconds later, either
 * an asset or a sentence about why not. Both ends of that were dishonest in the
 * same direction: the spinner claimed one indivisible operation, and the
 * failure could not say which of six things had failed. Ingest is a CHAIN —
 * resolve the link, ask YouTube what the video is, pull the transcript, pull
 * the filmstrip sheets, crop tiles out of them — and each link can succeed,
 * fail, or be skipped independently, with a count attached.
 *
 * So the chain is what is drawn. Every node is a real step with a real tally,
 * and a step that was skipped says SKIPPED rather than quietly not appearing —
 * "this video publishes no captions" is a finding about the video, and a node
 * that silently vanishes destroys it.
 *
 * MOTION RULES, inherited from the stylesheet and not negotiable here:
 *   · One clock. The travelling signal derives from `--pulse`, published by
 *     VitalsClock. A local `setInterval` here would beat against every other
 *     pulse in the interface and read as noise.
 *   · Motion is structural. The signal travels along a connector only while
 *     that connector's downstream node is working, so movement means work is
 *     happening rather than decorating the fact that a panel exists.
 */

import { motion } from "framer-motion";
import { useEntranceEnabled } from "@/components/Reveal";

export type StageState = "pending" | "active" | "done" | "skipped" | "failed";

export interface Stage {
  id: string;
  /** Imperative and short — this is a machine reporting, not prose. */
  label: string;
  state: StageState;
  /** Real tally: sheets pulled, cues parsed, tiles cropped. Never a percentage. */
  detail?: string;
}

const TONE: Record<StageState, { dot: string; text: string; edge: string }> = {
  pending: { dot: "bg-muted/40", text: "text-muted", edge: "border-hairline" },
  active: { dot: "bg-accent", text: "text-ink", edge: "border-accent/60" },
  done: { dot: "bg-good", text: "text-ink-2", edge: "border-good/40" },
  // Amber, not red. A skipped stage is a property of the content — this video
  // has no captions — and colouring it as a failure tells someone to go fix
  // something that is not broken.
  skipped: { dot: "bg-[#f2ad1f]", text: "text-muted", edge: "border-[#f2ad1f]/40" },
  failed: { dot: "bg-critical", text: "text-critical", edge: "border-critical/50" },
};

export default function IngestPipeline({ stages }: { stages: Stage[] }) {
  const entrance = useEntranceEnabled();
  if (!stages.length) return null;

  return (
    <div className="rounded-xl border border-hairline bg-black/40 p-3">
      <div className="mb-2.5 flex items-baseline justify-between">
        <span className="hud-label">INGEST PIPELINE</span>
        <span className="font-mono text-[9px] text-muted">
          {stages.filter((s) => s.state === "done").length}/{stages.length} stages
        </span>
      </div>

      <ol className="flex flex-wrap items-stretch gap-y-3">
        {stages.map((stage, i) => {
          const tone = TONE[stage.state];
          const last = i === stages.length - 1;
          return (
            <li key={stage.id} className="flex min-w-0 flex-1 items-center gap-2">
              <motion.div
                initial={entrance ? { opacity: 0, y: 4 } : false}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.04 }}
                className={`min-w-0 flex-1 rounded-lg border px-2.5 py-1.5 transition-colors ${tone.edge}`}
                style={
                  stage.state === "active"
                    ? {
                        // Breathing edge, from the shared clock. `--pulse` is
                        // 0‥1 with a sharp systolic upstroke, so the node reads
                        // as beating rather than fading in and out.
                        boxShadow:
                          "0 0 calc(6px + var(--pulse) * 10px) rgb(var(--region-rgb) / calc(0.18 + var(--pulse) * 0.22))",
                      }
                    : undefined
                }
              >
                <div className="flex items-center gap-1.5">
                  <span
                    className={`inline-block h-1.5 w-1.5 shrink-0 rounded-full ${tone.dot} ${
                      stage.state === "active" ? "pulse-dot" : ""
                    }`}
                  />
                  <span
                    className={`truncate font-mono text-[10px] uppercase tracking-wider ${tone.text}`}
                  >
                    {stage.label}
                  </span>
                </div>
                {/* Height is reserved whether or not there is a detail, so a
                    tally arriving mid-run does not shove the whole chain
                    downward at the moment the reader is watching it. */}
                <p className="mt-0.5 h-3 truncate font-mono text-[9px] leading-3 text-muted">
                  {stage.state === "skipped" && !stage.detail ? "skipped" : stage.detail ?? ""}
                </p>
              </motion.div>

              {!last && (
                <div className="relative h-px w-4 shrink-0 bg-hairline">
                  {/* The travelling signal, and the only moving thing here. It
                      exists ONLY while the downstream node is working, so
                      motion on this connector means the next stage is running
                      right now. */}
                  {stages[i + 1]?.state === "active" && (
                    <span
                      className="absolute top-1/2 h-1 w-1 -translate-y-1/2 rounded-full bg-accent"
                      style={{
                        left: "calc(var(--pulse) * 100%)",
                        opacity: "calc(0.35 + var(--pulse) * 0.65)",
                      }}
                    />
                  )}
                </div>
              )}
            </li>
          );
        })}
      </ol>
    </div>
  );
}
