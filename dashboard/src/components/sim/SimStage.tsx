"use client";

import { ReactNode } from "react";
import { AnimatePresence, motion } from "framer-motion";

/** The framed viewport a per-simulation visualization renders inside, with a
 *  live status readout in the corner. Appears while running and stays after. */
export default function SimStage({
  show,
  busy,
  runningLabel,
  doneLabel,
  progress,
  height = 240,
  children,
}: {
  show: boolean;
  busy: boolean;
  runningLabel: string;
  doneLabel: string;
  progress?: string; // live counter, e.g. "14/40 agents" — from real stream events
  height?: number;
  children: ReactNode;
}) {
  return (
    <AnimatePresence>
      {show && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height }}
          exit={{ opacity: 0, height: 0 }}
          transition={{ type: "spring", stiffness: 200, damping: 28 }}
          className="mb-4 overflow-hidden rounded-xl border border-hairline bg-black/40"
        >
          <div className="relative" style={{ height }}>
            {children}
            <div className="pointer-events-none absolute left-3 top-3 flex items-center gap-2 text-xs">
              <span
                className={`neon-dot inline-block h-2 w-2 rounded-full ${
                  busy ? "pulse-dot bg-accent" : "bg-good"
                }`}
              />
              <span className="hud-label" style={{ letterSpacing: "0.12em" }}>
                {busy ? runningLabel : doneLabel}
              </span>
              {progress && (
                <span className="font-mono text-[10px] text-accent/90">{progress}</span>
              )}
            </div>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
