"use client";

import { motion } from "framer-motion";
import { CognitiveLoad } from "@/lib/api";

const LOADS: CognitiveLoad[] = ["low", "medium", "high"];

export function LoadToggle({
  value,
  onChange,
  id,
}: {
  value: CognitiveLoad;
  onChange: (v: CognitiveLoad) => void;
  id: string;
}) {
  return (
    <>
      <div className="flex overflow-hidden rounded-lg border border-hairline text-xs">
        {LOADS.map((l) => (
          <button
            key={l}
            onClick={() => onChange(l)}
            className={`relative px-3 py-1.5 capitalize transition ${
              value === l ? "text-ink" : "text-muted hover:text-ink-2"
            }`}
          >
            {value === l && (
              <motion.span
                layoutId={`${id}-pill`}
                className="absolute inset-0 bg-white/[0.07]"
                transition={{ type: "spring", stiffness: 400, damping: 34 }}
              />
            )}
            <span className="relative">{l}</span>
          </button>
        ))}
      </div>
      <span className="text-xs text-muted">cognitive load</span>
    </>
  );
}
