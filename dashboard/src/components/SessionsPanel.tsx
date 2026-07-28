"use client";

import { Fragment, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { SessionRow } from "@/lib/api";
import PersonaDetail from "./PersonaDetail";
import CognitionPipeline from "./CognitionPipeline";

export default function SessionsPanel({
  sessions,
  onChanged,
}: {
  sessions: SessionRow[];
  onChanged: () => void;
}) {
  const [expanded, setExpanded] = useState<string | null>(null);
  // which sessions are running (or have run) the live profiling pipeline
  const [piping, setPiping] = useState<Set<string>>(new Set());

  function startPipeline(id: string) {
    setPiping((s) => new Set(s).add(id));
    setExpanded(id);
  }

  return (
    <div className="card flex h-full flex-col p-6">
      <div className="mb-4 flex items-baseline justify-between">
        <h2 className="font-display text-lg font-medium tracking-tight">
          Session segments
        </h2>
        <span className="text-xs text-muted">{sessions.length} captured</span>
      </div>

      {sessions.length === 0 ? (
        <p className="text-sm text-muted">
          No telemetry yet — open the demo site, accept consent, and interact.
        </p>
      ) : (
        <div className="-mx-2 flex-1 overflow-auto">
          <table className="w-full text-sm">
            <thead className="sticky top-0 bg-surface text-left text-xs uppercase tracking-wider text-muted">
              <tr>
                <th className="px-2 pb-2 font-medium">Segment</th>
                <th className="px-2 pb-2 font-medium tabular-nums">Events</th>
                <th className="px-2 pb-2 font-medium tabular-nums">Rage</th>
                <th className="px-2 pb-2 font-medium">Persona</th>
                <th className="px-2 pb-2" />
              </tr>
            </thead>
            <tbody>
              {sessions.map((s) => {
                const isOpen = expanded === s.id;
                const showPipeline = piping.has(s.id);
                return (
                  <Fragment key={s.id}>
                    <motion.tr
                      layout
                      initial={false}
                      animate={{ opacity: 1 }}
                      onClick={() =>
                        (s.persona || s.cognition) &&
                        setExpanded((e) => (e === s.id ? null : s.id))
                      }
                      className={`border-t border-hairline/60 transition hover:bg-white/[0.03] ${
                        s.persona || s.cognition ? "cursor-pointer" : ""
                      }`}
                    >
                      <td className="px-2 py-2.5">
                        <code className="text-xs text-ink-2">{s.id}</code>
                        <div className="text-xs text-muted">{s.page_path}</div>
                      </td>
                      <td className="px-2 py-2.5 tabular-nums">{s.features.event_count}</td>
                      <td className="px-2 py-2.5 tabular-nums">
                        {s.features.rage_click_bursts > 0 ? (
                          <span className="text-accent-2">{s.features.rage_click_bursts}</span>
                        ) : (
                          <span className="text-muted">0</span>
                        )}
                      </td>
                      <td className="px-2 py-2.5">
                        {s.persona ? (
                          <span
                            className="inline-flex max-w-44 items-center gap-1 border border-hairline px-2 py-0.5 font-mono text-[10px] uppercase tracking-wider text-ink-2"
                            title={s.persona.label}
                          >
                            <motion.span animate={{ rotate: isOpen ? 90 : 0 }}>›</motion.span>
                            <span className="truncate">{s.persona.label}</span>
                          </span>
                        ) : showPipeline ? (
                          <span className="font-mono text-[10px] text-accent">
                            pipeline running…
                          </span>
                        ) : (
                          <span className="text-xs text-muted">—</span>
                        )}
                        {s.cognition && (
                          <span
                            className="ml-1.5 border border-hairline px-1.5 py-0.5 font-mono text-[9px] uppercase tracking-wider text-muted"
                            title={`models: ${s.cognition.models_run.join(", ")}`}
                          >
                            {s.cognition.models_run.length} models
                          </span>
                        )}
                      </td>
                      <td className="px-2 py-2.5 text-right">
                        {!s.persona && !showPipeline && (
                          <motion.button
                            whileTap={{ scale: 0.96 }}
                            onClick={(e) => {
                              e.stopPropagation();
                              startPipeline(s.id);
                            }}
                            className="rounded-lg border border-hairline px-3 py-1 text-xs text-ink-2 transition hover:border-accent/50 hover:text-ink"
                          >
                            Profile
                          </motion.button>
                        )}
                      </td>
                    </motion.tr>
                    <AnimatePresence>
                      {isOpen && (showPipeline || s.persona || s.cognition) && (
                        <motion.tr
                          initial={{ opacity: 0 }}
                          animate={{ opacity: 1 }}
                          exit={{ opacity: 0 }}
                        >
                          <td colSpan={5} className="bg-white/[0.02]">
                            {showPipeline ? (
                              /* live: stream the pipeline as it computes */
                              <CognitionPipeline sessionId={s.id} onDone={onChanged} />
                            ) : (
                              <>
                                {s.cognition && (
                                  <CognitionPipeline sessionId={s.id} stored={s.cognition} />
                                )}
                                {s.persona && <PersonaDetail persona={s.persona} />}
                              </>
                            )}
                          </td>
                        </motion.tr>
                      )}
                    </AnimatePresence>
                  </Fragment>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
