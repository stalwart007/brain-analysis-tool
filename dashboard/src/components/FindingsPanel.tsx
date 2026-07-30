"use client";

/**
 * THE ANSWER SURFACE — what the study means for the question that was asked.
 *
 * Every panel in this product used to end at a wall of numbers plus one
 * template sentence. This renders the layer above that: a direct answer,
 * findings ranked by decision value, and — the part that matters — every claim
 * wired to the measurement that licenses it.
 *
 * THE CITATION CHIPS ARE THE DESIGN. A findings list written by a model is
 * indistinguishable, visually, from a findings list someone made up; fluent
 * prose is exactly the medium in which an unsupported claim looks like a
 * supported one. So each finding carries its evidence ids as chips, clicking
 * one opens the actual measurement with its interval and its q-value, and a
 * finding standing only on evidence that did not clear FDR control is marked as
 * such rather than being quietly rendered in the same weight as one that did.
 * The interface makes the difference legible instead of asking for trust.
 *
 * Instrument surfaces: zero radius, mono, uppercase — this is the readout, not
 * the tissue. Tinting comes from --region-rgb like everything else, so the
 * findings block belongs to whichever lobe you are standing in.
 */

import { useState } from "react";
import Interval from "./Interval";
import { AnimatePresence, motion } from "framer-motion";
import type { EvidenceRow, FindingRow, FindingsBlock } from "@/lib/api";

export type { EvidenceRow, FindingRow, FindingsBlock };

/** Direction drives the left rule and the glyph — colour is data here, not decoration. */
const DIRECTION: Record<
  FindingRow["direction"],
  { rule: string; text: string; glyph: string; label: string }
> = {
  risk: { rule: "#f0605f", text: "text-critical", glyph: "▲", label: "RISK" },
  strength: { rule: "#16d016", text: "text-good", glyph: "●", label: "STRENGTH" },
  opportunity: { rule: "#d95926", text: "text-accent-2", glyph: "◆", label: "OPPORTUNITY" },
  neutral: { rule: "#6f6c6c", text: "text-muted", glyph: "○", label: "OBSERVATION" },
};

const CONFIDENCE_BARS = { high: 3, moderate: 2, low: 1 } as const;

export default function FindingsPanel({ findings }: { findings?: FindingsBlock | null }) {
  const [openEvidence, setOpenEvidence] = useState<string | null>(null);
  const [showAllEvidence, setShowAllEvidence] = useState(false);

  if (!findings) return null;
  const byId = new Map(findings.evidence.map((e) => [e.id, e]));
  const report = findings.report;

  return (
    <section className="border border-hairline bg-black/30">
      {/* ── masthead ─────────────────────────────────────────────────── */}
      <header className="flex flex-wrap items-baseline gap-x-3 gap-y-1 border-b border-hairline px-4 py-2.5">
        <span className="hud-label" style={{ color: "rgb(var(--region-rgb))" }}>
          FINDINGS
        </span>
        {findings.question ? (
          <span className="flex-1 text-[12px] leading-snug text-ink-2">
            <span className="text-muted">asked:</span> {findings.question}
          </span>
        ) : (
          <span className="flex-1 font-mono text-[10px] text-muted">
            no question stated — answering the study&apos;s default
          </span>
        )}
        {/* The multiplicity ledger, always visible. How many tests were run is
            not a footnote: it is the number that decides what any single
            "significant" result is worth. */}
        <span className="font-mono text-[10px] tabular-nums text-muted">
          {findings.multiplicity.n_rejected}/{findings.multiplicity.n_tests} survived FDR
          {findings.multiplicity.cutoff !== null && (
            <> · p≤{findings.multiplicity.cutoff.toPrecision(2)}</>
          )}
        </span>
      </header>

      {findings.error && !report && (
        <div className="px-4 py-3">
          <p className="text-[12px] leading-relaxed text-critical">{findings.error}</p>
          <p className="mt-1 font-mono text-[10px] leading-relaxed text-muted">
            The measurements below were still taken — synthesis is a layer on
            top of them, not a prerequisite for them.
          </p>
        </div>
      )}

      {report && (
        <>
          {/* ── the answer ───────────────────────────────────────────── */}
          <div
            className="border-b border-hairline px-4 py-3"
            style={{ background: "rgb(var(--region-rgb) / 0.05)" }}
          >
            <p className="verdict text-ink">{report.answer}</p>
          </div>

          {/* ── findings ─────────────────────────────────────────────── */}
          {report.findings.length === 0 ? (
            <p className="px-4 py-3 text-[12px] leading-relaxed text-muted">
              Nothing in this study clears its evidential bar. That is a result,
              not an empty state — the measurements below are all consistent
              with chance once corrected for how many were taken.
            </p>
          ) : (
            <ol>
              {report.findings.map((f, i) => {
                const dir = DIRECTION[f.direction] ?? DIRECTION.neutral;
                const cited = f.evidence_ids.map((id) => byId.get(id)).filter(Boolean) as EvidenceRow[];
                return (
                  <motion.li
                    key={i}
                    initial={{ opacity: 0, y: 4 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: Math.min(i * 0.04, 0.2), duration: 0.25 }}
                    className="border-b border-hairline/60 px-4 py-3 last:border-b-0"
                    style={{ boxShadow: `inset 2px 0 0 ${dir.rule}` }}
                  >
                    <div className="flex items-baseline gap-2">
                      <span className={`font-mono text-[10px] ${dir.text}`} aria-hidden>
                        {dir.glyph}
                      </span>
                      <h4 className="flex-1 font-display text-[14px] font-medium leading-snug tracking-tight text-ink">
                        {f.headline}
                      </h4>
                      {f.answers_question && (
                        <span
                          className="shrink-0 border px-1.5 py-px font-mono text-[8px] uppercase tracking-widest"
                          style={{
                            borderColor: "rgb(var(--region-rgb) / 0.5)",
                            color: "rgb(var(--region-rgb))",
                          }}
                        >
                          answers it
                        </span>
                      )}
                      {/* Confidence as three ticks: readable at a glance, and it
                          does not pretend to a precision the label does not have. */}
                      <span className="flex shrink-0 items-center gap-px" title={`${f.confidence} confidence`}>
                        {[0, 1, 2].map((b) => (
                          <span
                            key={b}
                            className="block h-2.5 w-[3px]"
                            style={{
                              background:
                                b < CONFIDENCE_BARS[f.confidence] ? dir.rule : "rgba(223,217,217,0.14)",
                            }}
                          />
                        ))}
                      </span>
                    </div>

                    <p className="mt-1 text-[12px] leading-relaxed text-ink-2">{f.detail}</p>

                    {f.downgraded && (
                      <p className="mt-1.5 border-l-2 border-critical/60 pl-2 font-mono text-[9px] leading-relaxed text-critical">
                        downgraded — {f.downgraded}
                      </p>
                    )}

                    {f.recommended_action && (
                      <p className="mt-1.5 text-[11px] leading-relaxed text-ink-2">
                        <span className="hud-label mr-1.5">DO</span>
                        {f.recommended_action}
                      </p>
                    )}

                    {/* citation chips */}
                    <div className="mt-2 flex flex-wrap items-center gap-1">
                      {cited.map((e) => (
                        <button
                          key={e.id}
                          onClick={() => setOpenEvidence(openEvidence === e.id ? null : e.id)}
                          className={`border px-1.5 py-0.5 font-mono text-[9px] transition ${
                            e.supported
                              ? "border-hairline text-ink-2 hover:border-accent/60 hover:text-ink"
                              : "border-dashed border-hairline/60 text-muted hover:text-ink-2"
                          }`}
                          aria-expanded={openEvidence === e.id}
                        >
                          {e.supported ? "●" : "○"} {e.label} = {e.display}
                        </button>
                      ))}
                    </div>

                    <AnimatePresence>
                      {openEvidence && cited.some((e) => e.id === openEvidence) && (
                        <motion.div
                          initial={{ opacity: 0, height: 0 }}
                          animate={{ opacity: 1, height: "auto" }}
                          exit={{ opacity: 0, height: 0 }}
                          transition={{ duration: 0.18 }}
                          className="overflow-hidden"
                        >
                          <EvidenceDetail row={byId.get(openEvidence)!} />
                        </motion.div>
                      )}
                    </AnimatePresence>
                  </motion.li>
                );
              })}
            </ol>
          )}

          {/* ── the honest footer ────────────────────────────────────── */}
          <div className="grid gap-x-6 gap-y-2 border-t border-hairline px-4 py-3 sm:grid-cols-2">
            <div>
              <span className="hud-label">WHAT WOULD CHANGE THIS</span>
              <p className="mt-0.5 text-[11px] leading-relaxed text-ink-2">
                {report.what_would_change_the_answer}
              </p>
            </div>
            <div>
              <span className="hud-label">WHAT THIS CANNOT SHOW</span>
              <p className="mt-0.5 text-[11px] leading-relaxed text-ink-2">{report.limits}</p>
            </div>
          </div>
        </>
      )}

      {/* ── the full evidence ledger ─────────────────────────────────── */}
      <div className="border-t border-hairline">
        <button
          onClick={() => setShowAllEvidence((v) => !v)}
          className="flex w-full items-center gap-2 px-4 py-2 text-left transition hover:bg-white/[0.02]"
        >
          <span className="hud-label">
            {showAllEvidence ? "▾" : "▸"} EVERY MEASUREMENT ({findings.evidence.length})
          </span>
          <span className="font-mono text-[9px] text-muted">
            {findings.n_supported} supported · {findings.evidence.length - findings.n_supported} not
          </span>
        </button>
        {showAllEvidence && (
          <div className="border-t border-hairline/60">
            {findings.evidence.map((e) => (
              <EvidenceDetail key={e.id} row={e} dense />
            ))}
            <p className="px-4 py-2 font-mono text-[9px] leading-relaxed text-muted">
              {findings.method}. {findings.multiplicity.method}.
              {report && report.dropped_uncited > 0 && (
                <>
                  {" "}
                  {report.dropped_uncited} draft finding
                  {report.dropped_uncited === 1 ? " was" : "s were"} discarded for
                  citing no measurement.
                </>
              )}
            </p>
          </div>
        )}
      </div>
    </section>
  );
}

/** One measurement, with everything needed to argue with it. */
function EvidenceDetail({ row, dense }: { row: EvidenceRow; dense?: boolean }) {
  return (
    <div
      className={`border-l-2 px-3 py-2 ${dense ? "border-b border-b-hairline/40" : "mt-2"}`}
      style={{ borderLeftColor: row.supported ? "rgb(var(--region-rgb))" : "rgba(223,217,217,0.18)" }}
    >
      <div className="flex flex-wrap items-baseline gap-x-2 font-mono text-[10px] tabular-nums">
        <span className="text-ink">{row.label}</span>
        <span className="text-accent">{row.display}</span>
        {/* The absent case is a FINDING, not a gap. It used to render nothing
            at all, so a number with no interval was indistinguishable from one
            whose interval had simply not been plumbed through. */}
        <Interval ci={row.ci as [number, number] | null | undefined} />
        {row.p_value !== null && <span className="text-muted">p={row.p_value.toPrecision(2)}</span>}
        {row.q_value !== null && <span className="text-muted">q={row.q_value.toPrecision(2)}</span>}
        {row.n !== null && <span className="text-muted">n={row.n}</span>}
        <span className={row.supported ? "text-good" : "text-muted"}>
          {row.supported ? "● supported" : "○ not supported"}
        </span>
      </div>
      <p className="mt-0.5 text-[10px] leading-relaxed text-muted">
        {row.interpretation}
        {row.support_reason && <span className="text-ink-2"> — {row.support_reason}.</span>}
      </p>
    </div>
  );
}

/**
 * The question box. Deliberately plain: it is the only field in the product
 * that changes what the report is ABOUT rather than what the run measures.
 */
export function ResearchQuestion({
  value,
  onChange,
  disabled,
  examples,
}: {
  value: string;
  onChange: (v: string) => void;
  disabled?: boolean;
  examples?: string[];
}) {
  return (
    <div className="mb-3">
      <label className="hud-label mb-1 block" htmlFor="research-question">
        WHAT DO YOU WANT TO KNOW?
      </label>
      <input
        id="research-question"
        type="text"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        disabled={disabled}
        placeholder={examples?.[0] ?? "e.g. does the pricing section create anxiety for first-time buyers?"}
        className="w-full border border-hairline bg-surface-2 px-3 py-2 text-sm text-ink outline-none transition placeholder:text-muted focus:border-accent/60 disabled:opacity-50"
      />
      <p className="mt-1 font-mono text-[9px] leading-relaxed text-muted">
        Optional. Ranks the findings and aims the answer.{" "}
        <b className="text-ink-2">The twins never see it</b> — telling a
        synthetic audience what you are hoping to find is how you get an
        audience that finds it, so the measurements are identical either way.
      </p>
      {examples && examples.length > 1 && (
        <div className="mt-1 flex flex-wrap gap-1">
          {examples.slice(1).map((ex) => (
            <button
              key={ex}
              type="button"
              disabled={disabled}
              onClick={() => onChange(ex)}
              className="border border-dashed border-hairline px-1.5 py-0.5 text-left font-mono text-[9px] text-muted transition hover:border-accent/50 hover:text-ink-2 disabled:opacity-40"
            >
              {ex}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
