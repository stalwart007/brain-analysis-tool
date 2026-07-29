"use client";

/**
 * THE AUDIENCE PANEL — who the swarm is about to simulate, and a way to say
 * "no, also include…".
 *
 * Two problems it solves at once.
 *
 * Cold start: with no telemetry the backend infers a panel from the stimulus,
 * which unblocks a new account but happened invisibly — the run just worked and
 * the user never saw who answered. An audience you cannot see is one you cannot
 * disagree with.
 *
 * Curation: a researcher usually knows something the copy does not say. Typing
 * "add price-conscious students on a trial" is faster than any form, and the
 * backend returns the FULL resulting panel so what is shown is exactly what
 * will run.
 *
 * PROVENANCE IS THE LOAD-BEARING PART. Observed segments come from measured
 * behaviour; inferred ones are hypotheses about who the material would reach,
 * including the ones the researcher asked for by name — asserting a segment
 * exists is a stated assumption, not evidence. The panel says which is which on
 * every row, because a run against an imagined audience and one against a
 * measured audience are different claims.
 */

import { useState } from "react";
import { motion } from "framer-motion";

export interface Persona {
  persona_id: string;
  label: string;
  provenance?: "observed" | "inferred";
  signal: {
    likely_mindset: string;
    deliberation: string;
    frustration_signal: string;
    exploration_style: string;
    price_sensitivity_signal: string;
    confidence: number;
  };
}

export default function AudiencePanel({
  stimulus,
  personas,
  onChange,
  observedCount,
  disabled,
}: {
  /** what the audience would be reacting to; inference needs it */
  stimulus: string;
  /** the curated panel, or [] to let the backend decide at run time */
  personas: Persona[];
  onChange: (p: Persona[]) => void;
  /** profiled sessions available — when > 0 the backend prefers those */
  observedCount: number;
  disabled?: boolean;
}) {
  const [instruction, setInstruction] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [open, setOpen] = useState(false);

  const ready = stimulus.trim().length >= 20;

  async function compose(text: string) {
    setBusy(true);
    setError(null);
    try {
      const res = await fetch("/api/cs/audience/compose", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          stimulus,
          instruction: text,
          existing: personas,
        }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => null);
        throw new Error(body?.detail ?? `Compose failed (${res.status})`);
      }
      const data = await res.json();
      onChange(data.personas as Persona[]);
      setInstruction("");
      setOpen(true);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not compose the audience");
    } finally {
      setBusy(false);
    }
  }

  const inferred = personas.filter((p) => p.provenance === "inferred").length;

  return (
    <div className="mb-4 rounded-xl border border-hairline bg-surface-2/40 p-3">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <button
          onClick={() => setOpen((o) => !o)}
          aria-expanded={open}
          className="hud-label text-ink-2 transition hover:text-ink"
        >
          AUDIENCE {open ? "▾" : "▸"}
        </button>
        <span className="font-mono text-[10px] text-muted">
          {personas.length > 0 ? (
            <>
              {personas.length} segment{personas.length === 1 ? "" : "s"}
              {inferred > 0 && (
                <span className="text-accent-2"> · {inferred} inferred</span>
              )}
            </>
          ) : observedCount > 0 ? (
            `${observedCount} profiled session${observedCount === 1 ? "" : "s"} — observed audience`
          ) : (
            "none yet — will be inferred from your stimulus"
          )}
        </span>
      </div>

      {open && personas.length > 0 && (
        <ul className="mt-3 space-y-2">
          {personas.map((p) => (
            <li
              key={p.persona_id}
              className="group flex items-start gap-2 rounded-lg border border-hairline/60 bg-black/20 px-2.5 py-2"
            >
              <span
                title={
                  p.provenance === "inferred"
                    ? "Inferred from the stimulus — a hypothesis about who this reaches, not a measurement of anyone."
                    : "Derived from observed behavioural telemetry."
                }
                className={`mt-0.5 inline-block h-1.5 w-1.5 shrink-0 rounded-full ${
                  p.provenance === "inferred" ? "bg-accent-2" : "bg-good"
                }`}
              />
              <div className="min-w-0 flex-1">
                <div className="flex items-baseline gap-2">
                  <span className="truncate text-xs text-ink">{p.label}</span>
                  <span className="shrink-0 font-mono text-[9px] text-muted">
                    conf {p.signal.confidence.toFixed(2)}
                  </span>
                </div>
                <p className="mt-0.5 text-[11px] leading-snug text-ink-2">
                  {p.signal.likely_mindset}
                </p>
                <p className="mt-0.5 font-mono text-[9px] text-muted">
                  {p.signal.deliberation} deliberation · {p.signal.exploration_style} ·
                  price {p.signal.price_sensitivity_signal}
                </p>
              </div>
              <button
                onClick={() => onChange(personas.filter((x) => x.persona_id !== p.persona_id))}
                disabled={disabled || busy}
                aria-label={`Remove ${p.label}`}
                className="shrink-0 rounded px-1 text-xs text-muted opacity-0 transition group-hover:opacity-100 hover:text-critical focus:opacity-100"
              >
                ×
              </button>
            </li>
          ))}
        </ul>
      )}

      <div className="mt-3 flex flex-wrap items-center gap-2">
        <input
          value={instruction}
          onChange={(e) => setInstruction(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && instruction.trim() && ready && !busy) {
              compose(instruction);
            }
          }}
          disabled={disabled || busy || !ready}
          placeholder={
            ready
              ? "add price-conscious students on a trial…"
              : "paste your stimulus first"
          }
          className="min-w-48 flex-1 rounded-lg border border-hairline bg-surface-2 px-2.5 py-1.5 text-xs outline-none placeholder:text-muted focus:border-accent/60 disabled:opacity-40"
        />
        <motion.button
          whileTap={{ scale: 0.97 }}
          onClick={() => compose(instruction)}
          disabled={disabled || busy || !ready || !instruction.trim()}
          className="rounded-lg border border-hairline px-3 py-1.5 text-xs text-ink-2 transition hover:border-accent/50 hover:text-ink disabled:opacity-40"
        >
          {busy ? "composing…" : "apply"}
        </motion.button>
        <button
          onClick={() => compose("")}
          disabled={disabled || busy || !ready}
          title="Propose an audience from the stimulus alone"
          className="rounded-lg border border-dashed border-hairline px-3 py-1.5 text-xs text-muted transition hover:border-accent/50 hover:text-ink-2 disabled:opacity-40"
        >
          {personas.length ? "re-infer" : "infer audience"}
        </button>
        {personas.length > 0 && (
          <button
            onClick={() => onChange([])}
            disabled={disabled || busy}
            title="Clear the curated panel and let the run decide"
            className="rounded-lg px-2 py-1.5 text-xs text-muted transition hover:text-ink-2"
          >
            reset
          </button>
        )}
      </div>

      {error && <p className="mt-2 text-[11px] text-critical">{error}</p>}

      {inferred > 0 && (
        <p className="mt-2 text-[10px] leading-relaxed text-muted">
          Inferred segments are hypotheses about who this material would reach —
          including any you asked for by name. They are not measurements of
          anyone. Profile real sessions to replace them with observed segments.
        </p>
      )}
    </div>
  );
}
