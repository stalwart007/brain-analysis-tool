"use client";

/**
 * Chrome primitives for the technical-brutalist theme: hard-edged readout
 * blocks, giant page mastheads, and section rules. Monochrome by rule —
 * saturated color belongs to data, never to chrome.
 */

import { ReactNode } from "react";

/** Big mono value over a tiny uppercase label, in a 1px box. The reference
 *  pattern for every headline metric in the app. */
export function Readout({
  value,
  label,
  sub,
  invert,
}: {
  value: ReactNode;
  label: string;
  sub?: string;
  invert?: boolean;
}) {
  return (
    <div
      className={`relative border p-3 ${
        invert ? "border-bone bg-bone text-void" : "border-hairline bg-surface"
      }`}
    >
      <div
        className={`font-mono text-[26px] leading-none tabular-nums ${
          invert ? "text-void" : "text-bone"
        }`}
      >
        {value}
      </div>
      <div
        className={`mt-2 font-mono text-[9px] uppercase tracking-[0.16em] ${
          invert ? "text-void/60" : "text-muted"
        }`}
      >
        {label}
      </div>
      {sub && (
        <div
          className={`mt-0.5 font-mono text-[9px] tabular-nums ${
            invert ? "text-void/50" : "text-muted/80"
          }`}
        >
          {sub}
        </div>
      )}
    </div>
  );
}

/**
 * ELECTRODE — a reading taken *off* the tissue rather than a tile stacked on
 * top of it.
 *
 * The difference matters: `Readout` is a bordered box, and four of them in a
 * row is a wall that hides whatever is behind. An electrode is a leader line,
 * a numeral and a label with no fill at all, so the cortex reads straight
 * through the gaps between them. Used on the surface station, where the whole
 * viewport is supposed to be brain.
 */
export function Electrode({
  value,
  label,
  live,
}: {
  value: ReactNode;
  label: string;
  /** draws the contact as energised — used when the reading is fresh */
  live?: boolean;
}) {
  return (
    <div className="relative pl-4">
      {/* the contact: a lead running down into tissue, tipped with a bead */}
      <span
        aria-hidden
        className="absolute left-0 top-1 h-[calc(100%-0.5rem)] w-px"
        style={{
          background:
            "linear-gradient(180deg, rgb(var(--region-rgb) / 0.85), rgb(var(--region-rgb) / 0.08))",
        }}
      />
      <span
        aria-hidden
        className="absolute -left-[2.5px] top-0.5 h-1.5 w-1.5 rounded-full"
        style={{
          background: "rgb(var(--region-rgb))",
          boxShadow: live
            ? "0 0 calc(5px + var(--pulse) * 9px) rgb(var(--region-rgb) / 0.85)"
            : "none",
          opacity: live ? 1 : 0.45,
        }}
      />
      <div className="font-mono text-[34px] leading-none tabular-nums text-bone">
        {value}
      </div>
      <div className="mt-2 font-mono text-[9px] uppercase tracking-[0.18em] text-muted">
        {label}
      </div>
    </div>
  );
}

/** Page masthead: oversized uppercase wordmark with a bracketed kicker. */
export function Masthead({
  kicker,
  title,
  lede,
  index,
}: {
  kicker: string;
  title: string;
  lede?: string;
  index?: string;
}) {
  return (
    <header className="border-b border-hairline pb-5">
      <div className="flex items-baseline gap-3">
        <span className="tag text-muted">{kicker}</span>
        <span className="h-px flex-1 bg-hairline" />
        {index && <span className="panel-index">{index}</span>}
      </div>
      <h1 className="display reg mt-3 text-[clamp(2.6rem,7vw,5.2rem)] text-bone">
        {title}
      </h1>
      {lede && <p className="mt-2 max-w-2xl text-sm text-ink-2">{lede}</p>}
    </header>
  );
}

/** Section label with a hairline rule running to the edge. */
export function SectionRule({ label, index }: { label: string; index?: string }) {
  return (
    <div className="mb-3 flex items-center gap-3">
      {index && <span className="panel-index">{index}</span>}
      <span className="hud-label whitespace-nowrap text-ink-2">{label}</span>
      <span className="h-px flex-1 bg-hairline" />
    </div>
  );
}
