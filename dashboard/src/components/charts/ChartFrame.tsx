"use client";

/**
 * The interactive shell every chart renders inside.
 *
 * It supplies the parts that were missing everywhere and had to be identical
 * everywhere: an instant tooltip, a crosshair, a legend, keyboard focus, and a
 * copy-out. Eight hand-rolled SVG charts each growing their own version of
 * these is how a product ends up with eight subtly different behaviours.
 *
 * What it replaces: 21 native `title=` attributes. Those are a browser default
 * — roughly a second of delay, unstyleable, invisible on touch, and announced
 * inconsistently by screen readers. They read as "no tooltip" to most people
 * because nobody waits a second with a mouse held still.
 */

import { useEffect, useRef, useState, type ReactNode } from "react";
import type { ChartCursor } from "./cursor";

export interface ChartSeries {
  key: string;
  label: string;
  color: string;
  /** one value per index; null renders as "n/a" rather than as zero */
  values: (number | null)[];
  /** optional per-index interval, shown with the value */
  intervals?: ([number, number] | null)[];
  format?: (v: number) => string;
}

const fmtDefault = (v: number) => (Math.abs(v) < 1 ? v.toFixed(3) : v.toFixed(2));

export default function ChartFrame({
  cursor,
  title,
  categories,
  series,
  hidden,
  onToggleSeries,
  footnote,
  height = 160,
  children,
}: {
  cursor: ChartCursor;
  title: string;
  /** x-axis labels, one per index */
  categories: string[];
  /** what the tooltip reads out, and what "copy" exports */
  series: ChartSeries[];
  hidden?: Set<string>;
  onToggleSeries?: (key: string) => void;
  footnote?: string;
  height?: number;
  children: ReactNode;
}) {
  const hostRef = useRef<HTMLDivElement>(null);
  const [pointer, setPointer] = useState<{ x: number; y: number } | null>(null);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    if (!copied) return;
    const t = setTimeout(() => setCopied(false), 1400);
    return () => clearTimeout(t);
  }, [copied]);

  const i = cursor.index;
  const visible = series.filter((s) => !hidden?.has(s.key));

  async function copyCsv() {
    const cols = series.map((s) => s.label);
    const rows = categories.map((c, r) =>
      [c, ...series.map((s) => (s.values[r] == null ? "" : String(s.values[r])))].join(",")
    );
    const csv = [["position", ...cols].join(","), ...rows].join("\n");
    try {
      await navigator.clipboard.writeText(csv);
      setCopied(true);
    } catch {
      setCopied(false);
    }
  }

  return (
    <div className="relative" ref={hostRef}>
      <div className="mb-1 flex items-baseline justify-between gap-2">
        <span className="hud-label text-muted">{title}</span>
        <div className="flex items-center gap-2">
          {cursor.pinned !== null && (
            <button
              onClick={cursor.clear}
              className="font-mono text-[9px] text-accent transition hover:text-ink"
              title="Release the pinned position (Escape)"
            >
              pinned {categories[cursor.pinned] ?? cursor.pinned + 1} ✕
            </button>
          )}
          <button
            onClick={copyCsv}
            className="font-mono text-[9px] text-muted transition hover:text-ink-2"
            title="Copy this chart's underlying series as CSV"
          >
            {copied ? "copied" : "copy"}
          </button>
        </div>
      </div>

      {/* legend doubles as the series toggle — muting a dimension to read the
          rest is the most common thing anyone wants from a multi-series chart */}
      {series.length > 1 && onToggleSeries && (
        <div className="mb-1 flex flex-wrap gap-x-3 gap-y-1">
          {series.map((s) => {
            const off = hidden?.has(s.key);
            return (
              <button
                key={s.key}
                onClick={() => onToggleSeries(s.key)}
                aria-pressed={!off}
                className={`flex items-center gap-1 font-mono text-[9px] transition ${
                  off ? "text-muted line-through" : "text-ink-2"
                }`}
              >
                <span
                  className="inline-block h-1.5 w-1.5 rounded-full"
                  style={{ background: s.color, opacity: off ? 0.3 : 1 }}
                />
                {s.label}
              </button>
            );
          })}
        </div>
      )}

      <div
        {...cursor.surfaceProps}
        onPointerMove={(e) => {
          cursor.surfaceProps.onPointerMove(e);
          const r = hostRef.current?.getBoundingClientRect();
          if (r) setPointer({ x: e.clientX - r.left, y: e.clientY - r.top });
        }}
        onPointerLeave={() => {
          cursor.surfaceProps.onPointerLeave();
          setPointer(null);
        }}
        className="relative focus-visible:ring-1 focus-visible:ring-accent/60"
        style={{ ...cursor.surfaceProps.style, height }}
      >
        {children}

        {/* crosshair: a band rather than a line, because every axis here is
            categorical and a line between two bars is ambiguous */}
        {i !== null && cursor.length > 0 && (
          <div
            aria-hidden
            className="pointer-events-none absolute inset-y-0"
            style={{
              left: `${(i / cursor.length) * 100}%`,
              width: `${(1 / cursor.length) * 100}%`,
              background: cursor.isPinned(i)
                ? "rgb(var(--region-rgb) / 0.16)"
                : "rgba(255,255,255,0.06)",
              borderLeft: cursor.isPinned(i)
                ? "1px solid rgb(var(--region-rgb) / 0.6)"
                : "1px solid rgba(255,255,255,0.18)",
            }}
          />
        )}
      </div>

      {footnote && (
        <p className="mt-1 font-mono text-[9px] leading-relaxed text-muted">{footnote}</p>
      )}

      {/* Tooltip. Instant, styled, and flipped near the right edge so it never
          leaves the panel. */}
      {i !== null && pointer && (
        <div
          role="tooltip"
          className="pointer-events-none absolute z-30 min-w-32 rounded-lg border border-hairline bg-page/95 px-2.5 py-1.5 shadow-lg backdrop-blur-sm"
          style={{
            left: pointer.x > (hostRef.current?.clientWidth ?? 0) * 0.6 ? undefined : pointer.x + 14,
            right: pointer.x > (hostRef.current?.clientWidth ?? 0) * 0.6
              ? (hostRef.current?.clientWidth ?? 0) - pointer.x + 14
              : undefined,
            top: Math.max(0, pointer.y - 8),
          }}
        >
          <div className="font-mono text-[10px] text-ink">
            {categories[i] ?? `#${i + 1}`}
          </div>
          {visible.map((s) => {
            const v = s.values[i];
            const ci = s.intervals?.[i];
            return (
              <div key={s.key} className="mt-0.5 flex items-baseline gap-1.5">
                <span
                  className="inline-block h-1.5 w-1.5 shrink-0 rounded-full"
                  style={{ background: s.color }}
                />
                <span className="font-mono text-[9px] text-muted">{s.label}</span>
                <span className="ml-auto font-mono text-[10px] tabular-nums text-ink-2">
                  {v == null ? (
                    <span className="text-muted">n/a</span>
                  ) : (
                    (s.format ?? fmtDefault)(v)
                  )}
                  {/* The interval travels with the number, never separately —
                      a mean without its spread is the thing this codebase
                      spends most of its docstrings warning about. */}
                  {ci && (
                    <span className="ml-1 text-muted">
                      {/* Zero width is not precision — it is the 0.1 quantisation
                          of twin scores showing through, so it says so rather
                          than rendering as a very tight interval. */}
                      {ci[1] - ci[0] <= 0
                        ? " · no spread"
                        : ` · [${fmtDefault(ci[0])}, ${fmtDefault(ci[1])}]`}
                    </span>
                  )}
                </span>
              </div>
            );
          })}
          {cursor.pinned === null && (
            <div className="mt-1 font-mono text-[8px] text-muted">click to pin</div>
          )}
        </div>
      )}
    </div>
  );
}
