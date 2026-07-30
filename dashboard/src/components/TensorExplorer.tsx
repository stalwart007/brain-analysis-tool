"use client";

/**
 * THE APPRAISAL TENSOR, all of it.
 *
 * A neuro study measures nine appraisal dimensions across every beat for every
 * twin. The results surface rendered ONE of them — attention — as a line, and
 * collapsed the other eight into a seven-bar brain map averaged over the whole
 * piece. Eight ninths of the measurement had nowhere to be seen, and the eight
 * missing dimensions are where the MECHANISM lives: a piece that holds
 * attention while producing no reward anticipation has a completely different
 * problem from one that loses attention, and both looked identical.
 *
 * This is the missing view. Rows are dimensions, columns are beats, cell colour
 * is the population mean. Reading across a row is a dimension's arc; reading
 * down a column is everything that happened at one beat. That second reading is
 * the one that was impossible before and is usually the answer: beat 3 loses
 * attention AND spikes cognitive effort AND drops reward anticipation, so it is
 * not boring, it is confusing.
 *
 * DESIGN, per the system's rules:
 * · Colour is data. Chrome stays bone/void; the only saturated hues in here are
 *   the cell values, on a diverging scale anchored at each dimension's neutral
 *   point so "above/below expectation" is visible without reading a number.
 * · Zero radius. This is instrument, not tissue.
 * · It shares the "beats" cursor domain with the attention field, the retention
 *   curve and the peak-index distribution, so hovering a beat here lights up the
 *   same beat everywhere. Four views of one tensor behaving as one instrument.
 * · No independent animation timer — the only motion is a CSS transition on
 *   cell opacity, driven by interaction rather than by a clock.
 */

import { useMemo, useState } from "react";
import type { ChartCursor } from "./charts/cursor";

/** Order is the reading order of a reaction, not alphabetical: what grabbed
 *  them, how it felt, how activated they were, then the appraisals that explain
 *  why, then the cost of processing it. */
export const DIMENSIONS: { key: string; label: string; neutral: number; hint: string }[] = [
  { key: "attention", label: "attention", neutral: 0.5, hint: "0 ignored → 1 riveted" },
  { key: "valence", label: "valence", neutral: 0.0, hint: "−1 repelled → +1 delighted" },
  { key: "arousal", label: "arousal", neutral: 0.5, hint: "0 calm → 1 activated" },
  { key: "novelty", label: "novelty", neutral: 0.5, hint: "0 seen it before → 1 surprising" },
  { key: "goal_relevance", label: "goal relevance", neutral: 0.5, hint: "0 nothing for me → 1 exactly my problem" },
  { key: "reward_anticipation", label: "reward anticip.", neutral: 0.5, hint: "0 nothing promised → 1 can't wait" },
  { key: "social_resonance", label: "social resonance", neutral: 0.5, hint: "0 no urge to discuss → 1 must tell someone" },
  { key: "threat", label: "threat", neutral: 0.5, hint: "0 safe → 1 uneasy" },
  { key: "cognitive_effort", label: "cognitive effort", neutral: 0.5, hint: "0 effortless → 1 confusing" },
];

/**
 * Diverging scale anchored at the dimension's neutral point.
 *
 * Sequential (dark→bright) was tried and is wrong for this data: it makes
 * "high threat" and "high attention" read as the same event, when one is the
 * goal and the other is the failure. Diverging from neutral says ABOVE or
 * BELOW, and the reader supplies the valence from the row label — which is
 * the only place the domain knowledge actually lives.
 *
 * The two hues are the validated dark data-viz pair already used across the
 * product (#3987e5 / #d95926), CVD-checked against this surface.
 */
function cellColor(value: number, neutral: number, span: number): string {
  const t = Math.max(-1, Math.min(1, (value - neutral) / span));
  const mag = Math.abs(t);
  // Perceptual floor so a near-neutral cell is still visible as measured
  // rather than reading as missing data.
  const alpha = 0.1 + mag * 0.78;
  return t >= 0 ? `rgba(57, 135, 229, ${alpha})` : `rgba(217, 89, 38, ${alpha})`;
}

export default function TensorExplorer({
  curves,
  bands,
  segments,
  cursor,
  twinCurves,
}: {
  /** dimension -> per-beat population mean */
  curves: Record<string, number[]>;
  /** dimension -> per-beat simultaneous band, or null when too few twins */
  bands?: Record<string, ([number, number] | null)[] | null>;
  segments: string[];
  /** the shared "beats" cursor — same domain as every other beat view */
  cursor: ChartCursor;
  /** optional per-twin attention traces, for the spread strip */
  twinCurves?: number[][];
}) {
  const [dim, setDim] = useState<string | null>(null);
  const beats = segments.length;

  const rows = useMemo(
    () => DIMENSIONS.filter((d) => Array.isArray(curves[d.key]) && curves[d.key].length === beats),
    [curves, beats]
  );

  if (!rows.length || beats < 2) return null;

  const active = cursor.index;
  const focusRow = dim ? rows.find((r) => r.key === dim) : null;

  return (
    <div className="border border-hairline bg-black/40">
      <header className="flex flex-wrap items-baseline gap-x-3 border-b border-hairline px-3 py-2">
        <span className="hud-label">APPRAISAL TENSOR</span>
        <span className="font-mono text-[9px] text-muted">
          {rows.length} dimensions × {beats} beats
          {twinCurves?.length ? ` × ${twinCurves.length} twins` : ""}
        </span>
        <span className="ml-auto font-mono text-[9px] text-muted">
          <span style={{ color: "rgba(57,135,229,0.95)" }}>■</span> above neutral ·{" "}
          <span style={{ color: "rgba(217,89,38,0.95)" }}>■</span> below
        </span>
      </header>

      <div className="overflow-x-auto">
        <table className="w-full border-collapse">
          <thead>
            <tr>
              <th className="sticky left-0 z-10 bg-[#0b0c0e] px-2 py-1 text-left" />
              {segments.map((s, i) => (
                <th
                  key={i}
                  onMouseEnter={() => cursor.setHovered(i)}
                  onMouseLeave={() => cursor.setHovered(null)}
                  onClick={() => cursor.togglePin(i)}
                  title={s}
                  className={`cursor-pointer px-1 py-1 font-mono text-[9px] font-normal transition-colors ${
                    cursor.isMarked(i) ? "text-ink" : "text-muted"
                  }`}
                >
                  B{i + 1}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => {
              const series = curves[row.key];
              // Span from the row's own extent so a flat dimension does not
              // render as uniformly neutral — the contrast within a row is the
              // information, and a global span would flatten every row that
              // happens to sit in a narrow range.
              const span = Math.max(
                0.12,
                ...series.map((v) => Math.abs(v - row.neutral))
              );
              const isFocus = dim === row.key;
              return (
                <tr
                  key={row.key}
                  className={`transition-opacity ${dim && !isFocus ? "opacity-35" : ""}`}
                >
                  <th
                    scope="row"
                    onClick={() => setDim(isFocus ? null : row.key)}
                    title={row.hint}
                    className={`sticky left-0 z-10 cursor-pointer whitespace-nowrap bg-[#0b0c0e] py-1 pl-2 pr-3 text-left font-mono text-[9px] font-normal transition-colors ${
                      isFocus ? "text-ink" : "text-muted hover:text-ink-2"
                    }`}
                  >
                    {row.label}
                  </th>
                  {series.map((v, i) => {
                    const band = bands?.[row.key]?.[i] ?? null;
                    return (
                      <td
                        key={i}
                        onMouseEnter={() => cursor.setHovered(i)}
                        onMouseLeave={() => cursor.setHovered(null)}
                        onClick={() => cursor.togglePin(i)}
                        className="cursor-pointer p-0"
                      >
                        <div
                          className="h-6 min-w-7 transition-all"
                          style={{
                            background: cellColor(v, row.neutral, span),
                            outline: cursor.isMarked(i)
                              ? "1px solid rgba(223,217,217,0.55)"
                              : "none",
                            outlineOffset: "-1px",
                            // A cell whose simultaneous band is wide is a cell
                            // we do not actually know the value of. Fading it
                            // is the only honest way to draw it beside a cell
                            // we do — an identical block would claim identical
                            // precision.
                            opacity: band ? Math.max(0.4, 1 - (band[1] - band[0]) * 1.1) : 1,
                          }}
                        />
                      </td>
                    );
                  })}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* ── the readout: a beat column, or a dimension row ───────────── */}
      <div className="min-h-[3.25rem] border-t border-hairline px-3 py-2">
        {active !== null ? (
          <div>
            <div className="flex items-baseline gap-2">
              <span className="hud-label">BEAT {active + 1}</span>
              <span className="flex-1 truncate text-[11px] text-ink-2">{segments[active]}</span>
              {cursor.pinned !== null && (
                <button onClick={cursor.clear} className="font-mono text-[9px] text-accent hover:text-ink">
                  pinned ✕
                </button>
              )}
            </div>
            {/* Everything that happened at this beat, sorted by how far from
                neutral it sits. This is the column read — the one the old
                surface made impossible, and usually the one that explains why
                a beat underperforms. */}
            <div className="mt-1 flex flex-wrap gap-x-3 gap-y-0.5">
              {rows
                .map((r) => ({ r, v: curves[r.key][active], d: curves[r.key][active] - r.neutral }))
                .sort((a, b) => Math.abs(b.d) - Math.abs(a.d))
                .map(({ r, v, d }) => (
                  <span key={r.key} className="font-mono text-[10px] tabular-nums">
                    <span className="text-muted">{r.label}</span>{" "}
                    <span style={{ color: d >= 0 ? "#4fb6ff" : "#d95926" }}>{v.toFixed(2)}</span>
                  </span>
                ))}
            </div>
          </div>
        ) : focusRow ? (
          <p className="font-mono text-[10px] leading-relaxed text-muted">
            <span className="text-ink-2">{focusRow.label}</span> — {focusRow.hint} · range{" "}
            {Math.min(...curves[focusRow.key]).toFixed(2)}–
            {Math.max(...curves[focusRow.key]).toFixed(2)} · click the row again to release
          </p>
        ) : (
          <p className="font-mono text-[10px] leading-relaxed text-muted">
            Hover a column to read every dimension at that beat; click to pin it
            across all the charts. Click a row label to isolate one dimension.
            Faded cells have wide simultaneous bands — the value there is not
            well determined.
          </p>
        )}
      </div>
    </div>
  );
}
