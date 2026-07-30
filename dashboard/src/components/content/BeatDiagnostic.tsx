"use client";

/**
 * WHY a beat is weak, not just that it is.
 *
 * A low attention score is a symptom with two opposite causes, and the fixes
 * point in opposite directions. A beat nobody attends to because nothing is
 * happening needs MORE — a sharper claim, a cut, a stake. A beat nobody
 * attends to because it is hard work needs LESS — fewer ideas per second,
 * plainer language, one thought at a time. Acting on the wrong diagnosis makes
 * the beat worse, and "attention: 0.31" cannot tell them apart.
 *
 * Cognitive effort is what separates them, and it was already being measured:
 * `cognitive_effort` is one of the nine appraisal dimensions every twin rates
 * on every beat. It sat in the tensor with no view that put it against
 * attention, so the distinction existed in the data and nowhere on screen —
 * the panel's own source comment describes exactly this reading ("attention
 * falls AND cognitive effort spikes means confusing, not boring, and those want
 * opposite fixes") as something the reader should perform by eye across two
 * charts.
 *
 * So it is performed here. Each beat is placed by effort against attention, the
 * quadrant it lands in is named, and the name is the instruction.
 */

import { useMemo } from "react";
import type { ChartCursor } from "@/components/charts/cursor";

export interface Verdict {
  key: "working" | "demanding" | "boring" | "confusing";
  label: string;
  /** What to do about it — imperative, one clause. */
  fix: string;
  color: string;
}

const VERDICTS: Record<Verdict["key"], Verdict> = {
  working: {
    key: "working",
    label: "working",
    fix: "holds them without making them work — leave it alone",
    color: "#4fb477",
  },
  demanding: {
    key: "demanding",
    label: "demanding",
    fix: "they are engaged but working hard — survivable briefly, fatal if sustained",
    color: "#f2ad1f",
  },
  boring: {
    key: "boring",
    label: "boring",
    fix: "nothing is at stake here — cut it, or give it a reason to exist",
    color: "#b07ff0",
  },
  confusing: {
    key: "confusing",
    label: "confusing",
    fix: "they are losing the thread — one idea at a time, plainer words",
    color: "#f0605f",
  },
};

/** The split. Deliberately the midpoint of each rated scale rather than a
 *  percentile of this run: quadrants defined relative to the content's own
 *  distribution would always find a "worst" beat, including in content where
 *  every beat is fine. */
const MID = 0.5;

export function verdictFor(attention: number, effort: number): Verdict {
  if (attention >= MID) return effort >= MID ? VERDICTS.demanding : VERDICTS.working;
  return effort >= MID ? VERDICTS.confusing : VERDICTS.boring;
}

export default function BeatDiagnostic({
  attention,
  effort,
  segments,
  cursor,
  timeLabels,
}: {
  attention: number[];
  effort: number[];
  segments: string[];
  cursor: ChartCursor;
  /** `0:24`-style stamps when the content has a real clock; beat numbers when
   *  it does not. A moment someone can go and look at beats an index. */
  timeLabels?: string[];
}) {
  const S = 230;
  const pad = 26;

  const points = useMemo(
    () =>
      segments.map((_, i) => {
        const a = attention[i] ?? 0;
        const e = effort[i] ?? 0;
        return { i, a, e, verdict: verdictFor(a, e) };
      }),
    [attention, effort, segments]
  );

  // Counted rather than asserted: "three beats are confusing" is a finding, and
  // a diagnostic that only ever shows dots makes the reader tally them.
  const tally = useMemo(() => {
    const counts = new Map<Verdict["key"], number>();
    for (const p of points) counts.set(p.verdict.key, (counts.get(p.verdict.key) ?? 0) + 1);
    return counts;
  }, [points]);

  const x = (effortValue: number) => pad + effortValue * (S - pad * 2);
  const y = (attentionValue: number) => S - pad - attentionValue * (S - pad * 2);
  const active = cursor.index;

  return (
    <div className="rounded-xl border border-hairline bg-surface-2/60 p-3">
      <div className="flex items-baseline justify-between">
        <span className="hud-label">WHY EACH BEAT LANDS OR DOES NOT</span>
        <span className="font-mono text-[9px] text-muted">attention × effort</span>
      </div>

      <div className="mt-2 flex flex-wrap gap-3">
        <svg
          viewBox={`0 0 ${S} ${S}`}
          className="h-56 w-56 shrink-0 focus-visible:outline-none"
          {...cursor.surfaceProps}
        >
          {/* Quadrant beds. Tinted at very low alpha so the dots stay the
              foreground — the regions are context, the beats are the data. */}
          {(
            [
              ["boring", pad, pad, "start", "top"],
              ["confusing", S / 2, pad, "end", "top"],
              ["working", pad, S / 2, "start", "bottom"],
              ["demanding", S / 2, S / 2, "end", "bottom"],
            ] as const
          ).map(([key, qx, qy]) => {
            const v = VERDICTS[key as Verdict["key"]];
            return (
              <g key={key}>
                <rect
                  x={qx}
                  y={qy}
                  width={(S - pad * 2) / 2}
                  height={(S - pad * 2) / 2}
                  fill={v.color}
                  opacity={0.05}
                />
              </g>
            );
          })}

          {/* Quadrant names, placed in the corners they belong to. */}
          <text x={pad + 3} y={S - pad - 4} fontSize="7" fill="#4fb477" fontFamily="ui-monospace, monospace">working</text>
          <text x={S - pad - 3} y={S - pad - 4} fontSize="7" fill="#f2ad1f" textAnchor="end" fontFamily="ui-monospace, monospace">demanding</text>
          <text x={pad + 3} y={pad + 9} fontSize="7" fill="#b07ff0" fontFamily="ui-monospace, monospace">boring</text>
          <text x={S - pad - 3} y={pad + 9} fontSize="7" fill="#f0605f" textAnchor="end" fontFamily="ui-monospace, monospace">confusing</text>

          <line x1={S / 2} y1={pad} x2={S / 2} y2={S - pad} stroke="rgba(120,165,230,0.16)" />
          <line x1={pad} y1={S / 2} x2={S - pad} y2={S / 2} stroke="rgba(120,165,230,0.16)" />
          <rect x={pad} y={pad} width={S - pad * 2} height={S - pad * 2} fill="none" stroke="rgba(120,165,230,0.15)" />

          <text x={S / 2} y={S - 6} fontSize="6.5" fill="#5f6b7d" textAnchor="middle" fontFamily="ui-monospace, monospace">cognitive effort →</text>
          <text x={9} y={S / 2} fontSize="6.5" fill="#5f6b7d" textAnchor="middle" fontFamily="ui-monospace, monospace" transform={`rotate(-90 9 ${S / 2})`}>attention →</text>

          {/* The path through the beats, in running order. A cluster tells you
              the content is one thing throughout; a march from working to
              confusing tells you where it turned. */}
          {points.length > 1 && (
            <polyline
              fill="none"
              stroke="rgba(223,217,217,0.18)"
              strokeWidth="1"
              points={points.map((p) => `${x(p.e)},${y(p.a)}`).join(" ")}
            />
          )}

          {points.map((p) => {
            const on = cursor.isMarked(p.i);
            return (
              <g key={p.i}>
                <circle
                  cx={x(p.e)}
                  cy={y(p.a)}
                  r={on ? 5.5 : 4}
                  fill={p.verdict.color}
                  opacity={active === null ? 0.9 : on ? 1 : 0.25}
                  stroke={cursor.isPinned(p.i) ? "rgb(var(--region-rgb))" : "none"}
                  strokeWidth="1.5"
                />
                <text
                  x={x(p.e)}
                  y={y(p.a) - 6.5}
                  fontSize="6"
                  fill={on ? "#dfd9d9" : "#8fa3c0"}
                  textAnchor="middle"
                  opacity={active === null ? 0.8 : on ? 1 : 0.2}
                  fontFamily="ui-monospace, monospace"
                >
                  {timeLabels?.[p.i] ?? `B${p.i + 1}`}
                </text>
              </g>
            );
          })}
        </svg>

        <div className="min-w-44 flex-1 space-y-1.5">
          {/* The tally, worst first — a reader scanning this wants the problems,
              not an alphabetical inventory. */}
          {(["confusing", "boring", "demanding", "working"] as const).map((key) => {
            const count = tally.get(key) ?? 0;
            if (!count) return null;
            const v = VERDICTS[key];
            return (
              <div key={key} className="flex items-baseline gap-1.5">
                <span
                  className="mt-0.5 inline-block h-2 w-2 shrink-0 rounded-full"
                  style={{ background: v.color }}
                />
                <span className="font-mono text-[10px]" style={{ color: v.color }}>
                  {count} {v.label}
                </span>
              </div>
            );
          })}

          {/* Reserved height so hovering the scatter does not reflow the column
              beside it. Both states are variable-length prose. */}
          <div className="mt-2 min-h-[5.5rem] border-t border-hairline pt-2">
            {active !== null && points[active] ? (
              <>
                <p className="font-mono text-[10px]" style={{ color: points[active].verdict.color }}>
                  {timeLabels?.[active] ?? `Beat ${active + 1}`} —{" "}
                  {points[active].verdict.label}
                </p>
                <p className="mt-0.5 text-[10px] leading-relaxed text-ink-2">
                  {points[active].verdict.fix}
                </p>
                <p className="mt-1 font-mono text-[9px] text-muted">
                  attention {points[active].a.toFixed(2)} · effort {points[active].e.toFixed(2)}
                </p>
                <p className="mt-1 line-clamp-2 text-[9px] leading-snug text-muted">
                  {segments[active]}
                </p>
              </>
            ) : (
              <p className="text-[10px] leading-relaxed text-muted">
                A beat can lose an audience two ways. Low attention with low
                effort is <b className="text-ink-2">boring</b> — nothing is at
                stake. Low attention with high effort is{" "}
                <b className="text-ink-2">confusing</b> — they are losing the
                thread. The fixes are opposites. Hover a beat.
              </p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
