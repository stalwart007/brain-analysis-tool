"use client";

/**
 * Run history on the calibration station: what was predicted, and where a real
 * outcome disagreed.
 *
 * WHY THIS IS NOT A TREND LINE. It was an area chart with a monotone spline
 * through mean engagement and mean intent, which asserted two things that are
 * not true of this data. First, that consecutive runs are consecutive samples
 * of one process — they are not; run 4 might be a pricing page and run 5 a
 * 30-second spot, and a curve drawn between them invites reading a slope that
 * means nothing. Second, that each run is a point. Every one of these is a mean
 * over twins and arrives WITH an interval, and this codebase's position on
 * means without their spread is stated in a dozen docstrings; the chart that
 * introduced the product's numbers was the one place ignoring it.
 *
 * So: discrete marks, each carrying its 95% interval, with no line between
 * them. Nothing is interpolated because nothing joins.
 *
 * THE ACTUAL IS THE POINT OF THIS STATION. Where a real outcome has been
 * recorded, it is drawn as a second mark with a connector to the prediction —
 * the connector's length IS the error, and its direction is the sign of the
 * bias. A run still awaiting its outcome shows the prediction alone, so the
 * chart also reads as "how much of this history is actually validated", which
 * is the question the station exists to answer and which no arrangement of the
 * old chart could have shown.
 *
 * It shares the product's cursor and frame, so it hovers, pins, exports CSV and
 * announces itself like every other chart. No independent animation timer.
 */

import { useMemo, useState } from "react";
import ChartFrame from "./charts/ChartFrame";
import { useChartCursor } from "./charts/cursor";
import type { SwarmRunRow } from "@/lib/api";

// Validated dark categorical slots 1 & 2.
const ENGAGEMENT = "#3987e5";
const INTENT = "#d95926";

const PAD_L = 26;
const PAD_R = 8;
const PAD_T = 8;
const PAD_B = 16;
const W = 560;
const H = 190;

interface Mark {
  run: string;
  id: string;
  scenario: string;
  load: string;
  engagement: number;
  intent: number;
  engagementCi: [number, number] | null;
  intentCi: [number, number] | null;
  actualEngagement: number | null;
  actualIntent: number | null;
  twins: number;
}

export default function RunsChart({ runs }: { runs: SwarmRunRow[] }) {
  const [hidden, setHidden] = useState<Set<string>>(new Set());

  const marks = useMemo<Mark[]>(
    () =>
      [...runs]
        .reverse() // oldest → newest
        .map((r, i) => ({
          run: `#${i + 1}`,
          id: r.id,
          scenario: r.result.scenario_preview ?? r.request.scenario ?? "",
          load: r.request.cognitive_load,
          engagement: r.result.mean_engagement,
          intent: r.result.mean_intent,
          engagementCi: r.result.engagement_ci ?? null,
          intentCi: r.result.intent_ci ?? null,
          actualEngagement: r.actuals?.engagement ?? null,
          actualIntent: r.actuals?.intent ?? null,
          twins: r.result.twin_count,
        })),
    [runs]
  );

  const cursor = useChartCursor(
    "run-history",
    marks.length,
    "Run history",
    (i) => {
      const m = marks[i];
      if (!m) return "";
      const val = m.actualIntent != null ? `, actual intent ${m.actualIntent.toFixed(2)}` : ", no actual recorded";
      return `run ${m.run}, ${m.twins} twins, load ${m.load}, predicted intent ${m.intent.toFixed(2)}${val}`;
    },
    // Vertices, not slices. Every mark here sits at an exact x — `x(i)` places
    // run i at a point, with nothing occupying the space between runs — so the
    // default "band" mapping was wrong twice over: it hit-tested the midpoints
    // between marks as if each run owned a slice, and it drew the crosshair as
    // a band whose edges fall where no data is. The visible symptom is the
    // crosshair separating from the mark it is supposed to be reading.
    { inset: [PAD_L / W, PAD_R / W], mapping: "point" }
  );

  if (marks.length === 0) {
    return (
      <div className="flex h-full items-center justify-center text-center text-sm text-muted">
        Run history appears here after your first swarm.
      </div>
    );
  }

  const n = marks.length;
  const x = (i: number) =>
    PAD_L + (n === 1 ? (W - PAD_L - PAD_R) / 2 : (i / (n - 1)) * (W - PAD_L - PAD_R));
  const y = (v: number) => H - PAD_B - v * (H - PAD_T - PAD_B);

  const validated = marks.filter((m) => m.actualIntent != null || m.actualEngagement != null).length;

  const series = [
    {
      key: "intent",
      label: "Intent",
      color: INTENT,
      values: marks.map((m) => m.intent),
      intervals: marks.map((m) => m.intentCi),
    },
    {
      key: "engagement",
      label: "Engagement",
      color: ENGAGEMENT,
      values: marks.map((m) => m.engagement),
      intervals: marks.map((m) => m.engagementCi),
    },
    {
      key: "actual",
      label: "Actual intent",
      color: "#16d016",
      values: marks.map((m) => m.actualIntent),
    },
  ];

  const lanes = [
    { key: "intent", color: INTENT, pred: (m: Mark) => m.intent, ci: (m: Mark) => m.intentCi, act: (m: Mark) => m.actualIntent },
    { key: "engagement", color: ENGAGEMENT, pred: (m: Mark) => m.engagement, ci: (m: Mark) => m.engagementCi, act: (m: Mark) => m.actualEngagement },
  ];

  return (
    <ChartFrame
      cursor={cursor}
      title="Run history"
      categories={marks.map((m) => m.run)}
      series={series}
      hidden={hidden}
      onToggleSeries={(k) =>
        setHidden((s) => {
          const next = new Set(s);
          next.has(k) ? next.delete(k) : next.add(k);
          return next;
        })
      }
      height="auto"
      footnote={
        `${validated} of ${n} run${n === 1 ? "" : "s"} has a recorded outcome. ` +
        "Marks are per-run means with 95% intervals — not a series, so they are not joined."
      }
    >
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label="Run history">
        {/* gridlines at the quartiles — enough to read a level, few enough to
            stay out of the way of the intervals */}
        {[0, 0.25, 0.5, 0.75, 1].map((t) => (
          <g key={t}>
            <line
              x1={PAD_L}
              x2={W - PAD_R}
              y1={y(t)}
              y2={y(t)}
              stroke="rgba(223,217,217,0.10)"
              strokeWidth={0.5}
            />
            <text x={4} y={y(t) + 3} className="fill-muted" style={{ fontSize: 8, fontFamily: "var(--font-mono, monospace)" }}>
              {t.toFixed(1)}
            </text>
          </g>
        ))}

        {lanes
          .filter((l) => !hidden.has(l.key))
          .map((lane) => (
            <g key={lane.key}>
              {marks.map((m, i) => {
                const ci = lane.ci(m);
                const pv = lane.pred(m);
                const av = lane.act(m);
                // Lanes are nudged apart so two predictions at the same level
                // do not draw on top of each other and read as one mark.
                const dx = lane.key === "intent" ? -2.5 : 2.5;
                const cx = x(i) + dx;
                return (
                  <g key={m.id}>
                    {ci && (
                      <line
                        x1={cx}
                        x2={cx}
                        y1={y(ci[0])}
                        y2={y(ci[1])}
                        stroke={lane.color}
                        strokeWidth={1}
                        opacity={0.45}
                      />
                    )}
                    {/* The connector IS the error: length is |predicted −
                        actual| and direction is the sign of the bias. */}
                    {av != null && (
                      <>
                        <line
                          x1={cx}
                          x2={cx}
                          y1={y(pv)}
                          y2={y(av)}
                          stroke="#16d016"
                          strokeWidth={1}
                          strokeDasharray="2 1.5"
                          opacity={0.8}
                        />
                        <rect x={cx - 2.5} y={y(av) - 2.5} width={5} height={5} fill="#16d016" />
                      </>
                    )}
                    <circle
                      cx={cx}
                      cy={y(pv)}
                      r={cursor.index === i ? 4 : 2.8}
                      fill={lane.color}
                      stroke="var(--color-page, #111)"
                      strokeWidth={0.75}
                    />
                  </g>
                );
              })}
            </g>
          ))}

        {/* x labels, thinned so they never collide */}
        {marks.map((m, i) => {
          const step = Math.ceil(n / 12);
          if (i % step !== 0 && i !== n - 1) return null;
          return (
            <text
              key={m.id}
              x={x(i)}
              y={H - 4}
              textAnchor="middle"
              className="fill-muted"
              style={{ fontSize: 8, fontFamily: "var(--font-mono, monospace)" }}
            >
              {m.run}
            </text>
          );
        })}
      </svg>
    </ChartFrame>
  );
}
