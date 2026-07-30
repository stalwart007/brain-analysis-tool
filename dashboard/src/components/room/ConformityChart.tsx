"use client";

/**
 * INFORMATION, OR EACH OTHER?
 *
 * A room that agrees more than its evidence warrants has not reached consensus,
 * it has reached compliance. The backend measures how fast the spread of stated
 * positions collapses, and compares it to how fast a Bayesian updater's would
 * collapse given the same signals. The RATIO is the finding: 1.0 means the
 * agreement was informational; 3.0 means two thirds of it was social.
 *
 * Beside the two rates, the spread that was actually MEASURED, round by round —
 * and the placebo arm's spread on the same axes, because a control room whose
 * spread collapses just as fast is the fastest way to see that none of this
 * measured deliberation at all.
 */

import type { RoomConformity } from "@/lib/api";
import ChartFrame from "../charts/ChartFrame";
import { useChartCursor } from "../charts/cursor";
import { fmt2 } from "./scale";

/** Tone follows the RATIO, which is the measurement — not the verdict string,
 *  which is the backend's prose and is printed verbatim beside it. */
function tone(ratio: number): { color: string; label: string } {
  if (ratio >= 2) return { color: "#f0605f", label: "far beyond information" };
  if (ratio >= 1.4) return { color: "#d95926", label: "more than information" };
  if (ratio >= 0.7) return { color: "#16d016", label: "about what the evidence buys" };
  return { color: "#3987e5", label: "slower than a rational updater" };
}

export default function ConformityChart({
  conformity,
  spreadReal: spreadFromTurns,
  spreadPlacebo,
  rounds,
}: {
  conformity?: RoomConformity | null;
  /** measured cross-member SD of stated positions per round, real arm, reduced
   *  from the transcript — used only when the estimator did not return its own */
  spreadReal: (number | null)[];
  /** the same for the control arm, or null when no placebo arm was run */
  spreadPlacebo: (number | null)[] | null;
  rounds: number;
}) {
  // The estimator's own sd_by_round wins: it is the series the rates were
  // computed from, and plotting a second reduction beside those rates would put
  // two definitions of "spread" on one chart.
  const spreadReal =
    conformity?.sd_by_round && conformity.sd_by_round.length
      ? conformity.sd_by_round
      : spreadFromTurns;
  const refused = conformity?.refused ?? null;
  const measurableRates =
    !!conformity && !refused && Number.isFinite(conformity.ratio);
  const W = 720;
  const H = 130;
  const PAD_X = 30;
  const PAD_Y = 14;

  const cursor = useChartCursor(
    "room-rounds",
    rounds + 1,
    "Spread of stated positions by round",
    (r) => {
      const v = spreadReal[r] ?? null;
      return v === null
        ? `round ${r}, spread not measurable`
        : `round ${r}, spread ${v.toFixed(3)}`;
    },
    // The inset is the axis gutter as a fraction of the surface: without it the
    // crosshair drifts a round off near the edges, because the pointer maths
    // would use the element box instead of the plot box.
    { inset: [PAD_X / W, PAD_X / W], mapping: "point" }
  );

  const measurable = spreadReal.some((v) => v !== null);
  const top = Math.max(
    0.005,
    ...[...spreadReal, ...(spreadPlacebo ?? [])].map((v) => v ?? 0)
  );
  const x = (r: number) => PAD_X + (rounds < 1 ? 0.5 : r / rounds) * (W - PAD_X * 2);
  const y = (v: number) => H - PAD_Y - (v / top) * (H - PAD_Y * 2);

  const t = measurableRates && conformity ? tone(conformity.ratio) : null;
  const rateTop = measurableRates && conformity
    ? Math.max(0.01, Math.abs(conformity.observed_rate), Math.abs(conformity.bayes_rate))
    : 1;

  return (
    <section className="border border-hairline bg-black/40">
      <header className="flex flex-wrap items-baseline gap-x-3 gap-y-1 border-b border-hairline px-3 py-2">
        <span className="hud-label" style={{ color: "rgb(var(--region-rgb))" }}>
          CONFORMITY
        </span>
        <span className="font-mono text-[9px] text-muted">
          did they converge on information, or on each other
        </span>
      </header>

      {!measurableRates || !conformity ? (
        <p className="px-3 py-3 text-[12px] leading-relaxed text-ink-2">
          <span className="hud-label mr-2 text-critical">
            {refused ? "REFUSED" : "NOT COMPUTED"}
          </span>
          {refused ??
            "The conformity test needs enough rounds to measure a collapse rate at all, and this run did not produce one."}{" "}
          <span className="text-muted">
            The spread below is still real — it is a description, not a test.
          </span>
        </p>
      ) : (
        <div className="border-b border-hairline px-3 py-3">
          <div className="flex flex-wrap items-end gap-x-6 gap-y-3">
            <div>
              <span className="hud-label">EXCESS AGREEMENT</span>
              <div className="flex items-baseline gap-2">
                <span
                  className="font-display text-[30px] font-semibold leading-none tabular-nums"
                  style={{ color: t?.color }}
                >
                  {conformity.ratio.toFixed(2)}×
                </span>
                <span className="font-mono text-[9px] uppercase tracking-[0.14em]" style={{ color: t?.color }}>
                  {t?.label}
                </span>
              </div>
              <p className="mt-0.5 font-mono text-[9px] text-muted">
                observed collapse ÷ Bayesian benchmark
              </p>
            </div>

            {/* the two rates, on one scale, so the ratio above is checkable */}
            <div className="min-w-52 flex-1">
              {[
                {
                  label: "observed",
                  v: conformity.observed_rate,
                  color: t?.color ?? "#3987e5",
                  hint: "how fast the room's stated positions actually converged",
                },
                {
                  label: "Bayesian benchmark",
                  v: conformity.bayes_rate,
                  color: "rgba(223,217,217,0.45)",
                  hint: "how fast a rational updater would converge on the same signals",
                },
              ].map((row) => (
                <div key={row.label} className="mt-1 flex items-center gap-2" title={row.hint}>
                  <span className="w-32 shrink-0 font-mono text-[9px] text-muted">{row.label}</span>
                  <span className="h-2.5 flex-1 bg-white/[0.03]">
                    <span
                      className="block h-2.5"
                      style={{
                        width: `${Math.max(1, (Math.abs(row.v) / rateTop) * 100)}%`,
                        background: row.color,
                      }}
                    />
                  </span>
                  <span className="w-12 shrink-0 text-right font-mono text-[10px] tabular-nums text-ink-2">
                    {fmt2(row.v)}
                  </span>
                </div>
              ))}
            </div>
          </div>

          <p className="verdict mt-2.5 text-[12px] text-ink-2">
            <span className="hud-label mr-2">VERDICT</span>
            {conformity.verdict}
          </p>
        </div>
      )}

      <div className="p-3">
        {measurable ? (
          <ChartFrame
            cursor={cursor}
            title="SPREAD OF STATED POSITIONS"
            categories={Array.from({ length: rounds + 1 }, (_, r) => (r === 0 ? "R0" : `R${r}`))}
            series={[
              {
                key: "real",
                label: "real room",
                color: "#4fb6ff",
                values: spreadReal,
                format: (v) => v.toFixed(3),
              },
              ...(spreadPlacebo
                ? [
                    {
                      key: "placebo",
                      label: "placebo room",
                      color: "#d95926",
                      values: spreadPlacebo,
                      format: (v: number) => v.toFixed(3),
                    },
                  ]
                : []),
            ]}
            footnote={
              spreadPlacebo
                ? "cross-member variance of the positions stated each round, taken within each replicate then averaged · the control room heard statements from a DIFFERENT room, so any collapse there is agreeableness, not deliberation"
                : "cross-member variance of the positions stated each round, taken within each replicate then averaged · no control arm was run, so there is nothing to compare this collapse against"
            }
            height="auto"
          >
            <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full" aria-hidden>
              <line
                x1={PAD_X}
                y1={y(0)}
                x2={W - PAD_X}
                y2={y(0)}
                stroke="rgba(223,217,217,0.16)"
              />
              <text
                x={PAD_X - 4}
                y={y(top) + 3}
                textAnchor="end"
                style={{ fontSize: 8, fontFamily: "var(--font-mono)", fill: "var(--color-muted)" }}
              >
                {top.toFixed(2)}
              </text>
              <text
                x={PAD_X - 4}
                y={y(0) + 3}
                textAnchor="end"
                style={{ fontSize: 8, fontFamily: "var(--font-mono)", fill: "var(--color-muted)" }}
              >
                0
              </text>
              {[
                { vals: spreadPlacebo, color: "#d95926", dash: "4 3" },
                { vals: spreadReal, color: "#4fb6ff", dash: undefined },
              ].map((line, k) =>
                !line.vals ? null : (
                  <g key={k}>
                    <polyline
                      points={line.vals
                        .map((v, r) => (v === null ? null : `${x(r)},${y(v)}`))
                        .filter(Boolean)
                        .join(" ")}
                      fill="none"
                      stroke={line.color}
                      strokeWidth={1.8}
                      strokeDasharray={line.dash}
                    />
                    {line.vals.map((v, r) =>
                      v === null ? null : (
                        <circle
                          key={r}
                          cx={x(r)}
                          cy={y(v)}
                          r={cursor.isMarked(r) ? 3.4 : 2.2}
                          fill={line.color}
                        />
                      )
                    )}
                  </g>
                )
              )}
              {Array.from({ length: rounds + 1 }, (_, r) => (
                <text
                  key={r}
                  x={x(r)}
                  y={H - 2}
                  textAnchor="middle"
                  style={{
                    fontSize: 8,
                    fontFamily: "var(--font-mono)",
                    fill: cursor.isMarked(r) ? "var(--color-ink)" : "var(--color-muted)",
                  }}
                >
                  R{r}
                </text>
              ))}
            </svg>
          </ChartFrame>
        ) : (
          <p className="font-mono text-[10px] leading-relaxed text-muted">
            Fewer than two members stated a position in any round, so there is no
            spread to collapse and nothing to plot.
          </p>
        )}
      </div>
    </section>
  );
}
