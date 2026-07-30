"use client";

/**
 * THE TRAJECTORY — every position, every round, said and thought.
 *
 * Solid is public, dashed is private, and round 0 is the private baseline: the
 * position each member walked in with, before anyone had spoken. There is no
 * public value at round 0 and none is drawn — back-filling it with the private
 * one would invent the exact quantity this study measures.
 *
 * Focusing a member fills the band between their two lines, so their
 * falsification becomes a shape with a duration: someone who caved in round 2
 * and never came back looks nothing like someone who was quietly against it the
 * whole way, and both used to average to "gap = 0.3".
 *
 * The x-axis is ROUNDS, so both charts here share the "room-rounds" cursor
 * domain — hover round 3 in either and round 3 lights up in both. They
 * deliberately do NOT share the members domain the table uses: a member index
 * and a round index are different axes, and linking them would be a false
 * connection.
 */

import { useState } from "react";
import type { RoomFalsification, RoomMember } from "@/lib/api";
import ChartFrame, { type ChartSeries } from "../charts/ChartFrame";
import { useChartCursor, type ChartCursor } from "../charts/cursor";
import { fmt2, identityHue, positionInk, positionWord, signed } from "./scale";

const W = 720;
const H = 200;
const PAD_X = 34;
const PAD_Y = 16;

/** Split a series on nulls so a missing round breaks the line instead of
 *  drawing a straight cheat across it. */
function segments(values: (number | null)[]): { r: number; v: number }[][] {
  const out: { r: number; v: number }[][] = [];
  let run: { r: number; v: number }[] = [];
  values.forEach((v, r) => {
    if (v === null || !Number.isFinite(v)) {
      if (run.length) out.push(run);
      run = [];
    } else {
      run.push({ r, v });
    }
  });
  if (run.length) out.push(run);
  return out;
}

export default function OpinionTrajectory({
  cast,
  publicSeries,
  privateSeries,
  rounds,
  memberCursor,
  falsification,
  viewLabel,
}: {
  cast: RoomMember[];
  /** [member][round] public position, null where they did not speak */
  publicSeries: (number | null)[][];
  /** [member][round] private position — null everywhere until the reveal */
  privateSeries: (number | null)[][];
  rounds: number;
  /** the shared members cursor, so the table's pin emphasises a line here */
  memberCursor: ChartCursor;
  falsification?: RoomFalsification | null;
  /** which replicate/arm these means came from */
  viewLabel: string;
}) {
  const [hidden, setHidden] = useState<Set<string>>(new Set());
  const categories = Array.from({ length: rounds + 1 }, (_, r) =>
    r === 0 ? "Round 0 (sealed ballot)" : `Round ${r}`
  );

  const cursor = useChartCursor(
    "room-rounds",
    rounds + 1,
    "Positions by round",
    (r) => (r === 0 ? "round 0, the sealed ballot" : `round ${r} of ${rounds}`),
    { inset: [PAD_X / W, PAD_X / W], mapping: "point" }
  );

  const hasPrivate = privateSeries.some((s) => s.some((v) => v !== null));

  const series: ChartSeries[] = [];
  cast.forEach((m, i) => {
    series.push({
      key: `pub:${i}`,
      label: m.name,
      color: identityHue(i),
      values: publicSeries[i] ?? [],
      format: fmt2,
    });
    if (hasPrivate) {
      series.push({
        key: `prv:${i}`,
        label: `${m.name} · private`,
        color: identityHue(i),
        values: privateSeries[i] ?? [],
        format: fmt2,
      });
    }
  });

  const x = (r: number) => PAD_X + (rounds < 1 ? 0.5 : r / rounds) * (W - PAD_X * 2);
  const y = (v: number) => H - PAD_Y - v * (H - PAD_Y * 2);
  const focus = memberCursor.index;

  return (
    <section className="border border-hairline bg-black/40 p-3">
      <ChartFrame
        cursor={cursor}
        title={`POSITION BY ROUND — ${viewLabel}`}
        categories={categories}
        series={series}
        hidden={hidden}
        onToggleSeries={(k) =>
          setHidden((h) => {
            const next = new Set(h);
            if (next.has(k)) next.delete(k);
            else next.add(k);
            return next;
          })
        }
        footnote={
          hasPrivate
            ? "solid = said aloud · dashed = held privately · round 0 is the SEALED BALLOT, written alone before anyone spoke, so no public line starts there · the filled band is the pinned member's falsification over time · means across the replicates in view"
            : "solid = said aloud · round 0 is the sealed ballot · private positions are not streamed and arrive only with the result, so no dashed lines exist yet"
        }
        height="auto"
      >
        <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full" aria-hidden>
          {/* the 0.5 rule is the anchor of the whole colour scale: above it is
              for the motion, below it is against */}
          {[0, 0.5, 1].map((v) => (
            <g key={v}>
              <line
                x1={PAD_X}
                y1={y(v)}
                x2={W - PAD_X}
                y2={y(v)}
                stroke={v === 0.5 ? "rgba(223,217,217,0.20)" : "rgba(120,165,230,0.10)"}
                strokeDasharray={v === 0.5 ? "4 4" : undefined}
              />
              <text
                x={PAD_X - 5}
                y={y(v) + 3}
                textAnchor="end"
                style={{ fontSize: 8, fontFamily: "var(--font-mono)", fill: "var(--color-muted)" }}
              >
                {v === 1 ? "for" : v === 0 ? "against" : "0.5"}
              </text>
            </g>
          ))}

          {/* round ticks */}
          {Array.from({ length: rounds + 1 }, (_, r) => (
            <text
              key={r}
              x={x(r)}
              y={H - 3}
              textAnchor="middle"
              style={{
                fontSize: 8,
                fontFamily: "var(--font-mono)",
                fill: cursor.isMarked(r) ? "var(--color-ink)" : "var(--color-muted)",
              }}
            >
              {r === 0 ? "R0·ballot" : `R${r}`}
            </text>
          ))}

          {/* the pinned member's gap, as an area with a duration */}
          {focus !== null &&
            hasPrivate &&
            !hidden.has(`pub:${focus}`) &&
            (() => {
              const pub = publicSeries[focus] ?? [];
              const prv = privateSeries[focus] ?? [];
              const pts = pub
                .map((p, r) => ({ r, p, q: prv[r] }))
                .filter(
                  (d): d is { r: number; p: number; q: number } =>
                    d.p !== null && d.q !== null && d.q !== undefined
                );
              if (pts.length < 2) return null;
              const top = pts.map((d) => `${x(d.r)},${y(d.p)}`).join(" L ");
              const back = [...pts]
                .reverse()
                .map((d) => `${x(d.r)},${y(d.q)}`)
                .join(" L ");
              return (
                <path
                  d={`M ${top} L ${back} Z`}
                  fill="rgba(240,96,95,0.16)"
                  stroke="rgba(240,96,95,0.3)"
                  strokeWidth={0.6}
                />
              );
            })()}

          {cast.map((m, i) => {
            const hue = identityHue(i);
            const marked = memberCursor.isMarked(i);
            const dim = focus !== null && !marked;
            const showPub = !hidden.has(`pub:${i}`);
            const showPrv = hasPrivate && !hidden.has(`prv:${i}`);
            return (
              <g
                key={m.name}
                onPointerEnter={() => memberCursor.setHovered(i)}
                onPointerLeave={() => memberCursor.setHovered(null)}
                style={{ opacity: dim ? 0.16 : 1, transition: "opacity 0.18s ease" }}
              >
                {showPrv &&
                  segments(privateSeries[i] ?? []).map((seg, k) => (
                    <polyline
                      key={`prv-${k}`}
                      points={seg.map((d) => `${x(d.r)},${y(d.v)}`).join(" ")}
                      fill="none"
                      stroke={hue}
                      strokeWidth={marked ? 1.8 : 1.1}
                      strokeDasharray="4 3"
                      opacity={0.85}
                    />
                  ))}
                {showPub &&
                  segments(publicSeries[i] ?? []).map((seg, k) => (
                    <polyline
                      key={`pub-${k}`}
                      points={seg.map((d) => `${x(d.r)},${y(d.v)}`).join(" ")}
                      fill="none"
                      stroke={hue}
                      strokeWidth={marked ? 2.6 : 1.6}
                      strokeLinejoin="round"
                    />
                  ))}
                {/* vertices only for the marked member — 8 members × 5 rounds of
                    dots is a rash, not a chart */}
                {marked &&
                  showPub &&
                  (publicSeries[i] ?? []).map((v, r) =>
                    v === null ? null : (
                      <circle key={`d-${r}`} cx={x(r)} cy={y(v)} r={2.6} fill={hue} />
                    )
                  )}
              </g>
            );
          })}
        </svg>
      </ChartFrame>

      {/* ── when they started hiding: the room gap, round by round ────── */}
      {falsification?.by_round?.length ? (
        <GapByRound falsification={falsification} rounds={rounds} />
      ) : null}

      {/* ALWAYS rendered, and with tabular figures.
          This block used to exist only while a member was focused, so it
          appeared out of nothing on hover and vanished on leave — a 15px jump of
          everything below it, on every pass of the mouse. The numbers inside it
          also changed width per member without tabular figures, so the words
          after them shifted sideways as the cursor moved. */}
      <p className="mt-2 min-h-[1.7em] font-mono text-[9px] leading-relaxed tabular-nums text-muted">
        {focus === null || !cast[focus] ? (
          "Hover or pin a member — on this chart, the table, or any panel below — to read where they started and where they ended up."
        ) : (
          <>
            <span className="text-ink-2">{cast[focus].name}</span>
            {(() => {
            const pub = publicSeries[focus] ?? [];
            const prv = privateSeries[focus] ?? [];
            const last = pub.reduce<number | null>((acc, v, r) => (v !== null ? r : acc), null);
            const base = prv[0];
            if (last === null) return " has not spoken in this view.";
            const p = pub[last] as number;
            return (
              <>
                {" "}
                walked in at{" "}
                {base === null || base === undefined ? (
                  "an unknown private baseline"
                ) : (
                  <span style={{ color: positionInk(base) }}>
                    {fmt2(base)} ({positionWord(base)})
                  </span>
                )}{" "}
                and left saying{" "}
                <span style={{ color: positionInk(p) }}>
                  {fmt2(p)} ({positionWord(p)})
                </span>
                {base !== null && base !== undefined && (
                  <> · public movement from baseline {signed(p - base)}</>
                )}
                .
              </>
            );
            })()}
          </>
        )}
      </p>
    </section>
  );
}

/**
 * The room's mean falsification per round. Same axis as the trajectory above,
 * so it shares the cursor: hovering round 3 in one marks round 3 in the other.
 * This is the chart that answers "when did the room stop saying what it thinks"
 * — usually the round after the most senior person committed.
 */
function GapByRound({
  falsification,
  rounds,
}: {
  falsification: RoomFalsification;
  rounds: number;
}) {
  const byRound = falsification.by_round;
  const values: (number | null)[] = Array.from({ length: rounds + 1 }, (_, r) => {
    const hit = byRound.find((b) => b.round === r);
    return hit ? hit.gap : null;
  });
  const cursor = useChartCursor(
    "room-rounds",
    rounds + 1,
    "Room falsification gap by round",
    (r) => {
      const v = values[r];
      return v === null ? `round ${r}, no gap measured` : `round ${r}, mean gap ${v.toFixed(2)}`;
    },
    { inset: [0, 0], mapping: "band" }
  );
  const top = Math.max(0.1, ...values.map((v) => Math.abs(v ?? 0)));

  return (
    <div className="mt-3 border-t border-hairline pt-2">
      <ChartFrame
        cursor={cursor}
        title="ROOM GAP BY ROUND"
        categories={Array.from({ length: rounds + 1 }, (_, r) => (r === 0 ? "R0" : `R${r}`))}
        series={[
          {
            key: "gap",
            label: "mean |said − thought|",
            color: "#f0605f",
            values,
            format: fmt2,
          },
        ]}
        footnote={`mean absolute gap across the room · scaled to ${top.toFixed(2)} · a rising line is a room learning what it is not allowed to say`}
        height={54}
      >
        <div className="flex h-full items-end gap-0.5">
          {values.map((v, r) => (
            <div key={r} className="flex-1">
              {v === null ? (
                <div
                  className="w-full border border-dashed border-hairline"
                  style={{ height: 6 }}
                  title="no gap measured at this round"
                />
              ) : (
                <div
                  className="w-full transition-opacity"
                  style={{
                    height: `${Math.max(3, (Math.abs(v) / top) * 48)}px`,
                    background: "#f0605f",
                    opacity: cursor.index === null ? 0.8 : cursor.isMarked(r) ? 1 : 0.25,
                  }}
                />
              )}
            </div>
          ))}
        </div>
      </ChartFrame>
    </div>
  );
}
