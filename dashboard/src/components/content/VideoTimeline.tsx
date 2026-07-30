"use client";

/**
 * THE FRAMES, UNDER THE CURVE, ON A REAL CLOCK.
 *
 * Every other view in this study indexes beats by position — B1, B2, B3. That
 * is the correct axis for a script or a deck, where a beat has an order and no
 * duration. It is the wrong one for video, and quietly so: "attention drops at
 * B4" is a fact about a chart, while "attention drops at 1:24, on this frame"
 * is a fact about a video, and only the second one can be acted on by the
 * person who has to go and re-cut it.
 *
 * The information to do this was already present and unused. `timestamps_ms`
 * comes back on every temporal study, the keyframes are sitting in the asset
 * that was submitted, and the chapter titles come back on the manifest. All
 * three were being reduced to an integer index before anything was drawn.
 *
 * SPACING IS PROPORTIONAL TO TIME, not to beat count. A filmstrip laid out at
 * even intervals draws a fifteen-second gap and a two-second gap identically,
 * which invents a pace the video does not have — and pace is precisely what a
 * retention reader is looking for.
 */

import { useMemo } from "react";
import type { ChartCursor } from "@/components/charts/cursor";
import { clock } from "./YouTubeIngest";

export interface TimelineFrame {
  t_ms: number;
  image_b64: string;
  media_type?: string;
}

export interface TimelineChapter {
  start_s: number;
  end_s: number;
  title: string;
}

export default function VideoTimeline({
  frames,
  timestampsMs,
  attention,
  segments,
  durationS,
  chapters,
  changePoints,
  cursor,
}: {
  frames: TimelineFrame[];
  /** Where each BEAT sits on the clock. Not the same list as `frames` in
   *  general — a beat can be dropped upstream when its frame failed to
   *  describe — so the two are matched by time rather than by index. */
  timestampsMs?: number[] | null;
  attention: number[];
  segments: string[];
  durationS: number;
  chapters?: TimelineChapter[];
  changePoints?: number[] | null;
  cursor: ChartCursor;
}) {
  const W = 720;
  const H = 116;
  const padX = 8;
  const railY = 74;

  const totalMs = Math.max(1, (durationS || 0) * 1000);

  /** Beats, placed on the clock, each carrying the nearest real frame.
   *
   *  Matched by TIME rather than by index. When a keyframe fails to be
   *  described the beat list is shorter than the frame list, and zipping them
   *  positionally silently shifts every later thumbnail onto the wrong moment —
   *  a filmstrip that looks completely plausible and is wrong from beat 4
   *  onward. */
  const placed = useMemo(() => {
    return segments.map((segment, i) => {
      const t = timestampsMs?.[i] ?? (totalMs * (i + 0.5)) / Math.max(1, segments.length);
      let best: TimelineFrame | null = null;
      let bestGap = Infinity;
      for (const f of frames) {
        const gap = Math.abs(f.t_ms - t);
        if (gap < bestGap) {
          bestGap = gap;
          best = f;
        }
      }
      return {
        i,
        t,
        segment,
        // A frame more than four seconds from its beat is not that beat's
        // picture. Showing it anyway would caption a moment with an image from
        // somewhere else in the video.
        frame: bestGap <= 4000 ? best : null,
      };
    });
  }, [segments, timestampsMs, frames, totalMs]);

  const x = (tMs: number) => padX + (tMs / totalMs) * (W - padX * 2);
  const y = (v: number) => railY - v * 52;

  return (
    <div className="rounded-xl border border-hairline bg-black/40 p-3">
      <div className="mb-2 flex items-baseline justify-between">
        <span className="hud-label">TIMELINE · {clock(durationS)}</span>
        <span className="font-mono text-[9px] text-muted">
          frames sampled from the video · attention over real time
        </span>
      </div>

      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="h-32 w-full focus-visible:outline-none"
        {...cursor.surfaceProps}
      >
        {/* Chapter bands, when the creator wrote them. The single most useful
            piece of context on this chart: a drop INSIDE one section is a
            pacing problem, and a drop at a boundary is a transition problem. */}
        {chapters?.map((c, i) => (
          <g key={i}>
            <rect
              x={x(c.start_s * 1000)}
              y={8}
              width={Math.max(1, x(c.end_s * 1000) - x(c.start_s * 1000))}
              height={railY - 8}
              fill={i % 2 ? "rgba(120,165,230,0.05)" : "transparent"}
            />
            <line
              x1={x(c.start_s * 1000)}
              y1={8}
              x2={x(c.start_s * 1000)}
              y2={railY}
              stroke="rgba(120,165,230,0.22)"
              strokeDasharray="2 2"
            />
            <text
              x={x(c.start_s * 1000) + 3}
              y={15}
              fontSize="6.5"
              fill="#8fa3c0"
              fontFamily="ui-monospace, monospace"
            >
              {c.title.slice(0, 22)}
            </text>
          </g>
        ))}

        {/* Regime shifts, on the same clock as everything else. */}
        {changePoints?.map((cp) => {
          const at = placed[cp];
          if (!at) return null;
          return (
            <line
              key={cp}
              x1={x(at.t)}
              y1={12}
              x2={x(at.t)}
              y2={railY}
              stroke="#f0605f"
              strokeWidth="1"
              strokeDasharray="3 3"
            />
          );
        })}

        <line x1={padX} y1={railY} x2={W - padX} y2={railY} stroke="rgba(120,165,230,0.18)" />

        {/* Attention, drawn against elapsed time rather than beat index — so a
            long flat stretch LOOKS long. */}
        {placed.length > 1 && (
          <polyline
            fill="none"
            stroke="#4fb6ff"
            strokeWidth="1.75"
            points={placed.map((p) => `${x(p.t)},${y(attention[p.i] ?? 0)}`).join(" ")}
          />
        )}

        {placed.map((p) => {
          const on = cursor.isMarked(p.i);
          const fx = x(p.t);
          const thumbW = 34;
          const thumbH = 19;
          return (
            <g key={p.i}>
              <line
                x1={fx}
                y1={y(attention[p.i] ?? 0)}
                x2={fx}
                y2={railY}
                stroke={on ? "rgb(var(--region-rgb))" : "rgba(120,165,230,0.25)"}
              />
              <circle
                cx={fx}
                cy={y(attention[p.i] ?? 0)}
                r={on ? 4 : 2.6}
                fill={on ? "rgb(var(--region-rgb))" : "#4fb6ff"}
              />
              {p.frame ? (
                <image
                  href={`data:${p.frame.media_type ?? "image/jpeg"};base64,${p.frame.image_b64}`}
                  x={Math.min(W - padX - thumbW, Math.max(padX, fx - thumbW / 2))}
                  y={railY + 6}
                  width={thumbW}
                  height={thumbH}
                  preserveAspectRatio="xMidYMid slice"
                  opacity={cursor.index === null ? 0.82 : on ? 1 : 0.3}
                />
              ) : (
                <rect
                  x={Math.min(W - padX - thumbW, Math.max(padX, fx - thumbW / 2))}
                  y={railY + 6}
                  width={thumbW}
                  height={thumbH}
                  fill="none"
                  stroke="rgba(223,217,217,0.18)"
                  strokeDasharray="2 2"
                />
              )}
              <rect
                x={Math.min(W - padX - thumbW, Math.max(padX, fx - thumbW / 2))}
                y={railY + 6}
                width={thumbW}
                height={thumbH}
                fill="none"
                stroke={on ? "rgb(var(--region-rgb))" : "rgba(223,217,217,0.14)"}
                strokeWidth={on ? 1.4 : 0.8}
              />
              <text
                x={fx}
                y={H - 2}
                fontSize="6.5"
                fill={on ? "#dfd9d9" : "#5f6b7d"}
                textAnchor="middle"
                fontFamily="ui-monospace, monospace"
              >
                {clock(p.t / 1000)}
              </text>
            </g>
          );
        })}
      </svg>

      {/* Reserved height — this line is prose and changes length on every hover. */}
      <p className="mt-1 min-h-[2.2rem] font-mono text-[9px] leading-relaxed text-muted">
        {cursor.index !== null && placed[cursor.index] ? (
          <span className="line-clamp-2 text-ink-2">
            <span className="text-accent">{clock(placed[cursor.index].t / 1000)}</span> ·
            attention {(attention[cursor.index] ?? 0).toFixed(2)} —{" "}
            {placed[cursor.index].segment.slice(0, 150)}
          </span>
        ) : (
          <span>
            Frames are spaced by elapsed time, not by beat number, so a long
            stretch looks long. Hover to read a moment; click to pin it across
            every chart.
          </span>
        )}
      </p>
    </div>
  );
}
