"use client";

/**
 * The appraisal tensor, filling in.
 *
 * Clicking "Run neuro-impact study" starts a genuinely large computation and
 * showed nothing at all — the only panel of the six with a live stage that had
 * no visual. What is actually happening is that every twin experiences every
 * beat and rates it on nine appraisal dimensions, producing a
 * twins × beats × dimensions tensor; ISC, change-point detection, peak-end and
 * the retention hazard are then all reductions over that one object.
 *
 * So the tensor IS the visualization. Each arriving twin lands as a row of
 * cells, one per beat, coloured by that twin's attention at that moment. Within
 * a few seconds the grid shows the thing the numbers below will summarise:
 * vertical banding means the audience agrees about which beats grip (high ISC),
 * a noisy field means they do not, and a column going dark is the drop-off the
 * change-point detector will name.
 *
 * Drawn on canvas rather than as SVG because at 20 personas × 20 twins × 8
 * beats this is 3,200 cells re-rendering on every arrival, and that many DOM
 * nodes re-diffed per event is exactly the re-render storm the audit found
 * elsewhere in this app.
 */

import { useAgentCanvas } from "./useAgentCanvas";

export interface TensorRow {
  /** one attention value per beat, in beat order */
  attention: number[];
}

/** Analysis passes that run once the tensor is complete. Shown as a pipeline
 *  so the wait after the last twin reads as work rather than as a hang. */
export interface StageState {
  id: string;
  label: string;
  done: boolean;
  active: boolean;
}

export default function TensorSim({
  rows,
  beats,
  expected,
  stages,
}: {
  rows: TensorRow[];
  beats: number;
  /** total twins dispatched, so the grid reserves its full height immediately
   *  and fills downward — a grid that grows changes size under the reader */
  expected: number;
  stages: StageState[];
}) {
  const ref = useAgentCanvas((ctx, w, h, t) => {
    ctx.clearRect(0, 0, w, h);
    if (beats < 1) return;

    const padL = 46;
    const padR = 14;
    const padT = 34;
    const stripH = stages.length ? 26 : 0;
    const gridH = Math.max(20, h - padT - stripH - 16);
    const gridW = w - padL - padR;

    const cols = beats;
    const rowsN = Math.max(expected, rows.length, 1);
    const cw = gridW / cols;
    const rh = Math.min(14, gridH / rowsN);

    // beat headers
    ctx.font = '9px ui-monospace, monospace';
    ctx.fillStyle = "rgba(168,163,163,0.75)";
    ctx.textAlign = "center";
    for (let c = 0; c < cols; c++) {
      ctx.fillText(`B${c + 1}`, padL + c * cw + cw / 2, padT - 10);
    }
    ctx.textAlign = "right";
    ctx.fillStyle = "rgba(111,108,108,0.9)";
    ctx.fillText("twins", padL - 8, padT + 8);

    // the empty lattice: shows the full shape of the work before it is done,
    // so progress reads as filling rather than as growing
    ctx.fillStyle = "rgba(223,217,217,0.035)";
    for (let r = 0; r < rowsN; r++) {
      for (let c = 0; c < cols; c++) {
        ctx.fillRect(padL + c * cw + 1, padT + r * rh + 1, cw - 2, Math.max(1, rh - 2));
      }
    }

    // arrived appraisals
    rows.forEach((row, r) => {
      // a brief flash on the newest row so an arrival is perceptible even when
      // several land in the same frame
      const age = rows.length - r;
      const flash = age <= 1 ? 0.5 + 0.5 * Math.sin(t * 9) : 0;
      for (let c = 0; c < cols; c++) {
        const a = Math.max(0, Math.min(1, row.attention[c] ?? 0));
        // attention drives both brightness and hue: low attention reads amber
        // (the drop-off the analysis is looking for), high reads as the region
        // accent, so a dark or warm column is legible at a glance
        const lift = 0.1 + a * 0.85;
        ctx.fillStyle =
          a < 0.3
            ? `rgba(242,173,31,${lift})`
            : `rgb(var(--region-rgb) / ${lift})`;
        ctx.fillRect(padL + c * cw + 1, padT + r * rh + 1, cw - 2, Math.max(1, rh - 2));
        if (flash > 0) {
          ctx.fillStyle = `rgba(255,255,255,${0.28 * flash})`;
          ctx.fillRect(padL + c * cw + 1, padT + r * rh + 1, cw - 2, Math.max(1, rh - 2));
        }
      }
    });

    // per-beat mean, drawn over the grid — this is the attention curve the
    // change-point detector actually consumes, forming in real time
    if (rows.length >= 2) {
      ctx.beginPath();
      for (let c = 0; c < cols; c++) {
        const vals = rows.map((r) => r.attention[c] ?? 0);
        const mean = vals.reduce((s, v) => s + v, 0) / vals.length;
        const x = padL + c * cw + cw / 2;
        const y = padT + gridH - mean * gridH * 0.92;
        c === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      }
      ctx.strokeStyle = "rgba(255,255,255,0.65)";
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }

    // analysis pipeline
    if (stages.length) {
      const y = padT + gridH + 14;
      const sw = gridW / stages.length;
      stages.forEach((s, i) => {
        const x = padL + i * sw;
        const on = s.done || s.active;
        ctx.fillStyle = s.done
          ? "rgba(79,180,119,0.9)"
          : s.active
            ? `rgb(var(--region-rgb) / ${0.55 + 0.45 * Math.sin(t * 5)})`
            : "rgba(223,217,217,0.12)";
        ctx.fillRect(x, y, Math.max(2, sw - 6), 3);
        ctx.font = '8px ui-monospace, monospace';
        ctx.fillStyle = on ? "rgba(223,217,217,0.85)" : "rgba(111,108,108,0.7)";
        ctx.textAlign = "left";
        ctx.fillText(s.label, x, y + 14);
      });
    }
  }, `${rows.length}:${beats}:${stages.filter((s) => s.done).length}:${stages.filter((s) => s.active).length}`);

  return <canvas ref={ref} className="h-full w-full" />;
}
