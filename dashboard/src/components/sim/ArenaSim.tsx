"use client";

/**
 * A/B ARENA — one glowing lane per variant, DRIVEN BY REAL STREAMED AGENTS.
 * Every `agent` SSE frame spawns one dot falling into its variant's lane
 * (green = converted, blue = otherwise); lane fill tracks the *actual* running
 * conversion rate. Under Thompson-sampling allocation you literally watch the
 * bandit steer the agent stream toward the winning lane.
 */

import { useRef } from "react";
import { useAgentCanvas } from "./useAgentCanvas";

export interface ArenaArrival {
  lane: number;
  convert: boolean;
}

interface Lane {
  name: string;
  intent: number; // final mean intent 0..1 (drawn as the dashed target line)
  winner: boolean;
}

export default function ArenaSim({
  variants,
  arrivals,
  lanes,
}: {
  variants: string[];
  arrivals?: ArenaArrival[]; // growing array — one entry per streamed agent
  lanes?: Lane[]; // present once the final result lands
}) {
  const L = lanes ?? variants.map((name) => ({ name, intent: 0, winner: false }));
  const agents = useRef<
    { x: number; y: number; vy: number; lane: number; convert: boolean; settled: boolean }[]
  >([]);
  const spawned = useRef(0);

  const ref = useAgentCanvas((ctx, w, h, t, dt) => {
    const n = Math.max(1, L.length);
    ctx.clearRect(0, 0, w, h);
    const pad = 24;
    const laneW = (w - pad * 2) / n;
    const floor = h - 30;

    // spawn one dot per NEW streamed arrival
    if (arrivals) {
      while (spawned.current < arrivals.length) {
        const a = arrivals[spawned.current++];
        const lane = Math.min(n - 1, Math.max(0, a.lane));
        agents.current.push({
          x: pad + lane * laneW + laneW / 2 + (Math.random() - 0.5) * (laneW - 22),
          y: -6,
          vy: 60 + Math.random() * 60,
          lane,
          convert: a.convert,
          settled: false,
        });
      }
    }

    // per-lane live tallies from the real stream
    const counts = new Array(n).fill(0);
    const converts = new Array(n).fill(0);
    if (arrivals) {
      for (const a of arrivals) {
        const lane = Math.min(n - 1, Math.max(0, a.lane));
        counts[lane]++;
        if (a.convert) converts[lane]++;
      }
    }

    // lane frames, live conversion fill, target line, labels
    for (let i = 0; i < n; i++) {
      const x = pad + i * laneW;
      ctx.strokeStyle = "rgba(120,165,230,0.14)";
      ctx.strokeRect(x + 6, 10, laneW - 12, floor - 10);

      const rate = counts[i] > 0 ? converts[i] / counts[i] : 0;
      const fillTop = floor - rate * (floor - 20);
      const grad = ctx.createLinearGradient(0, fillTop, 0, floor);
      grad.addColorStop(0, L[i].winner ? "rgba(22,208,22,0.28)" : "rgba(63,143,240,0.22)");
      grad.addColorStop(1, "rgba(0,0,0,0)");
      ctx.fillStyle = grad;
      ctx.fillRect(x + 6, fillTop, laneW - 12, floor - fillTop);

      if (L[i].intent > 0) {
        const target = floor - L[i].intent * (floor - 20);
        ctx.strokeStyle = L[i].winner ? "rgba(22,208,22,0.55)" : "rgba(90,185,255,0.35)";
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(x + 6, target);
        ctx.lineTo(x + laneW - 6, target);
        ctx.stroke();
        ctx.setLineDash([]);
      }

      ctx.fillStyle = L[i].winner ? "#16d016" : "#898781";
      ctx.font = "11px ui-monospace, monospace";
      ctx.textAlign = "center";
      const label = L[i].name.length > 14 ? L[i].name.slice(0, 13) + "…" : L[i].name;
      ctx.fillText(label, x + laneW / 2, h - 12);
      ctx.fillStyle = "#5f6b7d";
      ctx.font = "9px ui-monospace, monospace";
      ctx.fillText(
        counts[i] > 0 ? `${converts[i]}/${counts[i]}` : "—",
        x + laneW / 2,
        h - 2
      );
    }

    // integrate + draw falling agents; converted ones flare on landing
    for (const a of agents.current) {
      if (!a.settled) {
        a.vy += 160 * dt;
        a.y += a.vy * dt;
        const landY = floor - 4 - Math.random() * 2;
        if (a.y >= landY) {
          a.y = landY;
          a.vy = -a.vy * 0.18;
          if (Math.abs(a.vy) < 8) a.settled = true;
        }
      }
      ctx.fillStyle = a.convert ? "rgba(22,208,22,0.92)" : "rgba(120,195,255,0.8)";
      ctx.beginPath();
      ctx.arc(a.x, a.y, a.convert ? 2.8 : 2.2, 0, Math.PI * 2);
      ctx.fill();
      if (a.convert && !a.settled) {
        ctx.strokeStyle = "rgba(22,208,22,0.35)";
        ctx.beginPath();
        ctx.arc(a.x, a.y, 5 + Math.sin(t * 8) * 1.5, 0, Math.PI * 2);
        ctx.stroke();
      }
    }
    if (agents.current.length > 400) agents.current.splice(0, 80);
  });

  return <canvas ref={ref} className="h-full w-full" />;
}
