"use client";

/**
 * PRICE DEMAND FIELD — DRIVEN BY REAL BUYER CURVES. Each streamed agent rates
 * every price; its dots drop into each column at the height of its stated
 * intent, and the live demand curve re-draws from the *actual* running means
 * the backend sends. The revenue-optimal column glows when the result lands.
 */

import { useRef } from "react";
import { useAgentCanvas } from "./useAgentCanvas";

interface Col {
  price: number;
  demand: number;
  best: boolean;
}

export default function DemandSim({
  prices,
  curves,
  runningDemand,
  cols,
}: {
  prices: number[];
  curves?: number[][]; // growing array — one intents-per-price row per agent
  runningDemand?: number[]; // live mean demand per price from the stream
  cols?: Col[]; // final result
}) {
  const C: Col[] =
    cols ??
    prices.map((price, i) => ({
      price,
      demand: runningDemand?.[i] ?? 0,
      best: false,
    }));
  const dots = useRef<{ x: number; y: number; targetY: number; vy: number; col: number }[]>([]);
  const consumed = useRef(0);

  const ref = useAgentCanvas((ctx, w, h, t, dt) => {
    const n = Math.max(1, C.length);
    ctx.clearRect(0, 0, w, h);
    const pad = 28;
    const colW = (w - pad * 2) / n;
    const floor = h - 26;
    const yFor = (d: number) => floor - d * (floor - 24);

    // consume NEW streamed curves: each agent drops one dot per column at the
    // height of ITS OWN intent for that price
    if (curves) {
      while (consumed.current < curves.length) {
        const curve = curves[consumed.current++];
        for (let i = 0; i < Math.min(n, curve.length); i++) {
          dots.current.push({
            x: pad + i * colW + colW / 2 + (Math.random() - 0.5) * (colW - 18),
            y: -6 - Math.random() * 20,
            targetY: yFor(Math.max(0, Math.min(1, curve[i]))),
            vy: 40,
            col: i,
          });
        }
      }
    }

    // columns + mean-demand markers + labels
    const tops: number[] = [];
    for (let i = 0; i < n; i++) {
      const x = pad + i * colW;
      const top = yFor(C[i].demand);
      tops.push(top);
      ctx.fillStyle = C[i].best ? "rgba(22,208,22,0.08)" : "rgba(90,185,255,0.045)";
      ctx.fillRect(x + 5, 12, colW - 10, floor - 12);
      ctx.strokeStyle = C[i].best ? "rgba(22,208,22,0.45)" : "rgba(120,165,230,0.14)";
      ctx.strokeRect(x + 5, 12, colW - 10, floor - 12);
      // mean marker
      ctx.strokeStyle = C[i].best ? "rgba(22,208,22,0.8)" : "rgba(120,195,255,0.6)";
      ctx.beginPath();
      ctx.moveTo(x + 8, top);
      ctx.lineTo(x + colW - 8, top);
      ctx.stroke();
      ctx.fillStyle = C[i].best ? "#16d016" : "#898781";
      ctx.font = "10px ui-monospace, monospace";
      ctx.textAlign = "center";
      ctx.fillText(`$${C[i].price}`, x + colW / 2, h - 8);
    }

    // live demand curve across the mean markers
    ctx.strokeStyle = "rgba(120,195,255,0.6)";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const cx = pad + i * colW + colW / 2;
      if (i === 0) ctx.moveTo(cx, tops[i]);
      else ctx.lineTo(cx, tops[i]);
    }
    ctx.stroke();
    ctx.lineWidth = 1;

    // dots fall to their own intent height and hover there
    for (const d of dots.current) {
      const dy = d.targetY - d.y;
      d.vy += dy * 6 * dt;
      d.vy *= 0.86;
      d.y += d.vy * dt * 8;
      ctx.fillStyle = C[d.col].best ? "rgba(22,208,22,0.85)" : "rgba(120,195,255,0.7)";
      ctx.beginPath();
      ctx.arc(d.x, d.y, 2.2, 0, Math.PI * 2);
      ctx.fill();
    }
    if (dots.current.length > 600) dots.current.splice(0, 120);
  }, `${curves?.length ?? 0}:${cols?.length ?? 0}`);

  return <canvas ref={ref} className="h-full w-full" />;
}
