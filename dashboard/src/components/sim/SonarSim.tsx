"use client";

/**
 * OBJECTION SONAR → SEMANTIC MAP. While the scan runs, every real streamed
 * twin verdict paints a live ping on the ring (red = dealbreaker). When the
 * result lands, themes take up their kernel-PCA coordinates — from that moment
 * blip POSITIONS ARE SEMANTIC: nearby blips are genuinely similar complaints,
 * and the sweep keeps re-lighting them.
 */

import { useRef } from "react";
import { useAgentCanvas } from "./useAgentCanvas";

export interface SonarPing {
  dealbreaker: boolean;
  sentiment: string;
}

export interface SonarBlip {
  count: number;
  dealbreaker: boolean;
  x?: number; // kernel-PCA coords in [-1, 1] — semantic position
  y?: number;
  label?: string;
}

interface LiveDot {
  angle: number;
  radius: number;
  born: number;
  dealbreaker: boolean;
  negative: boolean;
}

interface ThemeDot {
  x: number;
  y: number;
  tx: number;
  ty: number;
  size: number;
  dealbreaker: boolean;
  label?: string;
  lit: number;
}

export default function SonarSim({
  active,
  pings,
  blips,
}: {
  active: boolean;
  pings?: SonarPing[]; // growing array — one per streamed twin verdict
  blips?: SonarBlip[]; // final themes, with PCA coords when semantic
}) {
  const sweep = useRef(0);
  const consumed = useRef(0);
  const live = useRef<LiveDot[]>([]);
  const themes = useRef<ThemeDot[]>([]);
  const key = useRef("");

  const ref = useAgentCanvas((ctx, w, h, t, dt) => {
    ctx.clearRect(0, 0, w, h);
    const cx = w / 2,
      cy = h / 2;
    const R = Math.min(w, h) / 2 - 18;

    // consume NEW streamed pings → transient dots at pseudo-random positions
    if (pings) {
      while (consumed.current < pings.length) {
        const p = pings[consumed.current++];
        const i = consumed.current;
        live.current.push({
          angle: (i * 2.399963) % (Math.PI * 2), // golden-angle spiral
          radius: 0.3 + ((i * 41) % 60) / 100,
          born: t,
          dealbreaker: p.dealbreaker,
          negative: p.sentiment === "negative",
        });
      }
    }

    // (re)build theme layout when the final result changes
    const k = (blips ?? []).map((b) => `${b.count}:${b.dealbreaker}:${b.x}`).join("|");
    if (k !== key.current) {
      key.current = k;
      const src = blips ?? [];
      const maxC = Math.max(1, ...src.map((b) => b.count));
      themes.current = src.map((b, i) => {
        const hasCoords = typeof b.x === "number" && typeof b.y === "number";
        const angle = (i / Math.max(1, src.length)) * Math.PI * 2 + i * 0.7;
        const fallR = 0.35 + ((i * 37) % 55) / 100;
        return {
          x: cx,
          y: cy,
          tx: hasCoords ? cx + (b.x as number) * R * 0.72 : cx + Math.cos(angle) * R * fallR,
          ty: hasCoords ? cy + (b.y as number) * R * 0.72 : cy + Math.sin(angle) * R * fallR,
          size: 0.3 + (b.count / maxC) * 0.7,
          dealbreaker: b.dealbreaker,
          label: b.label,
          lit: 1,
        };
      });
    }

    // rings + crosshair
    ctx.strokeStyle = "rgba(120,165,230,0.14)";
    for (let r = 1; r <= 3; r++) {
      ctx.beginPath();
      ctx.arc(cx, cy, (R * r) / 3, 0, Math.PI * 2);
      ctx.stroke();
    }
    ctx.beginPath();
    ctx.moveTo(cx - R, cy);
    ctx.lineTo(cx + R, cy);
    ctx.moveTo(cx, cy - R);
    ctx.lineTo(cx, cy + R);
    ctx.stroke();
    // axis hints once the map is semantic
    if (themes.current.length && blips?.some((b) => typeof b.x === "number")) {
      ctx.fillStyle = "rgba(95,107,125,0.75)";
      ctx.font = "8px ui-monospace, monospace";
      ctx.textAlign = "left";
      ctx.fillText("PC1 →", cx + R - 34, cy - 5);
      ctx.save();
      ctx.translate(cx + 5, cy - R + 30);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText("PC2 →", 0, 0);
      ctx.restore();
    }

    // sweep
    sweep.current += dt * (active ? 2.4 : 0.6);
    const ang = sweep.current % (Math.PI * 2);
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.arc(cx, cy, R, ang - 0.5, ang);
    ctx.closePath();
    ctx.fillStyle = "rgba(90,185,255,0.10)";
    ctx.fill();
    ctx.strokeStyle = "rgba(120,195,255,0.55)";
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(cx + Math.cos(ang) * R, cy + Math.sin(ang) * R);
    ctx.stroke();

    // live pings: expanding ripple that fades over ~3s
    live.current = live.current.filter((p) => t - p.born < 3);
    for (const p of live.current) {
      const age = t - p.born;
      const alpha = Math.max(0, 1 - age / 3);
      const px = cx + Math.cos(p.angle) * R * p.radius;
      const py = cy + Math.sin(p.angle) * R * p.radius;
      const base = p.dealbreaker ? "240,96,95" : p.negative ? "242,173,31" : "120,195,255";
      ctx.fillStyle = `rgba(${base},${alpha * 0.9})`;
      ctx.beginPath();
      ctx.arc(px, py, 2.5, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = `rgba(${base},${alpha * 0.4})`;
      ctx.beginPath();
      ctx.arc(px, py, 3 + age * 9, 0, Math.PI * 2);
      ctx.stroke();
    }

    // final themes: glide to semantic coords, re-lit by the sweep
    for (const b of themes.current) {
      b.x += (b.tx - b.x) * Math.min(1, dt * 4);
      b.y += (b.ty - b.y) * Math.min(1, dt * 4);
      const bAng = Math.atan2(b.y - cy, b.x - cx);
      const da = Math.abs(((bAng - ang + Math.PI * 3) % (Math.PI * 2)) - Math.PI);
      if (da > Math.PI - 0.25) b.lit = 1;
      b.lit = Math.max(0.25, b.lit - dt * 0.4);
      const rad = 2.5 + b.size * 7;
      const base = b.dealbreaker ? "240,96,95" : "120,195,255";
      ctx.fillStyle = `rgba(${base},${0.25 + b.lit * 0.65})`;
      ctx.beginPath();
      ctx.arc(b.x, b.y, rad, 0, Math.PI * 2);
      ctx.fill();
      if (b.dealbreaker) {
        ctx.strokeStyle = `rgba(240,96,95,${0.3 + b.lit * 0.5})`;
        ctx.beginPath();
        ctx.arc(b.x, b.y, rad + 3 + Math.sin(t * 4) * 1.5, 0, Math.PI * 2);
        ctx.stroke();
      }
      if (b.label && b.lit > 0.6) {
        ctx.fillStyle = `rgba(168,204,255,${(b.lit - 0.6) * 2})`;
        ctx.font = "8px ui-monospace, monospace";
        ctx.textAlign = "center";
        const short = b.label.length > 26 ? b.label.slice(0, 25) + "…" : b.label;
        ctx.fillText(short, b.x, b.y - rad - 5);
      }
    }

    // center
    ctx.fillStyle = "rgba(168,204,255,0.9)";
    ctx.beginPath();
    ctx.arc(cx, cy, 3, 0, Math.PI * 2);
    ctx.fill();
  });

  return <canvas ref={ref} className="h-full w-full" />;
}
