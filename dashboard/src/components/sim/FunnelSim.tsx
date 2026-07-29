"use client";

/**
 * FLOW CASCADE — DRIVEN BY REAL PER-STEP EVENTS. The backend streams every gate
 * decision (`step` frames: twin i reached gate g and continued / hesitated /
 * abandoned / converted), so each dot you see advancing or leaking red is an
 * actual agent decision, at the exact gate where it happened.
 */

import { useRef } from "react";
import { useAgentCanvas } from "./useAgentCanvas";

export interface GateEvent {
  gate: number; // step index the decision happened at
  action: string; // continue | hesitate | abandon | convert
}

export default function FunnelSim({
  steps,
  events,
}: {
  steps: string[];
  events?: GateEvent[]; // growing array of real streamed gate decisions
}) {
  interface Dot {
    x: number;
    y: number;
    fromX: number;
    toX: number;
    t: number; // 0..1 travel progress
    action: string;
    leakVy: number;
    alpha: number;
  }
  const dots = useRef<Dot[]>([]);
  const consumed = useRef(0);
  // live tallies per gate
  const tallies = useRef<{ passed: number[]; dropped: number[] }>({ passed: [], dropped: [] });

  const ref = useAgentCanvas((ctx, w, h, t, dt) => {
    const n = Math.max(1, steps.length);
    if (tallies.current.passed.length !== n) {
      tallies.current = { passed: new Array(n).fill(0), dropped: new Array(n).fill(0) };
    }
    ctx.clearRect(0, 0, w, h);
    const padL = 26,
      padR = 26;
    const laneY = h * 0.4;
    const gateX = (i: number) => padL + ((i + 1) / (n + 1)) * (w - padL - padR);

    // consume NEW streamed gate decisions → spawn a traveling dot each
    if (events) {
      while (consumed.current < events.length) {
        const e = events[consumed.current++];
        const g = Math.min(n - 1, Math.max(0, e.gate));
        if (e.action === "abandon") tallies.current.dropped[g]++;
        else tallies.current.passed[g]++;
        dots.current.push({
          x: g === 0 ? padL : gateX(g - 1),
          fromX: g === 0 ? padL : gateX(g - 1),
          toX: gateX(g),
          y: laneY + (Math.random() - 0.5) * 14,
          t: 0,
          action: e.action,
          leakVy: 0,
          alpha: 1,
        });
      }
    }

    // gates: posts sized by live hazard (dropped / entered at that gate)
    for (let i = 0; i < n; i++) {
      const x = gateX(i);
      const entered = tallies.current.passed[i] + tallies.current.dropped[i];
      const hazard = entered > 0 ? tallies.current.dropped[i] / entered : 0;
      ctx.strokeStyle = `rgba(${hazard > 0.3 ? "240,96,95" : "120,165,230"},${0.25 + hazard * 0.5})`;
      ctx.lineWidth = 1 + hazard * 2;
      ctx.beginPath();
      ctx.moveTo(x, laneY - 36);
      ctx.lineTo(x, laneY + 36);
      ctx.stroke();
      ctx.lineWidth = 1;
      ctx.fillStyle = "#7d7b76";
      ctx.font = "9px ui-monospace, monospace";
      ctx.textAlign = "center";
      ctx.fillText(`${i + 1}`, x, laneY + 52);
      if (entered > 0) {
        ctx.fillStyle = hazard > 0.3 ? "#f0605f" : "#5f6b7d";
        ctx.fillText(`${(hazard * 100).toFixed(0)}%`, x, laneY - 44);
      }
    }
    // rail
    ctx.strokeStyle = "rgba(90,185,255,0.16)";
    ctx.beginPath();
    ctx.moveTo(padL, laneY);
    ctx.lineTo(w - padR, laneY);
    ctx.stroke();

    // animate dots
    const alive: Dot[] = [];
    for (const d of dots.current) {
      if (d.action === "abandon" && d.t >= 1) {
        // leak downward and fade
        d.leakVy += 150 * dt;
        d.y += d.leakVy * dt;
        d.alpha -= dt * 0.8;
        if (d.alpha > 0 && d.y < h + 8) alive.push(d);
      } else {
        d.t = Math.min(1, d.t + dt * 1.6);
        d.x = d.fromX + (d.toX - d.fromX) * d.t;
        if (d.t < 1 || d.action === "abandon") {
          alive.push(d);
        } else if (d.action === "convert") {
          // burst then vanish
          d.alpha -= dt * 1.6;
          if (d.alpha > 0) alive.push(d);
        }
        // continue/hesitate dots vanish on arrival — the next gate event re-spawns them
      }
      const color =
        d.action === "abandon"
          ? `rgba(240,96,95,${Math.max(0, d.alpha)})`
          : d.action === "convert"
            ? `rgba(22,208,22,${Math.max(0, d.alpha)})`
            : d.action === "hesitate"
              ? "rgba(242,173,31,0.9)"
              : "rgba(120,195,255,0.9)";
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(d.x, d.y, d.action === "convert" ? 3 : 2.3, 0, Math.PI * 2);
      ctx.fill();
    }
    dots.current = alive;
    if (dots.current.length > 300) dots.current.splice(0, 60);
  }, `${events?.length ?? 0}:${steps.length}`);

  return <canvas ref={ref} className="h-full w-full" />;
}
