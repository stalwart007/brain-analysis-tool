"use client";

import { useEffect, useRef } from "react";

export type DrawFn = (ctx: CanvasRenderingContext2D, w: number, h: number, t: number, dt: number) => void;

/**
 * Minimal 2D-canvas animation host for the per-simulation visualizations.
 * Handles DPR, resize, rAF, and pausing when the tab is hidden / reduced-motion
 * (drawing at least one static frame either way). The draw callback is kept in a
 * ref so the loop is created once and never thrashes on re-render.
 *
 * `redrawKey` exists for interaction under reduced motion. With the animation
 * loop stopped after one frame — which is the correct behaviour for someone who
 * asked for no motion — a canvas that highlights whatever the chart cursor is
 * pointing at would simply never repaint, so the whole linked-highlighting layer
 * would be invisible to exactly the users who cannot rely on movement to find
 * things. Changing the key paints one more frame: a response, not an animation.
 */
export function useAgentCanvas(draw: DrawFn, redrawKey?: unknown) {
  const ref = useRef<HTMLCanvasElement>(null);
  const drawRef = useRef(draw);
  drawRef.current = draw;
  const paintOnce = useRef<(() => void) | null>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    let w = 0,
      h = 0,
      raf = 0,
      last = 0,
      running = true;

    const resize = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const r = canvas.getBoundingClientRect();
      w = r.width || 600;
      h = r.height || 300;
      canvas.width = w * dpr;
      canvas.height = h * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(canvas);

    const frame = (ts: number) => {
      if (!running) return;
      const dt = last ? Math.min(0.05, (ts - last) / 1000) : 0.016;
      last = ts;
      drawRef.current(ctx, w, h, ts / 1000, dt);
      if (!reduced) raf = requestAnimationFrame(frame);
    };
    raf = requestAnimationFrame(frame);
    // Only the reduced-motion path needs manual repaints; when the loop is
    // live it picks up the new draw closure on its next frame anyway.
    paintOnce.current = () => {
      if (!reduced) return;
      drawRef.current(ctx, w, h, performance.now() / 1000, 0);
    };

    const onVis = () => {
      running = document.visibilityState === "visible" && !reduced;
      if (running) {
        cancelAnimationFrame(raf);
        last = 0;
        raf = requestAnimationFrame(frame);
      }
    };
    document.addEventListener("visibilitychange", onVis);
    return () => {
      running = false;
      paintOnce.current = null;
      cancelAnimationFrame(raf);
      ro.disconnect();
      document.removeEventListener("visibilitychange", onVis);
    };
  }, []);

  useEffect(() => {
    paintOnce.current?.();
  }, [redrawKey]);

  return ref;
}
