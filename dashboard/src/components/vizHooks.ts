"use client";

import { RefObject, useEffect, useRef, useState } from "react";

/**
 * Liveness for a 3D canvas: true only when the element is on-screen, the tab
 * is visible, and the user hasn't asked for reduced motion. Consumers switch
 * the render loop between "always" (live) and "demand" + one static frame —
 * never a fully blank canvas.
 */
export function useCanvasLiveness(holder: RefObject<HTMLElement | null>): boolean {
  const [inView, setInView] = useState(true);
  const [tabVisible, setTabVisible] = useState(true);
  const [reduced, setReduced] = useState(false);

  useEffect(() => {
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    setReduced(mq.matches);
    const onMq = () => setReduced(mq.matches);
    mq.addEventListener("change", onMq);

    setTabVisible(document.visibilityState === "visible");
    const onVis = () => setTabVisible(document.visibilityState === "visible");
    document.addEventListener("visibilitychange", onVis);

    const io = new IntersectionObserver(([entry]) => setInView(entry.isIntersecting), {
      threshold: 0.05,
    });
    if (holder.current) io.observe(holder.current);

    return () => {
      mq.removeEventListener("change", onMq);
      document.removeEventListener("visibilitychange", onVis);
      io.disconnect();
    };
  }, [holder]);

  return inView && tabVisible && !reduced;
}

/**
 * Animated count-up for stat tiles. SSR-safe (initial render is the final
 * value) and hidden-tab-safe (snaps instantly when rAF wouldn't run).
 */
export function useCountUp(target: number, duration = 650): number {
  const [display, setDisplay] = useState(target);
  const prev = useRef(target);

  useEffect(() => {
    const from = prev.current;
    prev.current = target;
    if (from === target) return;
    if (document.visibilityState === "hidden") {
      setDisplay(target);
      return;
    }
    let raf = 0;
    const start = performance.now();
    const tick = (now: number) => {
      const t = Math.min(1, (now - start) / duration);
      const eased = 1 - Math.pow(1 - t, 3);
      setDisplay(Math.round(from + (target - from) * eased));
      if (t < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [target, duration]);

  return display;
}
