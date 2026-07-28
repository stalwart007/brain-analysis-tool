import type { RawEvent, RawEventType } from "./types";
import { createFeatureWorker, type FeatureWorkerLike } from "./worker";

const MOVE_SAMPLE_MS = 50; // downsample mousemove to <= 20 Hz
const SCROLL_SAMPLE_MS = 30;
const MAX_PENDING = 500; // hard cap: drop oldest if a flush stalls

export interface CaptureController {
  worker: FeatureWorkerLike;
  stop(): void;
}

/**
 * Wires DOM listeners -> rAF-batched postMessage into the feature worker.
 * All listeners are passive; per-event work is O(1) and does no layout reads.
 */
export function startCapture(zones: string[]): CaptureController {
  const worker = createFeatureWorker();
  const pending: RawEvent[] = [];
  let rafScheduled = false;
  let lastMove = 0;
  let lastScroll = 0;
  const zoneSet = new Set(zones);

  const drain = () => {
    rafScheduled = false;
    if (pending.length > 0) {
      worker.post({ kind: "events", events: pending.splice(0) });
    }
  };

  const push = (ev: RawEvent) => {
    pending.push(ev);
    if (pending.length > MAX_PENDING) pending.shift();
    if (!rafScheduled) {
      rafScheduled = true;
      requestAnimationFrame(drain);
    }
  };

  const resolveZone = (target: EventTarget | null): string | undefined => {
    const el = target instanceof Element ? target.closest("[data-cs]") : null;
    const zone = el?.getAttribute("data-cs") ?? undefined;
    // whitelist only — never attribute to zones the integrator didn't declare
    return zone && zoneSet.has(zone) ? zone : undefined;
  };

  const record = (type: RawEventType, e: Event) => {
    const me = e as MouseEvent;
    push({
      t: performance.now(),
      type,
      x: typeof me.clientX === "number" ? me.clientX : 0,
      y: window.scrollY,
      zone: resolveZone(e.target),
    });
  };

  const onScroll = (e: Event) => {
    const now = performance.now();
    if (now - lastScroll < SCROLL_SAMPLE_MS) return;
    lastScroll = now;
    record("scroll", e);
  };
  const onMove = (e: Event) => {
    const now = performance.now();
    if (now - lastMove < MOVE_SAMPLE_MS) return;
    lastMove = now;
    record("move", e);
  };
  const onClick = (e: Event) => record("click", e);

  const opts: AddEventListenerOptions = { passive: true, capture: true };
  window.addEventListener("scroll", onScroll, opts);
  window.addEventListener("mousemove", onMove, opts);
  window.addEventListener("click", onClick, opts);

  return {
    worker,
    stop() {
      window.removeEventListener("scroll", onScroll, opts);
      window.removeEventListener("mousemove", onMove, opts);
      window.removeEventListener("click", onClick, opts);
      worker.terminate();
    },
  };
}
