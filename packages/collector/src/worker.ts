import type { FeaturePayload, RawEvent } from "./types";

/**
 * The entire Layer-1 feature extractor runs inside this function, which is
 * stringified into a Blob so the SDK works from a single <script> tag with no
 * separate worker file to host. It must stay fully self-contained — no
 * references to module scope survive Function.prototype.toString().
 *
 * Note: pages with a strict CSP (worker-src 'self' without blob:) will refuse
 * blob workers; capture.ts falls back to main-thread extraction in that case.
 */
function workerMain() {
  type Ev = {
    t: number;
    type: string;
    x: number;
    y: number;
    zone?: string;
  };

  const HESITATION_MS = 1200; // mid-interaction pause threshold
  const RAGE_WINDOW_MS = 800;
  const RAGE_CLICKS = 3;

  let buffer: Ev[] = [];

  const mean = (a: number[]) =>
    a.length ? a.reduce((s, v) => s + v, 0) / a.length : 0;

  const variance = (a: number[]) => {
    if (a.length < 2) return 0;
    const m = mean(a);
    return mean(a.map((v) => (v - m) * (v - m)));
  };

  function compute(evts: Ev[]) {
    if (evts.length === 0) return null;

    const scrolls = evts.filter((e) => e.type === "scroll");
    const clicks = evts.filter((e) => e.type === "click");

    // scroll kinematics
    const velocities: number[] = [];
    let directionChanges = 0;
    let upTravel = 0;
    let downTravel = 0;
    let lastDir = 0;
    for (let i = 1; i < scrolls.length; i++) {
      const dt = scrolls[i].t - scrolls[i - 1].t;
      const dy = scrolls[i].y - scrolls[i - 1].y;
      if (dt > 0) velocities.push(Math.abs(dy) / dt);
      if (dy > 0) downTravel += dy;
      else upTravel += -dy;
      const dir = Math.sign(dy);
      if (dir !== 0 && lastDir !== 0 && dir !== lastDir) directionChanges++;
      if (dir !== 0) lastDir = dir;
    }
    // backtracking (re-reading / lost-user signal): upward travel share
    const totalTravel = upTravel + downTravel;
    const backtrackRatio = totalTravel > 0 ? upTravel / totalTravel : 0;

    // micro-hesitations: pauses mid-interaction
    let hesitations = 0;
    let hesitationMs = 0;
    for (let i = 1; i < evts.length; i++) {
      const gap = evts[i].t - evts[i - 1].t;
      if (gap > HESITATION_MS) {
        hesitations++;
        hesitationMs += gap;
      }
    }

    // rage-click bursts: >= RAGE_CLICKS clicks inside RAGE_WINDOW_MS
    let rage = 0;
    for (let i = RAGE_CLICKS - 1; i < clicks.length; i++) {
      if (clicks[i].t - clicks[i - RAGE_CLICKS + 1].t < RAGE_WINDOW_MS) rage++;
    }

    // zone attribution
    const dwell: Record<string, number> = {};
    for (let i = 1; i < evts.length; i++) {
      const z = evts[i - 1].zone;
      if (z) dwell[z] = (dwell[z] || 0) + (evts[i].t - evts[i - 1].t);
    }
    const zoneClicks: Record<string, number> = {};
    for (const c of clicks) {
      if (c.zone) zoneClicks[c.zone] = (zoneClicks[c.zone] || 0) + 1;
    }

    // ── cognitive-modeling series (numbers only, no content, no PII) ──
    // symbol alphabet mirrors the server: 0 scroll, 1 dwell, 2 click,
    // 3 hesitate, 4 rage, 5 backtrack, 6 zone-change
    const stream: number[][] = [];
    let lastZone: string | undefined;
    let lastDy = 0;
    for (let i = 0; i < evts.length; i++) {
      const e = evts[i];
      const gap = i > 0 ? e.t - evts[i - 1].t : 0;
      if (gap > HESITATION_MS) stream.push([evts[i - 1].t, 3]);
      else if (i > 0 && gap > 400 && evts[i - 1].type === "move")
        stream.push([evts[i - 1].t, 1]); // settled pointer = dwell
      if (e.zone && e.zone !== lastZone) {
        if (lastZone !== undefined) stream.push([e.t, 6]);
        lastZone = e.zone;
      }
      if (e.type === "scroll") {
        const dy = i > 0 ? e.y - evts[i - 1].y : 0;
        stream.push([e.t, dy < 0 && lastDy >= 0 ? 5 : 0]);
        if (dy !== 0) lastDy = dy;
      } else if (e.type === "click") {
        stream.push([e.t, 2]);
      }
    }
    // mark rage bursts at the burst end
    for (let i = RAGE_CLICKS - 1; i < clicks.length; i++) {
      if (clicks[i].t - clicks[i - RAGE_CLICKS + 1].t < RAGE_WINDOW_MS)
        stream.push([clicks[i].t, 4]);
    }
    stream.sort((a, b) => a[0] - b[0]);

    // velocity series for spectral / entropy analysis (irregular sampling ok)
    const velSeries: number[][] = [];
    for (let i = 1; i < scrolls.length; i++) {
      const dt = scrolls[i].t - scrolls[i - 1].t;
      if (dt > 0)
        velSeries.push([scrolls[i].t, Math.abs(scrolls[i].y - scrolls[i - 1].y) / dt]);
    }

    // decision latencies: quiet-period → click intervals (visual search time)
    const latencies: number[] = [];
    for (const c of clicks) {
      let prev = -1;
      for (const e of evts) {
        if (e.t >= c.t) break;
        if (e.type !== "click") prev = e.t;
      }
      if (prev > 0 && c.t - prev > 80 && c.t - prev < 10000) latencies.push(c.t - prev);
    }

    const CAP = 2000;
    return {
      segment_start: evts[0].t,
      segment_end: evts[evts.length - 1].t,
      event_count: evts.length,
      scroll_velocity_mean: mean(velocities),
      scroll_velocity_variance: variance(velocities),
      scroll_direction_changes: directionChanges,
      micro_hesitation_count: hesitations,
      hesitation_total_ms: hesitationMs,
      rage_click_bursts: rage,
      click_count: clicks.length,
      backtrack_ratio: backtrackRatio,
      zone_dwell_ms: dwell,
      zone_click_counts: zoneClicks,
      event_stream: stream.slice(0, CAP).map(([t, s]) => [Math.round(t), s]),
      velocity_series: velSeries.slice(0, CAP).map(([t, v]) => [
        Math.round(t),
        Math.round(v * 10000) / 10000,
      ]),
      decision_latencies_ms: latencies.slice(0, 200).map((x) => Math.round(x)),
    };
  }

  self.onmessage = (e: MessageEvent) => {
    const msg = e.data as { kind: "events" | "flush" | "drop"; events?: Ev[] };
    if (msg.kind === "events" && msg.events) {
      buffer.push(...msg.events);
    } else if (msg.kind === "drop") {
      buffer = [];
    } else if (msg.kind === "flush") {
      const features = compute(buffer);
      buffer = [];
      if (features) (self as unknown as Worker).postMessage(features);
    }
  };
}

export interface FeatureWorkerLike {
  post(msg: { kind: "events" | "flush" | "drop"; events?: RawEvent[] }): void;
  onFeatures(cb: (f: FeaturePayload) => void): void;
  terminate(): void;
}

/** Spin up the Blob worker; fall back to a synchronous main-thread shim if blocked (e.g. CSP). */
export function createFeatureWorker(): FeatureWorkerLike {
  try {
    const src = `(${workerMain.toString()})()`;
    const url = URL.createObjectURL(
      new Blob([src], { type: "application/javascript" })
    );
    const worker = new Worker(url);
    URL.revokeObjectURL(url);
    let handler: (f: FeaturePayload) => void = () => {};
    worker.onmessage = (e) => handler(e.data as FeaturePayload);
    return {
      post: (msg) => worker.postMessage(msg),
      onFeatures: (cb) => (handler = cb),
      terminate: () => worker.terminate(),
    };
  } catch {
    return createInlineShim();
  }
}

/** Main-thread fallback with the same message contract (used only when workers are unavailable). */
function createInlineShim(): FeatureWorkerLike {
  let handler: (f: FeaturePayload) => void = () => {};
  let buffer: RawEvent[] = [];
  // Re-run workerMain's logic by hosting it in a fake `self`
  const fakeSelf = {
    onmessage: null as null | ((e: { data: unknown }) => void),
    postMessage: (f: FeaturePayload) => handler(f),
  };
  const realSelf = globalThis as Record<string, unknown>;
  const prev = realSelf.self;
  realSelf.self = fakeSelf;
  try {
    workerMain();
  } finally {
    realSelf.self = prev;
  }
  return {
    post: (msg) => fakeSelf.onmessage?.({ data: msg }),
    onFeatures: (cb) => (handler = cb),
    terminate: () => {
      buffer = [];
      void buffer;
    },
  };
}
