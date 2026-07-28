/**
 * THE PHYSIOLOGICAL CLOCK — one signal source for the whole interface.
 *
 * Why a singleton rather than per-component timers: an interface animated by
 * a dozen independent `animation-delay`s reads as *noise*. An interface where
 * every pulse — status dots, membrane glow, the cortex's breathing, the EEG
 * trace — derives from one shared waveform reads as a single living organism.
 * That coherence is the entire effect, and it is only achievable from a
 * common clock.
 *
 * The loop publishes three CSS custom properties on <html>:
 *
 *   --eeg    −1‥1   slow cortical drift + alpha envelope (glow, opacity)
 *   --pulse   0‥1   cardiac envelope with a sharp systolic upstroke (dots)
 *   --theta  −1‥1   6 Hz theta rhythm (fine detail; use sparingly)
 *
 * Cost control, because this writes to the root element:
 *   · CSS variables are flushed at ~30 Hz, not per frame.
 *   · A value is only written when it moved past a visible threshold.
 *   · The loop stops entirely when no one is subscribed, when the tab is
 *     hidden, or when the user asks for reduced motion (in which case the
 *     variables are parked at neutral rest values, so nothing depending on
 *     them collapses to zero).
 */

export interface Vitals {
  /** seconds since the clock started */
  t: number;
  /** −1‥1 — slow cortical drift, band-limited */
  eeg: number;
  /** 0‥1 — cardiac envelope, sharp upstroke then decay */
  pulse: number;
  /** −1‥1 — 6 Hz theta */
  theta: number;
  /** true when an interictal-style spike is firing this frame */
  spike: boolean;
}

type Listener = (v: Vitals) => void;

const listeners = new Set<Listener>();
let raf = 0;
let started = 0;
/** last values flushed to CSS, so we can skip no-op style writes */
const flushed = { eeg: NaN, pulse: NaN, theta: NaN };
let lastFlush = 0;

const REST: Vitals = { t: 0, eeg: 0, pulse: 0.5, theta: 0, spike: false };

function reducedMotion(): boolean {
  return (
    typeof window !== "undefined" &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches
  );
}

/**
 * The waveform itself.
 *
 * Real cortical EEG is roughly 1/f — power falls off with frequency — so the
 * sum below weights slow bands far more heavily than fast ones. That is what
 * keeps the trace organic instead of buzzing: delta and theta carry the shape,
 * alpha and beta only texture it.
 */
export function sample(t: number): Vitals {
  const delta = 0.8 * Math.sin(2 * Math.PI * 1.6 * t);
  const theta = Math.sin(2 * Math.PI * 6.0 * t + 1.3);
  const alpha = 0.34 * Math.sin(2 * Math.PI * 10.5 * t + 0.7);
  const beta = 0.16 * Math.sin(2 * Math.PI * 21.0 * t + 2.1);

  // an occasional sharp transient, the way a real trace throws one
  const gate = Math.sin(2 * Math.PI * 0.21 * t);
  const spike = gate > 0.9965;
  const transient = spike ? 1.6 : 0;

  const raw = delta + 0.55 * theta + alpha + beta + transient;
  const eeg = Math.max(-1, Math.min(1, raw / 2.2));

  // cardiac: a fast upstroke and a slow decay, not a sinusoid — the asymmetry
  // is what makes a pulsing dot read as a heartbeat rather than a throb.
  const hr = 1.05; // Hz ≈ 63 bpm, resting
  const phase = (t * hr) % 1;
  const pulse =
    phase < 0.12
      ? phase / 0.12
      : Math.exp(-(phase - 0.12) * 4.2) * 0.92 + 0.04;

  return { t, eeg, pulse, theta, spike };
}

function flush(v: Vitals, now: number) {
  // 30 Hz is indistinguishable from 60 for a glow, and halves the number of
  // root-element style recalculations this loop forces.
  if (now - lastFlush < 33) return;
  lastFlush = now;

  const root = document.documentElement;
  const eeg = Math.round(v.eeg * 1000) / 1000;
  const pulse = Math.round(v.pulse * 1000) / 1000;
  const theta = Math.round(v.theta * 1000) / 1000;

  if (Math.abs(eeg - flushed.eeg) > 0.004) {
    root.style.setProperty("--eeg", String(eeg));
    flushed.eeg = eeg;
  }
  if (Math.abs(pulse - flushed.pulse) > 0.004) {
    root.style.setProperty("--pulse", String(pulse));
    flushed.pulse = pulse;
  }
  if (Math.abs(theta - flushed.theta) > 0.004) {
    root.style.setProperty("--theta", String(theta));
    flushed.theta = theta;
  }
}

function tick(now: number) {
  raf = requestAnimationFrame(tick);
  const v = sample((now - started) / 1000);
  flush(v, now);
  for (const fn of listeners) fn(v);
}

function shouldRun(): boolean {
  return (
    listeners.size > 0 &&
    typeof document !== "undefined" &&
    document.visibilityState === "visible" &&
    !reducedMotion()
  );
}

function park() {
  // Park at rest rather than at zero: a component that scales a glow by
  // --pulse would otherwise vanish for reduced-motion users.
  const root = document.documentElement;
  root.style.setProperty("--eeg", "0");
  root.style.setProperty("--pulse", "0.5");
  root.style.setProperty("--theta", "0");
  flushed.eeg = 0;
  flushed.pulse = 0.5;
  flushed.theta = 0;
}

function sync() {
  if (shouldRun()) {
    if (!raf) {
      if (!started) started = performance.now();
      raf = requestAnimationFrame(tick);
    }
  } else if (raf) {
    cancelAnimationFrame(raf);
    raf = 0;
    park();
  } else if (listeners.size > 0) {
    park();
  }
}

let wired = false;
function wire() {
  if (wired || typeof document === "undefined") return;
  wired = true;
  document.addEventListener("visibilitychange", sync);
  window.matchMedia("(prefers-reduced-motion: reduce)").addEventListener("change", sync);
}

/**
 * Subscribe to the clock. Returns an unsubscribe function.
 *
 * Pass no callback to simply keep the CSS variables flowing — that is what
 * <VitalsClock/> does; it needs the loop running but doesn't read the values
 * in JavaScript.
 */
export function subscribeVitals(fn?: Listener): () => void {
  wire();
  const listener: Listener = fn ?? (() => {});
  listeners.add(listener);
  sync();
  return () => {
    listeners.delete(listener);
    sync();
  };
}
