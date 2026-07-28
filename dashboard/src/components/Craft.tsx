"use client";

/**
 * CRAFT — the interaction layer that separates a competent instrument from a
 * built one.
 *
 * Three primitives, each doing real work rather than decorating:
 *
 *   <Probe/>        a cursor that behaves like a recording electrode: it is
 *                   magnetically attracted to whatever it can act on, and it
 *                   drags a decaying membrane-potential trace behind it.
 *   <ScrollWarp/>   couples scroll VELOCITY to a shear on the content layer.
 *                   Motion you feel rather than see; it is what makes a page
 *                   read as having mass.
 *   <Kinetic/>      per-character reveal, staggered along the string, with the
 *                   characters arriving out of depth.
 *
 * All three degrade to nothing under prefers-reduced-motion, and all three run
 * off a single shared rAF rather than one per component.
 */

import {
  CSSProperties,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

/* ── shared frame loop ──────────────────────────────────────────────────
   One loop, many subscribers. Three components each running their own rAF is
   three style recalcs per frame; this is one. */

type Frame = (dt: number) => void;
const frameSubs = new Set<Frame>();
let frameRaf = 0;
let lastT = 0;

function frameTick(now: number) {
  frameRaf = requestAnimationFrame(frameTick);
  const dt = Math.min(0.05, (now - lastT) / 1000 || 0.016);
  lastT = now;
  for (const fn of frameSubs) fn(dt);
}

function onFrame(fn: Frame): () => void {
  frameSubs.add(fn);
  if (!frameRaf) {
    lastT = performance.now();
    frameRaf = requestAnimationFrame(frameTick);
  }
  return () => {
    frameSubs.delete(fn);
    if (frameSubs.size === 0 && frameRaf) {
      cancelAnimationFrame(frameRaf);
      frameRaf = 0;
    }
  };
}

function prefersReduced(): boolean {
  return (
    typeof window !== "undefined" &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches
  );
}

/* ═══════════════════════════════════════════════════════════════════════
   THE PROBE — cursor as recording electrode
   ═══════════════════════════════════════════════════════════════════════ */

const TRAIL = 18;

export function Probe() {
  const dot = useRef<HTMLDivElement>(null);
  const ring = useRef<HTMLDivElement>(null);
  const svg = useRef<SVGPolylineElement>(null);
  const [enabled, setEnabled] = useState(false);

  useEffect(() => {
    // Pointer-coarse devices have no cursor to augment, and a fake one is
    // actively worse than none on touch.
    const fine = window.matchMedia("(pointer: fine)").matches;
    if (!fine || prefersReduced()) return;
    setEnabled(true);

    const target = { x: innerWidth / 2, y: innerHeight / 2 };
    const pos = { x: target.x, y: target.y };
    const ringPos = { x: target.x, y: target.y };
    let scale = 1;
    let wantScale = 1;
    let magnet: DOMRect | null = null;
    const history: { x: number; y: number }[] = [];

    const onMove = (e: PointerEvent) => {
      target.x = e.clientX;
      target.y = e.clientY;

      // What is under the pointer decides the probe's state. Reading it here
      // (rather than binding listeners to every interactive element) means
      // dynamically-rendered panels are covered for free.
      const el = document.elementFromPoint(e.clientX, e.clientY) as HTMLElement | null;
      const hit = el?.closest(
        "a,button,input,textarea,select,[role=button],.probe,summary"
      ) as HTMLElement | null;
      if (hit) {
        magnet = hit.getBoundingClientRect();
        wantScale = 2.1;
      } else {
        magnet = null;
        wantScale = 1;
      }
    };

    const onDown = () => (wantScale = 0.7);
    const onUp = () => (wantScale = magnet ? 2.1 : 1);

    window.addEventListener("pointermove", onMove, { passive: true });
    window.addEventListener("pointerdown", onDown, { passive: true });
    window.addEventListener("pointerup", onUp, { passive: true });

    const stop = onFrame((dt) => {
      // the dot tracks precisely…
      pos.x += (target.x - pos.x) * Math.min(1, dt * 34);
      pos.y += (target.y - pos.y) * Math.min(1, dt * 34);

      // …the ring lags, and is pulled toward the centre of whatever the probe
      // can act on. That lag IS the affordance: the ring snapping to a button
      // reads as contact before you click.
      let rx = target.x;
      let ry = target.y;
      if (magnet) {
        const cx = magnet.left + magnet.width / 2;
        const cy = magnet.top + magnet.height / 2;
        rx = target.x + (cx - target.x) * 0.34;
        ry = target.y + (cy - target.y) * 0.34;
      }
      ringPos.x += (rx - ringPos.x) * Math.min(1, dt * 12);
      ringPos.y += (ry - ringPos.y) * Math.min(1, dt * 12);
      scale += (wantScale - scale) * Math.min(1, dt * 12);

      if (dot.current) {
        dot.current.style.transform = `translate3d(${pos.x}px,${pos.y}px,0) translate(-50%,-50%)`;
      }
      if (ring.current) {
        ring.current.style.transform = `translate3d(${ringPos.x}px,${ringPos.y}px,0) translate(-50%,-50%) scale(${scale.toFixed(3)})`;
      }

      // the recorded trace
      history.push({ x: pos.x, y: pos.y });
      if (history.length > TRAIL) history.shift();
      if (svg.current && history.length > 1) {
        svg.current.setAttribute(
          "points",
          history.map((p) => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" ")
        );
      }
    });

    return () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerdown", onDown);
      window.removeEventListener("pointerup", onUp);
      stop();
    };
  }, []);

  if (!enabled) return null;

  return (
    <div className="probe-layer" aria-hidden>
      <svg className="probe-trace">
        <polyline ref={svg} />
      </svg>
      <div ref={ring} className="probe-ring" />
      <div ref={dot} className="probe-dot" />
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════
   SCROLL WARP — velocity as shear
   ═══════════════════════════════════════════════════════════════════════ */

export function ScrollWarp() {
  useEffect(() => {
    if (prefersReduced()) return;
    const root = document.documentElement;
    let lastY = window.scrollY;
    let vel = 0;
    let shown = 0;

    const stop = onFrame((dt) => {
      const y = window.scrollY;
      const raw = (y - lastY) / Math.max(dt, 0.001); // px/s
      lastY = y;
      // critically-damped-ish smoothing, then a hard clamp: unbounded shear on
      // a trackpad fling turns the page into soup
      vel += (raw - vel) * Math.min(1, dt * 9);
      const target = Math.max(-1, Math.min(1, vel / 4200));
      shown += (target - shown) * Math.min(1, dt * 11);
      if (Math.abs(shown) < 0.0008) shown = 0;
      root.style.setProperty("--scroll-warp", shown.toFixed(4));
    });

    return () => {
      stop();
      root.style.setProperty("--scroll-warp", "0");
    };
  }, []);
  return null;
}

/* ═══════════════════════════════════════════════════════════════════════
   KINETIC — per-character reveal
   ═══════════════════════════════════════════════════════════════════════ */

export function Kinetic({
  text,
  className = "",
  /** ms between adjacent characters */
  stagger = 26,
  delay = 0,
  as: Tag = "span",
}: {
  text: string;
  className?: string;
  stagger?: number;
  delay?: number;
  as?: "span" | "h1" | "h2" | "h3";
}) {
  // Split on spaces first so words never break across a line mid-reveal —
  // splitting the raw string into characters lets the browser wrap anywhere
  // and produces orphaned letters at line ends.
  const words = useMemo(() => text.split(" "), [text]);
  let index = 0;

  return (
    <Tag className={`kinetic ${className}`} aria-label={text}>
      {words.map((word, w) => (
        <span className="kinetic-word" key={`${word}-${w}`} aria-hidden>
          {Array.from(word).map((ch, i) => {
            const d = delay + index++ * stagger;
            return (
              <span
                className="kinetic-char"
                key={i}
                style={{ "--kd": `${d}ms` } as CSSProperties}
              >
                {ch}
              </span>
            );
          })}
          {w < words.length - 1 && <span className="kinetic-space"> </span>}
        </span>
      ))}
    </Tag>
  );
}
