"use client";

/**
 * THE JOURNEY — scroll as descent through tissue.
 *
 * The cortex rail flies the camera between *routes*. This adds the second
 * axis: within a single route, scrolling walks the camera between anatomical
 * stations, so reading the page IS travelling through the brain. A section
 * declares where in the head it lives; when it becomes the dominant thing on
 * screen, the camera goes there and the whole interface re-tints to that
 * region's hue.
 *
 * Why an explicit provider rather than each section poking the camera: two
 * sections are frequently on screen at once during a scroll, and the camera
 * must not fight itself. The provider arbitrates — one winner at a time,
 * chosen by how close a section is to the viewport's focal line — and it is
 * the single place that decides.
 */

import {
  createContext,
  ReactNode,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { RegionId, Station } from "./cortex";

export interface Waypoint {
  /** stable id, also the flight-change key */
  id: string;
  station: Station;
  region: RegionId;
  /** shown in the depth HUD while you are here */
  anatomy: string;
  /** one line explaining why this content lives in this structure */
  rationale: string;
  /** which histological field paints behind the content on arrival */
  field?: string;
}

interface JourneyState {
  /** the waypoint currently driving the camera, or null → use the route default */
  active: Waypoint | null;
  /** 0‥1 progress through the whole descent, for the depth gauge */
  progress: number;
}

interface JourneyApi extends JourneyState {
  register: (el: HTMLElement, wp: Waypoint) => () => void;
}

const Ctx = createContext<JourneyApi | null>(null);

export function useJourney(): JourneyApi {
  const ctx = useContext(Ctx);
  if (!ctx) {
    // Pages outside a provider (every route except the descent) simply get a
    // null waypoint, and the shell falls back to the route's own station.
    return { active: null, progress: 0, register: () => () => {} };
  }
  return ctx;
}

export function JourneyProvider({ children }: { children: ReactNode }) {
  const [active, setActive] = useState<Waypoint | null>(null);
  const [progress, setProgress] = useState(0);
  const entries = useRef(new Map<HTMLElement, Waypoint>());
  const raf = useRef(0);

  /* Arbitration runs on a rAF-coalesced scroll handler rather than inside an
     IntersectionObserver callback. IO fires on threshold crossings, which is
     too coarse here: we need the *closest* section to the focal line on every
     frame of a scroll, not a notification that one became 30% visible. */
  const recompute = useCallback(() => {
    raf.current = 0;
    if (entries.current.size === 0) return;

    // The focal line sits at 42% of the viewport — slightly above centre,
    // which is where the eye actually rests when reading a long page.
    const focus = window.innerHeight * 0.42;
    let best: Waypoint | null = null;
    let bestDist = Infinity;

    for (const [el, wp] of entries.current) {
      const r = el.getBoundingClientRect();
      // distance from the focal line to the section's nearest edge (0 while
      // the line is inside the section, so a tall section stays locked in)
      const d = r.top > focus ? r.top - focus : r.bottom < focus ? focus - r.bottom : 0;
      if (d < bestDist) {
        bestDist = d;
        best = wp;
      }
    }

    setActive((prev) => (prev?.id === best?.id ? prev : best));

    const doc = document.documentElement;
    const scrollable = doc.scrollHeight - window.innerHeight;
    setProgress(scrollable > 0 ? Math.min(1, Math.max(0, window.scrollY / scrollable)) : 0);
  }, []);

  const schedule = useCallback(() => {
    if (raf.current) return;
    raf.current = requestAnimationFrame(recompute);
  }, [recompute]);

  /* Registration and mount resolve SYNCHRONOUSLY, not through rAF.
     requestAnimationFrame does not fire in a hidden or backgrounded tab, so a
     purely rAF-coalesced version leaves `active` null until the user both
     focuses the tab AND scrolls — meaning the page loads with no camera
     station, no region tint and no histology. rAF is only ever used to
     coalesce bursts of scroll events, which is the one job it is right for. */
  const register = useCallback(
    (el: HTMLElement, wp: Waypoint) => {
      entries.current.set(el, wp);
      recompute();
      return () => {
        entries.current.delete(el);
        recompute();
      };
    },
    [recompute]
  );

  useEffect(() => {
    recompute();
    const onVisible = () => {
      if (document.visibilityState === "visible") recompute();
    };
    window.addEventListener("scroll", schedule, { passive: true });
    window.addEventListener("resize", recompute);
    document.addEventListener("visibilitychange", onVisible);
    return () => {
      window.removeEventListener("scroll", schedule);
      window.removeEventListener("resize", recompute);
      document.removeEventListener("visibilitychange", onVisible);
      if (raf.current) cancelAnimationFrame(raf.current);
    };
  }, [schedule, recompute]);

  const api = useMemo<JourneyApi>(
    () => ({ active, progress, register }),
    [active, progress, register]
  );

  return <Ctx.Provider value={api}>{children}</Ctx.Provider>;
}

/** Attach a waypoint to a section. Returns the ref to spread onto the element. */
export function useWaypoint(wp: Waypoint) {
  const { register } = useJourney();
  const ref = useRef<HTMLElement | null>(null);
  // The waypoint object is rebuilt on every render by callers writing it
  // inline; key the effect on its id so we don't re-register every frame.
  const wpRef = useRef(wp);
  wpRef.current = wp;

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    return register(el, wpRef.current);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [register, wp.id]);

  return ref;
}
