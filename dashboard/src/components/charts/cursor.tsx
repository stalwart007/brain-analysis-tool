"use client";

/**
 * One cursor, shared across every chart that plots the same index.
 *
 * The charts in this product are not independent pictures — they are different
 * reductions of one object. In a content study, beat 3 in the attention curve,
 * beat 3 in the peak-index distribution, beat 3 in the retention hazard and
 * beat 3 on the brain map are the same moment seen four ways. Reading across
 * them was previously a matter of counting bars with your finger.
 *
 * So the cursor lives in a context keyed by DOMAIN rather than in each chart.
 * Hovering beat 3 anywhere highlights beat 3 everywhere that plots beats, and
 * nothing at all in a chart plotting cascade generations — which is a different
 * axis and would be a false link.
 *
 * Two states, deliberately:
 *   hovered  transient, follows the pointer
 *   pinned   sticky, survives the pointer leaving
 * Hover alone is unusable for the thing people actually want, which is to look
 * at one beat across four charts — that requires holding the mouse still in one
 * of them while reading another. Pinning is what makes cross-reading possible.
 */

import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";

type DomainState = { hovered: number | null; pinned: number | null };

interface CursorApi {
  get(domain: string): DomainState;
  setHovered(domain: string, i: number | null): void;
  togglePin(domain: string, i: number): void;
  clear(domain: string): void;
}

const Ctx = createContext<CursorApi | null>(null);

const EMPTY: DomainState = { hovered: null, pinned: null };

/** Wrap a panel (or a page) so every chart inside shares its cursor per domain. */
export function ChartCursorProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<Record<string, DomainState>>({});

  const api = useMemo<CursorApi>(
    () => ({
      get: (d) => state[d] ?? EMPTY,
      setHovered: (d, i) =>
        setState((s) => {
          const cur = s[d] ?? EMPTY;
          if (cur.hovered === i) return s; // identity-stable: no re-render on repeat
          return { ...s, [d]: { ...cur, hovered: i } };
        }),
      togglePin: (d, i) =>
        setState((s) => {
          const cur = s[d] ?? EMPTY;
          return { ...s, [d]: { ...cur, pinned: cur.pinned === i ? null : i } };
        }),
      clear: (d) => setState((s) => ({ ...s, [d]: EMPTY })),
    }),
    [state]
  );

  return <Ctx.Provider value={api}>{children}</Ctx.Provider>;
}

export interface ChartCursor {
  /** what the reader is actually looking at: pinned wins over hovered */
  index: number | null;
  hovered: number | null;
  pinned: number | null;
  isActive(i: number): boolean;
  isPinned(i: number): boolean;
  setHovered(i: number | null): void;
  togglePin(i: number): void;
  clear(): void;
  length: number;
  /** spread onto the element that receives pointer and keyboard input */
  surfaceProps: {
    ref: (el: HTMLElement | SVGElement | null) => void;
    onPointerMove: (e: React.PointerEvent) => void;
    onPointerLeave: () => void;
    onPointerDown: (e: React.PointerEvent) => void;
    onKeyDown: (e: React.KeyboardEvent) => void;
    onBlur: () => void;
    tabIndex: number;
    role: string;
    "aria-label": string;
    "aria-activedescendant"?: string;
    style: { touchAction: "none"; outline: "none" };
  };
}

/**
 * @param domain  charts sharing an x-axis share a domain: "beats", "generations", …
 * @param length  number of positions on that axis
 * @param label   what the chart shows, for screen readers
 * @param describe (i) => spoken description of position i
 */
export function useChartCursor(
  domain: string,
  length: number,
  label: string,
  describe?: (i: number) => string
): ChartCursor {
  const api = useContext(Ctx);
  const local = useState<DomainState>(EMPTY);
  const surfaceRef = useRef<HTMLElement | SVGElement | null>(null);

  // Usable outside a provider — a chart should never crash because someone
  // rendered it somewhere unwrapped; it just stops linking to its neighbours.
  const state = api ? api.get(domain) : local[0];
  const setHovered = useCallback(
    (i: number | null) =>
      api ? api.setHovered(domain, i) : local[1]((s) => ({ ...s, hovered: i })),
    [api, domain, local]
  );
  const togglePin = useCallback(
    (i: number) =>
      api
        ? api.togglePin(domain, i)
        : local[1]((s) => ({ ...s, pinned: s.pinned === i ? null : i })),
    [api, domain, local]
  );
  const clear = useCallback(
    () => (api ? api.clear(domain) : local[1](EMPTY)),
    [api, domain, local]
  );

  const index = state.pinned ?? state.hovered;

  /** Pointer x → index, by equal-width bands. Every chart on this axis is an
   *  evenly-spaced indexed series, so one rule serves all of them. */
  const indexAt = useCallback(
    (clientX: number) => {
      const el = surfaceRef.current;
      if (!el || length < 1) return null;
      const r = el.getBoundingClientRect();
      if (r.width <= 0) return null;
      const t = (clientX - r.left) / r.width;
      return Math.max(0, Math.min(length - 1, Math.floor(t * length)));
    },
    [length]
  );

  const surfaceProps = useMemo(
    () => ({
      ref: (el: HTMLElement | SVGElement | null) => {
        surfaceRef.current = el;
      },
      onPointerMove: (e: React.PointerEvent) => setHovered(indexAt(e.clientX)),
      onPointerLeave: () => setHovered(null),
      onPointerDown: (e: React.PointerEvent) => {
        const i = indexAt(e.clientX);
        if (i !== null) togglePin(i);
      },
      onKeyDown: (e: React.KeyboardEvent) => {
        // Keyboard parity, not an afterthought: the audit found the cortex rail
        // was mouse-only, so a whole layer of meaning was unreachable without a
        // pointer. Every chart here is navigable the same way.
        const cur = index ?? 0;
        if (e.key === "ArrowRight" || e.key === "ArrowDown") {
          e.preventDefault();
          setHovered(Math.min(length - 1, cur + 1));
        } else if (e.key === "ArrowLeft" || e.key === "ArrowUp") {
          e.preventDefault();
          setHovered(Math.max(0, cur - 1));
        } else if (e.key === "Home") {
          e.preventDefault();
          setHovered(0);
        } else if (e.key === "End") {
          e.preventDefault();
          setHovered(length - 1);
        } else if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          if (index !== null) togglePin(index);
        } else if (e.key === "Escape") {
          e.preventDefault();
          clear();
        }
      },
      // Blur clears the HOVER but keeps a pin: tabbing away should not discard
      // something the reader deliberately locked.
      onBlur: () => setHovered(null),
      tabIndex: 0,
      role: "application",
      "aria-label": describe && index !== null
        ? `${label}. ${describe(index)}${state.pinned !== null ? " (pinned)" : ""}`
        : `${label}. Use arrow keys to inspect, Enter to pin.`,
      style: { touchAction: "none" as const, outline: "none" as const },
    }),
    [indexAt, setHovered, togglePin, clear, index, length, label, describe, state.pinned]
  );

  return {
    index,
    hovered: state.hovered,
    pinned: state.pinned,
    isActive: (i) => index === i,
    isPinned: (i) => state.pinned === i,
    setHovered,
    togglePin,
    clear,
    length,
    surfaceProps,
  };
}
