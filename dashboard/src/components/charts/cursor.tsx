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
  /**
   * What the reader is looking at RIGHT NOW: hover wins over the pin.
   *
   * The other way round reads as a broken chart — the tooltip tracks the
   * pointer while reporting numbers from somewhere else entirely, and hover
   * appears dead the moment anything is pinned. The pin is not a mode; it is a
   * marker that persists in every chart on the domain (see `isPinned`), which
   * is what makes cross-chart reading work. Both facts stay on screen.
   */
  index: number | null;
  hovered: number | null;
  pinned: number | null;
  isActive(i: number): boolean;
  isPinned(i: number): boolean;
  /** active or pinned — what a chart should keep lit while dimming the rest */
  isMarked(i: number): boolean;
  setHovered(i: number | null): void;
  togglePin(i: number): void;
  clear(): void;
  length: number;
  /** How position i maps onto the surface, as 0..1 fractions of its width.
   *  Anything drawn over a chart — crosshair, marker, brush — reads this
   *  instead of re-deriving the geometry and drifting from the hit-testing. */
  band(i: number): { left: number; width: number };
  mapping: "band" | "point";
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

/* ── 2-D variant ───────────────────────────────────────────────────────────
 *
 * A heatmap is not a series, so it cannot borrow the index cursor: reading one
 * needs a cell, and — more importantly — needs the row and column that cell
 * sits in, because "does B benefit from following A" is only interpretable
 * against the rest of A's row and B's column. So this highlights the cross,
 * not just the square.
 *
 * It stores the cell as `row * cols + col` in the SAME per-domain state as the
 * index cursor, which keeps one pin model, one Escape, one provider.
 */
export interface MatrixCursor {
  cell: { row: number; col: number } | null;
  pinnedCell: { row: number; col: number } | null;
  isActive(row: number, col: number): boolean;
  isPinned(row: number, col: number): boolean;
  /** in the cursor's row or column — this is what makes a heatmap readable */
  inCross(row: number, col: number): boolean;
  setHovered(row: number, col: number): void;
  clearHover(): void;
  togglePin(row: number, col: number): void;
  clear(): void;
  surfaceProps: ChartCursor["surfaceProps"];
}

export function useMatrixCursor(
  domain: string,
  rows: number,
  cols: number,
  label: string,
  describe?: (row: number, col: number) => string,
  opts?: { inset?: { left: number; top: number } }
): MatrixCursor {
  const api = useContext(Ctx);
  const local = useState<DomainState>(EMPTY);
  const surfaceRef = useRef<HTMLElement | SVGElement | null>(null);
  const insetL = opts?.inset?.left ?? 0;
  const insetT = opts?.inset?.top ?? 0;

  const state = api ? api.get(domain) : local[0];
  const setFlat = useCallback(
    (i: number | null) =>
      api ? api.setHovered(domain, i) : local[1]((s) => ({ ...s, hovered: i })),
    [api, domain, local]
  );
  const pinFlat = useCallback(
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

  const decode = (flat: number | null) =>
    flat === null || cols < 1 || flat < 0
      ? null
      : { row: Math.floor(flat / cols), col: flat % cols };

  const flat = state.hovered ?? state.pinned;
  const cell = decode(flat);
  const pinnedCell = decode(state.pinned);

  const cellAt = useCallback(
    (clientX: number, clientY: number) => {
      const el = surfaceRef.current;
      if (!el || rows < 1 || cols < 1) return null;
      const r = el.getBoundingClientRect();
      const w = r.width * (1 - insetL);
      const h = r.height * (1 - insetT);
      if (w <= 0 || h <= 0) return null;
      const col = Math.floor(((clientX - r.left - r.width * insetL) / w) * cols);
      const row = Math.floor(((clientY - r.top - r.height * insetT) / h) * rows);
      if (col < 0 || col >= cols || row < 0 || row >= rows) return null;
      return { row, col };
    },
    [rows, cols, insetL, insetT]
  );

  const surfaceProps = useMemo(
    () => ({
      ref: (el: HTMLElement | SVGElement | null) => {
        surfaceRef.current = el;
      },
      onPointerMove: (e: React.PointerEvent) => {
        const c = cellAt(e.clientX, e.clientY);
        setFlat(c ? c.row * cols + c.col : null);
      },
      onPointerLeave: () => setFlat(null),
      onPointerDown: (e: React.PointerEvent) => {
        const c = cellAt(e.clientX, e.clientY);
        if (c) pinFlat(c.row * cols + c.col);
      },
      onKeyDown: (e: React.KeyboardEvent) => {
        const cur = cell ?? { row: 0, col: 0 };
        const go = (row: number, col: number) => {
          e.preventDefault();
          setFlat(
            Math.max(0, Math.min(rows - 1, row)) * cols +
              Math.max(0, Math.min(cols - 1, col))
          );
        };
        if (e.key === "ArrowRight") go(cur.row, cur.col + 1);
        else if (e.key === "ArrowLeft") go(cur.row, cur.col - 1);
        else if (e.key === "ArrowDown") go(cur.row + 1, cur.col);
        else if (e.key === "ArrowUp") go(cur.row - 1, cur.col);
        else if (e.key === "Home") go(cur.row, 0);
        else if (e.key === "End") go(cur.row, cols - 1);
        else if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          if (cell) pinFlat(cell.row * cols + cell.col);
        } else if (e.key === "Escape") {
          e.preventDefault();
          clear();
        }
      },
      onBlur: () => setFlat(null),
      tabIndex: 0,
      role: "application",
      "aria-label":
        describe && cell
          ? `${label}. ${describe(cell.row, cell.col)}${
              state.pinned !== null && state.pinned === cell.row * cols + cell.col ? " (pinned)" : ""
            }`
          : `${label}. Use arrow keys to inspect cells, Enter to pin.`,
      style: { touchAction: "none" as const, outline: "none" as const },
    }),
    [cellAt, setFlat, pinFlat, clear, cell, rows, cols, label, describe, state.pinned]
  );

  return {
    cell,
    pinnedCell,
    isActive: (row, col) => cell?.row === row && cell?.col === col,
    isPinned: (row, col) => pinnedCell?.row === row && pinnedCell?.col === col,
    inCross: (row, col) => !!cell && (cell.row === row || cell.col === col),
    setHovered: (row, col) => setFlat(row * cols + col),
    clearHover: () => setFlat(null),
    togglePin: (row, col) => pinFlat(row * cols + col),
    clear,
    surfaceProps,
  };
}

/**
 * Keyboard and ARIA only — for a chart whose marks carry their own pointer
 * handlers because its geometry is not a fixed fraction of the container: a
 * matrix with an `auto`-width label column, or a graph whose nodes sit at
 * arbitrary positions rather than on an axis.
 *
 * Keyboard parity is the reason this exists separately. Dropping `surfaceProps`
 * entirely to get per-mark hover would silently drop arrow-key navigation with
 * it, which is how charts end up mouse-only without anyone deciding that.
 */
export function keyboardOnlyProps(cursor: {
  surfaceProps: ChartCursor["surfaceProps"];
}) {
  const {
    onPointerMove: _move,
    onPointerDown: _down,
    onPointerLeave: _leave,
    ...keyboard
  } = cursor.surfaceProps;
  return keyboard;
}

export interface CursorOptions {
  /**
   * Fractions of the surface width that are axis gutter, not plot area:
   * [left, right]. An SVG chart with a 30/340 label gutter needs this or the
   * cursor lands one position off near the edges — the pointer maths must use
   * the plot box, not the element box.
   */
  inset?: [number, number];
  /**
   * "band"  bars: position i owns an equal-width slice (the default).
   * "point" lines: position i sits ON a vertex, with 0 at the left edge and
   *         length-1 at the right, so the nearest vertex wins.
   * Using band maths on a line chart biases every reading half a slice left.
   */
  mapping?: "band" | "point";
}

/**
 * @param domain  charts sharing an x-axis share a domain: "beats", "generations", …
 * @param length  number of positions on that axis
 * @param label   what the chart shows, for screen readers
 * @param describe (i) => spoken description of position i
 * @param opts    plot-box inset and band-vs-point mapping
 */
export function useChartCursor(
  domain: string,
  length: number,
  label: string,
  describe?: (i: number) => string,
  opts?: CursorOptions
): ChartCursor {
  const insetL = opts?.inset?.[0] ?? 0;
  const insetR = opts?.inset?.[1] ?? 0;
  const mapping = opts?.mapping ?? "band";
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

  const index = state.hovered ?? state.pinned;

  /** Pointer x → index, over the PLOT box rather than the element box. */
  const indexAt = useCallback(
    (clientX: number) => {
      const el = surfaceRef.current;
      if (!el || length < 1) return null;
      const r = el.getBoundingClientRect();
      const plot = r.width * (1 - insetL - insetR);
      if (plot <= 0) return null;
      const t = (clientX - r.left - r.width * insetL) / plot;
      if (mapping === "point") {
        // vertices, not slices: index 0 sits at t=0 and length-1 at t=1
        if (length === 1) return 0;
        return Math.max(0, Math.min(length - 1, Math.round(t * (length - 1))));
      }
      return Math.max(0, Math.min(length - 1, Math.floor(t * length)));
    },
    [length, insetL, insetR, mapping]
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
        ? `${label}. ${describe(index)}${state.pinned === index ? " (pinned)" : ""}`
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
    isMarked: (i) => index === i || state.pinned === i,
    setHovered,
    togglePin,
    clear,
    length,
    mapping,
    band: (i: number) => {
      const plot = 1 - insetL - insetR;
      if (length < 1) return { left: insetL, width: 0 };
      if (mapping === "point") {
        // a hairline on the vertex, not a slice around it
        const at = length === 1 ? 0.5 : i / (length - 1);
        return { left: insetL + at * plot, width: 0 };
      }
      return { left: insetL + (i / length) * plot, width: (1 / length) * plot };
    },
    surfaceProps,
  };
}
