"use client";

/**
 * THE CORTEX RAIL — navigation as anatomy, with nothing hidden.
 *
 * Each sector is a probe site on the brain: hovering one previews that region
 * on the 3D cortex behind the page and explains what the route is for;
 * selecting it flies the camera inside. The strip underneath is a live depth
 * readout — exterior vs intracortical — so you always know where in the brain
 * you are standing.
 *
 * WHY THIS IS A MONTAGE AND NOT A SCROLLING STRIP.
 *
 * The previous rail laid the eight sectors out in a single `overflow-x-auto`
 * row of `shrink-0` items, each carrying its full anatomical name — and
 * "dorsolateral prefrontal cortex" at tracking is a very wide piece of text.
 * The intrinsic width of that row measured 1269px against 479px of viewport:
 * five of the eight destinations were off-screen. `.scrollbar-none` then
 * removed the scrollbar, so there was no fade, no arrow, no counter, nothing
 * at all to say more existed. A whole section of the product was invisible
 * unless you happened to drag sideways over it.
 *
 * So the row became a MONTAGE — the EEG sense of the word: the complete set of
 * electrode sites, laid out at once, every channel always on screen. Eight
 * equal grid cells that reflow 8→4→4 columns instead of overflowing, each
 * showing its lobe, its name and its region colour, so the rail reads as a
 * coloured index of the cortex rather than a tab bar. Nothing is ever behind a
 * gesture.
 *
 * The full anatomical names, and the plain-language purpose of each sector,
 * move into the ATLAS (below): an overview panel that is also a quick
 * switcher. It is strictly an addition — every destination remains one click
 * away in the montage, because a shortcut nobody finds is the same bug again.
 */

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useReducedMotion } from "framer-motion";
import {
  CSSProperties,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { REGION_RGB, RegionId, Sector, SECTORS } from "@/lib/cortex";

/** Region colour as an inline channel, so one CSS var drives vein, glow and
 *  swatch for a given cell without a class per region. */
function regionVars(region: RegionId): CSSProperties {
  return { "--sector-rgb": REGION_RGB[region] } as CSSProperties;
}

/* ── the atlas ─────────────────────────────────────────────────────────── */

/**
 * THE CORTICAL ATLAS — the overview, and the quick switcher, in one surface.
 *
 * Grouped by lobe rather than listed flat, because the grouping is the site
 * map: three sectors genuinely live in the prefrontal cortex, and seeing them
 * gathered there is the point of mapping routes onto anatomy in the first
 * place. Typing filters across every field a person might remember — name,
 * code, lobe, anatomy, purpose — and arrow keys walk the visible order.
 */
function CortexAtlas({
  open,
  onClose,
  active,
  onHover,
}: {
  open: boolean;
  onClose: () => void;
  active: Sector;
  onHover: (r: RegionId | null) => void;
}) {
  const router = useRouter();
  const [query, setQuery] = useState("");
  const [cursor, setCursor] = useState(0);
  const sheet = useRef<HTMLDivElement>(null);
  const input = useRef<HTMLInputElement>(null);
  const reduced = useReducedMotion();

  const matches = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return SECTORS;
    return SECTORS.filter((s) =>
      [s.label, s.code, s.region, s.anatomy, s.purpose, s.rationale]
        .join(" ")
        .toLowerCase()
        .includes(q)
    );
  }, [query]);

  /* Lobes in the order the montage shows them, so the atlas and the rail agree
     about where things are. */
  const groups = useMemo(() => {
    const byRegion = new Map<RegionId, Sector[]>();
    for (const s of matches) {
      const list = byRegion.get(s.region);
      if (list) list.push(s);
      else byRegion.set(s.region, [s]);
    }
    return [...byRegion.entries()];
  }, [matches]);

  /* Grouping reorders the list — three prefrontal sectors gather at the top,
     so the fourth card on screen is not SECTORS[3]. The keyboard cursor has to
     index what is RENDERED or the highlight and the Enter target drift apart
     (arrow-down twice lit "Optimize" while Enter opened "Studies"). */
  const ordered = useMemo(() => groups.flatMap(([, list]) => list), [groups]);

  /* Reset per opening, not per render: reopening should start clean, but
     typing must not reset the cursor to a stale index. */
  useEffect(() => {
    if (!open) return;
    setQuery("");
    setCursor(0);
    // focus the filter so the panel is immediately typeable
    input.current?.focus();
  }, [open]);

  // a shortened result list must never leave the cursor past the end
  useEffect(() => {
    setCursor((c) => Math.min(c, Math.max(0, ordered.length - 1)));
  }, [ordered.length]);

  const go = useCallback(
    (s: Sector) => {
      onClose();
      router.push(s.href);
    },
    [onClose, router]
  );

  /* Dialog keyboard contract: Escape closes, arrows move, Enter commits, and
     Tab is trapped inside the sheet so focus cannot wander behind the scrim. */
  function onKeyDown(e: React.KeyboardEvent) {
    if (e.key === "Escape") {
      e.preventDefault();
      onClose();
      return;
    }
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      if (!ordered.length) return;
      const d = e.key === "ArrowDown" ? 1 : -1;
      setCursor((c) => (c + d + ordered.length) % ordered.length);
      return;
    }
    if (e.key === "Enter" && ordered[cursor]) {
      e.preventDefault();
      go(ordered[cursor]);
      return;
    }
    if (e.key === "Tab") {
      const nodes = sheet.current?.querySelectorAll<HTMLElement>(
        'a[href], button, input, [tabindex]:not([tabindex="-1"])'
      );
      if (!nodes?.length) return;
      const first = nodes[0];
      const last = nodes[nodes.length - 1];
      if (!e.shiftKey && document.activeElement === last) {
        e.preventDefault();
        first.focus();
      } else if (e.shiftKey && document.activeElement === first) {
        e.preventDefault();
        last.focus();
      }
    }
  }

  if (!open) return null;

  let flat = -1;

  return (
    <div className="atlas-layer" onKeyDown={onKeyDown}>
      {/* The scrim closes on click, which is why it is a button: a bare div
          with onClick is unreachable by keyboard and invisible to AT. */}
      <button
        type="button"
        className="atlas-scrim"
        aria-label="Close the cortical atlas"
        onClick={onClose}
        data-still={reduced ? "1" : undefined}
      />
      <div
        ref={sheet}
        role="dialog"
        aria-modal="true"
        aria-labelledby="atlas-title"
        className="atlas-sheet"
        data-still={reduced ? "1" : undefined}
      >
        <div className="atlas-head">
          <div className="flex items-baseline gap-3">
            <h2 id="atlas-title" className="display text-[15px] text-bone">
              Cortical Atlas
            </h2>
            <span className="h-px flex-1 bg-hairline" />
            <span className="hud-label">
              {matches.length} of {SECTORS.length} sectors
            </span>
          </div>
          <input
            ref={input}
            type="text"
            value={query}
            onChange={(e) => {
              setQuery(e.target.value);
              setCursor(0);
            }}
            placeholder="Filter by name, lobe, or what you want to do…"
            aria-label="Filter sectors"
            className="atlas-filter"
            autoComplete="off"
            spellCheck={false}
          />
        </div>

        <div className="atlas-body">
          {groups.map(([region, list]) => (
            <section key={region} className="atlas-group" style={regionVars(region)}>
              <div className="atlas-lobe">
                <span className="atlas-swatch" aria-hidden />
                <span>{region}</span>
                <span className="h-px flex-1 bg-hairline" />
              </div>
              <ul>
                {list.map((s) => {
                  flat += 1;
                  const isActive = s.href === active.href;
                  const isCursor = flat === cursor;
                  return (
                    <li key={s.href}>
                      <Link
                        href={s.href}
                        onClick={onClose}
                        onMouseEnter={() => {
                          onHover(s.region);
                          setCursor(ordered.indexOf(s));
                        }}
                        onMouseLeave={() => onHover(null)}
                        className="atlas-item"
                        data-active={isActive ? "1" : undefined}
                        data-cursor={isCursor ? "1" : undefined}
                        aria-current={isActive ? "page" : undefined}
                      >
                        <span className="atlas-item-vein" aria-hidden />
                        <span className="atlas-item-head">
                          <span className="display text-[14px] text-bone">
                            {s.label}
                          </span>
                          <span className="atlas-code">{s.code}</span>
                          {isActive && (
                            <span className="atlas-here">You are here</span>
                          )}
                        </span>
                        <span className="atlas-purpose">{s.purpose}</span>
                        <span className="atlas-anatomy">{s.anatomy}</span>
                      </Link>
                    </li>
                  );
                })}
              </ul>
            </section>
          ))}
          {!matches.length && (
            <p className="px-1 py-6 font-mono text-[11px] text-muted">
              No sector matches “{query}”.
            </p>
          )}
        </div>

        <div className="atlas-foot">
          <span className="hud-label">↑↓ move</span>
          <span className="hud-label">⏎ open</span>
          <span className="hud-label">esc close</span>
        </div>
      </div>
    </div>
  );
}

/* ── the rail ──────────────────────────────────────────────────────────── */

export default function CortexRail({
  active,
  hovered,
  onHover,
  onSignOut,
  depthLabel,
  depthRationale,
  inside,
}: {
  active: Sector;
  hovered: RegionId | null;
  onHover: (r: RegionId | null) => void;
  onSignOut: () => void;
  /** live anatomy from the scroll journey, when a page declares waypoints */
  depthLabel?: string;
  depthRationale?: string;
  /** live depth from the journey; falls back to the route's own station */
  inside?: 0 | 1;
}) {
  const [atlas, setAtlas] = useState(false);
  const trigger = useRef<HTMLButtonElement>(null);

  const preview = hovered
    ? SECTORS.find((s) => s.region === hovered) ?? active
    : active;

  // The depth strip reports where the *camera* is, which during a scroll
  // descent is finer-grained than the route.
  const atDepth = inside ?? active.station.inside;
  const anatomy = hovered ? preview.anatomy : depthLabel ?? preview.anatomy;
  /* Hovering a sector answers "what is this FOR" in plain language; with no
     pointer on the rail, a journey waypoint's rationale wins, because that is
     reporting live camera position rather than describing a destination. */
  const detail = hovered
    ? preview.purpose
    : depthRationale ?? preview.purpose;

  const closeAtlas = useCallback(() => {
    setAtlas(false);
    onHover(null);
    // returning focus to the control that opened the panel, not to <body>
    trigger.current?.focus();
  }, [onHover]);

  /* ⌘K / Ctrl-K opens the atlas, and so does "/" — but never while the caret
     is in a field, or the shortcut would eat the keystroke mid-sentence. */
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      const el = document.activeElement as HTMLElement | null;
      const typing =
        !!el &&
        (el.tagName === "INPUT" ||
          el.tagName === "TEXTAREA" ||
          el.tagName === "SELECT" ||
          el.isContentEditable);
      const combo = (e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k";
      if (combo || (e.key === "/" && !typing)) {
        e.preventDefault();
        setAtlas((v) => !v);
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  return (
    <header className="cortex-rail sticky top-0 z-30">
      {/* masthead: wordmark, the atlas key, and sign-out. Kept apart from the
          montage so the sector grid always gets the full viewport width and
          its column count never depends on how wide the wordmark happens to
          be — which is what let the old row overflow in the first place. */}
      <div className="rail-top">
        <Link
          href="/"
          className="flex shrink-0 items-center gap-2.5"
          onMouseEnter={() => onHover(null)}
        >
          <span className="cortex-pip" />
          <span className="display reg text-[15px] leading-none text-bone">
            CogniSwarm
          </span>
        </Link>

        <span className="h-px flex-1 bg-hairline" />

        <button
          ref={trigger}
          type="button"
          onClick={() => (atlas ? closeAtlas() : setAtlas(true))}
          onMouseEnter={() => onHover(null)}
          className="atlas-key"
          aria-haspopup="dialog"
          aria-expanded={atlas}
          /* Keeps the visible word "Atlas" inside the accessible name (WCAG
             2.5.3) while spelling out what the panel actually is, since the
             glyph and the shortcut hint beside it are aria-hidden. */
          aria-label={`Atlas — index of all ${SECTORS.length} sectors`}
        >
          <span aria-hidden>⌖</span>
          <span>Atlas</span>
          <span className="atlas-key-count">{SECTORS.length}</span>
          <kbd aria-hidden>⌘K</kbd>
        </button>

        <button
          type="button"
          onClick={onSignOut}
          onMouseEnter={() => onHover(null)}
          className="rail-signout"
        >
          <span aria-hidden>⏻</span>
          <span className="hidden sm:inline">Sign out</span>
          <span className="sr-only sm:hidden">Sign out</span>
        </button>
      </div>

      {/* THE MONTAGE — all eight probe sites, always on screen. */}
      <nav aria-label="Cortical sectors" onMouseLeave={() => onHover(null)}>
        <ul className="sector-grid">
          {SECTORS.map((s) => {
            const isActive = s.href === active.href;
            return (
              <li key={s.href} className="sector-cell">
                <Link
                  href={s.href}
                  onMouseEnter={() => onHover(s.region)}
                  onFocus={() => onHover(s.region)}
                  onBlur={() => onHover(null)}
                  className="sector"
                  style={regionVars(s.region)}
                  data-active={isActive ? "1" : undefined}
                  aria-current={isActive ? "page" : undefined}
                  title={`${s.label} — ${s.purpose}`}
                >
                  <span className="sector-vein" aria-hidden />
                  {/* Lobe and code share one line so the cell costs two text
                      rows rather than three — this rail is sticky on every
                      route, and a third row was 11px of permanent chrome for
                      no extra information. */}
                  <span className="sector-top">
                    <span className="sector-lobe">{s.region}</span>
                    {/* The active route is marked three ways over: inverted
                        fill (luminance), aria-current (assistive tech), and
                        this word (text, so it survives colour blindness,
                        greyscale and a forced-colours mode alike). It is
                        flex:none, so it is the lobe name that truncates on a
                        narrow cell, never the state cue. */}
                    {isActive && <em className="sector-here">here</em>}
                    <span className="sector-code">{s.code}</span>
                  </span>
                  <span className="sector-label">{s.label}</span>
                </Link>
              </li>
            );
          })}
        </ul>
      </nav>

      {/* Depth readout. Reports where the *camera* is, which during a scroll
          descent is finer-grained than the route — so it updates as you fall
          through the tissue, not only when you change page. */}
      <div className="depth-strip flex items-center gap-3 border-b border-hairline bg-surface/90 px-4 py-1 backdrop-blur-md">
        <span className="hud-label shrink-0">
          {atDepth ? "Depth · Intracortical" : "Depth · Exterior"}
        </span>
        <span className="depth-track" data-inside={atDepth}>
          <i />
        </span>
        <span
          className="shrink-0 font-mono text-[10px] tracking-wider transition-colors duration-500"
          style={{ color: "rgb(var(--region-rgb))" }}
        >
          {anatomy}
        </span>
        <span className="hidden truncate font-mono text-[10px] text-muted md:inline">
          — {detail}
        </span>
      </div>

      <CortexAtlas
        open={atlas}
        onClose={closeAtlas}
        active={active}
        onHover={onHover}
      />
    </header>
  );
}
