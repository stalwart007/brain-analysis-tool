"use client";

/**
 * THE CORTEX RAIL — navigation as anatomy.
 *
 * Replaces the sidebar entirely. Each sector is a probe site on the brain:
 * hovering one previews that region on the 3D cortex behind the page and
 * explains why the route lives there; selecting it flies the camera inside.
 * The strip underneath is a live depth readout — exterior vs intracortical —
 * so you always know where in the brain you are standing.
 */

import Link from "next/link";
import { REGION_COLORS, RegionId, Sector, SECTORS } from "@/lib/cortex";

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
  const preview = hovered
    ? SECTORS.find((s) => s.region === hovered) ?? active
    : active;

  // The depth strip reports where the *camera* is, which during a scroll
  // descent is finer-grained than the route.
  const atDepth = inside ?? active.station.inside;
  const anatomy = hovered ? preview.anatomy : depthLabel ?? preview.anatomy;
  const rationale = hovered ? preview.rationale : depthRationale ?? preview.rationale;

  return (
    <header className="cortex-rail sticky top-0 z-30">
      <div className="flex items-stretch gap-0 border-b border-hairline bg-page/85 backdrop-blur-md">
        {/* wordmark */}
        <Link
          href="/"
          className="group flex shrink-0 items-center gap-2.5 border-r border-hairline px-5"
          onMouseEnter={() => onHover(null)}
        >
          <span className="cortex-pip" />
          <span className="display reg text-[15px] leading-none text-bone">
            CogniSwarm
          </span>
        </Link>

        {/* probe sites */}
        <nav
          className="scrollbar-none flex flex-1 items-stretch overflow-x-auto"
          onMouseLeave={() => onHover(null)}
        >
          {SECTORS.map((s) => {
            const isActive = s.href === active.href;
            return (
              <Link
                key={s.href}
                href={s.href}
                onMouseEnter={() => onHover(s.region)}
                className={`probe group relative flex shrink-0 flex-col justify-center border-r border-hairline px-4 py-2.5 transition-colors ${
                  isActive ? "bg-bone text-void" : "text-ink-2 hover:bg-white/[0.05]"
                }`}
                aria-current={isActive ? "page" : undefined}
              >
                <span
                  className="probe-vein"
                  style={{ background: REGION_COLORS[s.region] }}
                />
                <span
                  className={`font-mono text-[8.5px] uppercase leading-none tracking-[0.14em] ${
                    isActive ? "text-void/55" : "text-muted"
                  }`}
                >
                  {s.anatomy}
                </span>
                <span className="display mt-1 text-[13px] leading-none">
                  {s.label}
                </span>
                <span
                  className={`mt-1 font-mono text-[8.5px] leading-none tracking-[0.2em] ${
                    isActive ? "text-void/45" : "text-muted/60"
                  }`}
                >
                  {s.code}
                </span>
              </Link>
            );
          })}
        </nav>

        <button
          onClick={onSignOut}
          onMouseEnter={() => onHover(null)}
          className="shrink-0 border-l border-hairline px-4 font-mono text-[10px] uppercase tracking-[0.14em] text-muted transition hover:text-critical"
        >
          ⏻ <span className="hidden sm:inline">Sign out</span>
        </button>
      </div>

      {/* Depth readout. Reports where the *camera* is, which during a scroll
          descent is finer-grained than the route — so it updates as you fall
          through the tissue, not only when you change page. */}
      <div className="depth-strip flex items-center gap-3 border-b border-hairline bg-surface/90 px-5 py-1.5 backdrop-blur-md">
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
          — {rationale}
        </span>
      </div>
    </header>
  );
}
