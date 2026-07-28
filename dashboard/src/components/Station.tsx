"use client";

/**
 * A STATION — one stop on the descent.
 *
 * Scrolling a station into the focal line does four things at once:
 *   1. flies the 3D camera to that structure (via the journey waypoint),
 *   2. re-tints the entire interface to the structure's hue,
 *   3. crossfades the histology field to that structure's architecture,
 *   4. grows the station's arrival mark.
 *
 * All four are driven off the same `active` decision, which is why arriving
 * feels like one event rather than four animations that happen to overlap.
 */

import { ReactNode } from "react";
import { RegionId, Station as CameraStation } from "@/lib/cortex";
import { useJourney, useWaypoint } from "@/lib/journey";
import { Kinetic } from "./Craft";

/** Which histological field a structure paints behind the content. */
export type Histology = "columns" | "folia" | "laminae" | "grid" | "none";

export interface StationSpec {
  id: string;
  /** the structure, named honestly — shown in the depth HUD on arrival */
  anatomy: string;
  /** why this functionality lives in this structure */
  rationale: string;
  region: RegionId;
  camera: CameraStation;
  histology: Histology;
  /** short label for the arrival marker */
  kicker: string;
  /** the numeral, e.g. "01" */
  index: string;
}

export function useStationActive(id: string): boolean {
  const { active } = useJourney();
  return active?.id === id;
}

export default function Station({
  spec,
  children,
  className = "",
  /** the surface station is full-height and centred; the rest flow normally */
  full = false,
}: {
  spec: StationSpec;
  children: ReactNode;
  className?: string;
  full?: boolean;
}) {
  const ref = useWaypoint({
    id: spec.id,
    station: spec.camera,
    region: spec.region,
    anatomy: spec.anatomy,
    rationale: spec.rationale,
    field: spec.histology,
  });

  return (
    <section
      ref={ref}
      data-station={spec.id}
      className={`relative ${full ? "flex min-h-dvh flex-col justify-center" : "py-20"} ${className}`}
    >
      {children}
    </section>
  );
}

/**
 * The arrival marker. Reads as an instrument label pressed against tissue:
 * depth, structure, and the one-line reason this functionality is here.
 */
export function StationHeader({
  spec,
  title,
  lede,
}: {
  spec: StationSpec;
  title: string;
  lede?: string;
}) {
  const here = useStationActive(spec.id);

  return (
    <header className="station-mark mb-8" data-here={here ? "1" : "0"}>
      <div className="flex flex-wrap items-baseline gap-3">
        <span className="tag text-muted">{spec.kicker}</span>
        <span
          className="font-mono text-[10px] uppercase tracking-[0.16em] transition-colors duration-700"
          style={{ color: here ? "rgb(var(--region-rgb))" : "var(--color-muted)" }}
        >
          {spec.anatomy}
        </span>
        <span className="h-px flex-1 bg-hairline" />
        <span className="panel-index">{spec.index}</span>
      </div>

      {/* Re-keyed on arrival so the reveal replays each time you scroll back
          to a station — a one-shot mount animation only ever fires once, and
          the descent is something you move up and down through. */}
      <Kinetic
        key={`${spec.id}-${here}`}
        as="h2"
        text={title}
        stagger={30}
        className="display mt-3 text-[clamp(1.8rem,4.5vw,3rem)] text-bone"
      />

      {/* Serif, not sans. A lede is written prose, not chrome — giving it the
          same face as the navigation was what made it read as UI copy. */}
      {lede && <p className="lede mt-3">{lede}</p>}

      {/* the rationale is the honest part: why this lives in this structure.
          Italic serif, because it is an aside in the author's voice. */}
      <p className="rationale mt-2.5">{spec.rationale}</p>
    </header>
  );
}
