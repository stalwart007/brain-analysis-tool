/**
 * Procedural cortex: the geometry and route-mapping behind the navigator.
 *
 * The brain is not decoration — it IS the site map. Each route lives in the
 * region that actually performs that cognitive job, and navigating flies the
 * camera from outside the skull *through the shell* into that lobe.
 *
 * Everything here is deterministic (seeded), so the anatomy is identical on
 * every render and between server and client.
 */

import * as THREE from "three";

/* ── deterministic noise ──────────────────────────────────────────────── */

function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Cortical folding: layered sinusoids give the deep gyri/sulci relief that
 *  makes a point cloud read as a brain rather than a lumpy sphere. */
function sulci(x: number, y: number, z: number): number {
  return (
    0.085 * Math.sin(8.2 * x + 2.1 * y) * Math.cos(6.8 * z - 1.3) +
    0.058 * Math.sin(12.6 * z + 4.7 * x) * Math.cos(10.4 * y + 2.2) +
    0.034 * Math.sin(19.3 * y - 3.4 * z) * Math.cos(16.1 * x) +
    0.018 * Math.sin(27.5 * x + 9.1 * z) * Math.cos(24.2 * y)
  );
}

/* ── regions ──────────────────────────────────────────────────────────── */

export type RegionId =
  | "prefrontal"
  | "orbitofrontal"
  | "parietal"
  | "temporal"
  | "occipital"
  | "hippocampus"
  | "brainstem"
  | "cerebellum";

export interface Station {
  /** camera position in brain space */
  cam: [number, number, number];
  /** point the camera looks at */
  look: [number, number, number];
  /** 0 = outside the skull, 1 = inside the tissue */
  inside: 0 | 1;
}

export interface Sector {
  href: string;
  label: string;
  code: string;
  region: RegionId;
  /** the anatomical structure, named honestly */
  anatomy: string;
  /** why this route lives in this region */
  rationale: string;
  /**
   * What the sector is FOR, in one line of plain language.
   *
   * `anatomy` and `rationale` explain why the route lives in this lobe — they
   * are answers to a question you only ask once you already know what the page
   * does. "White Room" and "Studies" mean nothing cold, so every destination
   * also states its job in the user's terms. The atlas leads with this.
   */
  purpose: string;
  station: Station;
}

/** Route → brain region. The mapping is functional, not arbitrary. */
export const SECTORS: Sector[] = [
  {
    href: "/",
    label: "Swarm",
    code: "SWRM",
    region: "prefrontal",
    anatomy: "whole cortex",
    rationale: "the intact brain — telemetry in, simulated minds out",
    purpose: "Profile real telemetry into segments, then fan one stimulus across the swarm",
    // a fixed, art-directed 3/4 view: the brain sits right of the masthead
    // and rotates in place rather than the camera orbiting around it
    station: { cam: [3.4, 0.6, 2.6], look: [-1.1, -0.5, 0], inside: 0 },
  },
  {
    href: "/experiments",
    label: "Experiments",
    code: "EXPR",
    region: "prefrontal",
    anatomy: "dorsolateral prefrontal cortex",
    rationale: "where choices between alternatives are weighed",
    purpose: "A/B up to eight variants, ranked with posteriors and a stopping rule",
    station: { cam: [0.05, 0.12, 0.15], look: [0.1, 0.16, 1.15], inside: 1 },
  },
  {
    href: "/studies",
    label: "Studies",
    code: "STDY",
    region: "orbitofrontal",
    anatomy: "orbitofrontal cortex",
    rationale: "subjective value — what a thing is worth to you",
    purpose: "Price curves, objection themes and viral spread for a single asset",
    station: { cam: [0, 0.05, 0.25], look: [0.18, -0.42, 0.95], inside: 1 },
  },
  {
    href: "/jobs",
    label: "Batch Jobs",
    code: "BTCH",
    region: "brainstem",
    anatomy: "brainstem",
    rationale: "autonomic processes that run without attention",
    purpose: "Queue large unattended swarms through the Batch API at half the cost",
    station: { cam: [0.15, 0.15, 0.35], look: [0, -0.95, -0.25], inside: 1 },
  },
  {
    href: "/panel",
    label: "Research Panel",
    code: "PANL",
    region: "hippocampus",
    anatomy: "hippocampal formation",
    rationale: "consolidation — the record of who consented, and when",
    purpose: "The consented, compensated respondent pool — invite, disclose, revoke",
    station: { cam: [-0.05, 0.4, 0.55], look: [0.42, -0.22, -0.05], inside: 1 },
  },
  {
    // The only sector that *generates* rather than measures. Anterior PFC is
    // where candidate futures are constructed and compared before any of them
    // is acted on, which is exactly what an evolutionary copy search does.
    href: "/optimize",
    label: "Optimize",
    code: "OPTM",
    region: "prefrontal",
    anatomy: "rostral prefrontal cortex",
    rationale: "generating candidate futures and selecting between them",
    purpose: "Evolve stronger copy over generations, and order a sequence optimally",
    station: { cam: [0.02, 0.2, 0.1], look: [0.05, 0.1, 1.2], inside: 1 },
  },
  {
    href: "/content",
    label: "Content Lab",
    code: "CNTL",
    region: "occipital",
    anatomy: "occipito-temporal stream",
    rationale: "where seen and heard content becomes meaning",
    purpose: "Score an image, video or audio asset through the swarm's minds",
    station: { cam: [0, 0.1, 0.1], look: [0.05, -0.05, -1.2], inside: 1 },
  },
  {
    // The TPJ is the canonical theory-of-mind region: it represents what
    // ANOTHER mind believes as separate from what you believe. A table of
    // characters reading each other — and stating one position while holding
    // a different one — is that computation, which is why the deliberation
    // study lives here rather than in the prefrontal decision sectors.
    href: "/room",
    label: "White Room",
    code: "ROOM",
    region: "temporal",
    anatomy: "temporoparietal junction",
    rationale:
      "theory of mind — representing what someone else believes, which is what a room of people reading each other is doing",
    purpose: "Seat characters at a table and make them argue a motion out loud",
    // laterally out through the right temporal lobe, at the posterior end
    // where the temporal and parietal cortices meet
    station: { cam: [0.12, 0.18, 0.14], look: [1.22, -0.06, -0.34], inside: 1 },
  },
];

export const REGION_COLORS: Record<RegionId, string> = {
  prefrontal: "#7db8ff",
  orbitofrontal: "#9d8bff",
  parietal: "#5fd0c8",
  temporal: "#ffb45f",
  occipital: "#ff7fc4",
  hippocampus: "#c08bff",
  brainstem: "#8fe08f",
  cerebellum: "#9aa4b8",
};

/**
 * The same hues as space-separated RGB channels.
 *
 * CSS needs the bare triple — `rgb(var(--region-rgb) / 0.22)` — so that one
 * variable can be re-used at any opacity. A hex string can't do that, and
 * `color-mix` on every usage would be far more expensive to recompute. Shell
 * writes exactly one of these onto <html> per navigation, and the entire
 * interface re-tints from it.
 */
export const REGION_RGB: Record<RegionId, string> = {
  prefrontal: "125 184 255",
  orbitofrontal: "157 139 255",
  parietal: "95 208 200",
  temporal: "255 180 95",
  occipital: "255 127 196",
  hippocampus: "192 139 255",
  brainstem: "143 224 143",
  cerebellum: "154 164 184",
};

/* ── shape ────────────────────────────────────────────────────────────── */

/** Which structure a point belongs to, from its position in brain space.
 *  Axes: +x right hemisphere, +y superior, +z anterior (front of head). */
function classify(x: number, y: number, z: number): RegionId {
  if (y < -0.62 && Math.abs(x) < 0.3 && z > -0.55) return "brainstem";
  if (z < -0.42 && y < -0.3) return "cerebellum";
  // hippocampus sits deep and medial, under the temporal lobe
  if (y < -0.05 && Math.abs(x) > 0.16 && Math.abs(x) < 0.58 && z > -0.45 && z < 0.35)
    return "hippocampus";
  if (z > 0.5) return y < -0.18 ? "orbitofrontal" : "prefrontal";
  if (z < -0.45) return "occipital";
  if (y > 0.42) return "parietal";
  if (Math.abs(x) > 0.55 && y < 0.25) return "temporal";
  return y > 0.1 ? "parietal" : "temporal";
}

export interface CortexCloud {
  positions: Float32Array;
  colors: Float32Array;
  /** outward surface direction per point — lets the shader shade by facing,
   *  which is what makes the cloud read as a solid volume instead of a haze */
  normals: Float32Array;
  regions: Uint8Array;
  regionIndex: Record<RegionId, number>;
  count: number;
}

const REGION_ORDER: RegionId[] = [
  "prefrontal", "orbitofrontal", "parietal", "temporal",
  "occipital", "hippocampus", "brainstem", "cerebellum",
];

/**
 * Build the cortical point cloud: two folded hemispheres split by the
 * longitudinal fissure, plus cerebellum and brainstem.
 */
export function buildCortex(count = 26000): CortexCloud {
  const rand = mulberry32(20260728);
  const positions = new Float32Array(count * 3);
  const colors = new Float32Array(count * 3);
  const normals = new Float32Array(count * 3);
  const regions = new Uint8Array(count);
  const regionIndex = Object.fromEntries(
    REGION_ORDER.map((r, i) => [r, i])
  ) as Record<RegionId, number>;
  const c = new THREE.Color();

  const cerebellumShare = 0.13;
  const stemShare = 0.05;
  const golden = Math.PI * (3 - Math.sqrt(5));

  for (let i = 0; i < count; i++) {
    const roll = rand();
    let x: number, y: number, z: number;

    if (roll < stemShare) {
      // brainstem: tapering column descending from the midbrain
      const t = rand();
      const r = 0.16 * (1 - 0.45 * t);
      const a = rand() * Math.PI * 2;
      x = Math.cos(a) * r;
      z = Math.sin(a) * r * 1.1 - 0.18 - t * 0.14;
      y = -0.5 - t * 0.72;
    } else if (roll < stemShare + cerebellumShare) {
      // cerebellum: dense, finely-foliated lobe behind and below
      const k = i * golden;
      const yy = 1 - (rand() * 2);
      const rr = Math.sqrt(Math.max(0, 1 - yy * yy));
      let px = Math.cos(k) * rr, py = yy, pz = Math.sin(k) * rr;
      const fold = 1 + 0.05 * Math.sin(26 * py) + 0.03 * Math.sin(19 * px);
      px *= 0.56 * fold; py *= 0.3 * fold; pz *= 0.44 * fold;
      x = px;
      y = py - 0.56;
      z = pz - 0.76;
    } else {
      // cerebrum: fibonacci direction → brain-shaped radius → sulcal folding
      const k = i * golden;
      const yy = 1 - (i / (count - 1)) * 2;
      const rr = Math.sqrt(Math.max(0, 1 - yy * yy));
      let dx = Math.cos(k) * rr;
      const dy = yy;
      let dz = Math.sin(k) * rr;

      // ellipsoid: longer front-to-back, flatter top-to-bottom
      let px = dx * 0.92;
      let py = dy * 0.86;
      let pz = dz * 1.16;

      // frontal pole rounds in, occipital pole tapers
      if (pz > 0) pz *= 1 - 0.1 * pz;
      else pz *= 1 + 0.06 * pz;
      // temporal lobes bulge outward and down
      if (py < 0 && Math.abs(px) > 0.35 && pz > -0.5) {
        px *= 1.14;
        py *= 1.12;
      }
      // underside flattens where the brain rests
      if (py < -0.45) py = -0.45 + (py + 0.45) * 0.55;

      // sulci: fold the surface
      const s = 1 + sulci(px, py, pz);
      px *= s; py *= s; pz *= s;

      // longitudinal fissure — the deep midline groove between hemispheres
      const side = px >= 0 ? 1 : -1;
      const absx = Math.abs(px);
      if (absx < 0.13) {
        const d = absx / 0.13;
        px = side * (0.055 + d * 0.075);
        py -= (1 - d) * 0.11;
      }

      // slight jitter so the surface reads as tissue, not a lattice
      x = px + (rand() - 0.5) * 0.016;
      y = py + (rand() - 0.5) * 0.016;
      z = pz + (rand() - 0.5) * 0.016;
    }

    positions[i * 3] = x;
    positions[i * 3 + 1] = y;
    positions[i * 3 + 2] = z;

    // radial direction approximates the surface normal closely enough for
    // facing-based shading on a convex-ish shell
    const nl = Math.hypot(x, y, z) || 1;
    normals[i * 3] = x / nl;
    normals[i * 3 + 1] = y / nl;
    normals[i * 3 + 2] = z / nl;

    const region = classify(x, y, z);
    regions[i] = regionIndex[region];
    c.set(REGION_COLORS[region]);
    colors[i * 3] = c.r;
    colors[i * 3 + 1] = c.g;
    colors[i * 3 + 2] = c.b;
  }

  return { positions, colors, normals, regions, regionIndex, count };
}

/** Centroid of a region — used to aim highlights and spawn synaptic sparks. */
export function regionCentroid(cloud: CortexCloud, region: RegionId): THREE.Vector3 {
  const idx = cloud.regionIndex[region];
  const v = new THREE.Vector3();
  let n = 0;
  for (let i = 0; i < cloud.count; i++) {
    if (cloud.regions[i] !== idx) continue;
    v.x += cloud.positions[i * 3];
    v.y += cloud.positions[i * 3 + 1];
    v.z += cloud.positions[i * 3 + 2];
    n++;
  }
  return n ? v.divideScalar(n) : v;
}

export function sectorFor(pathname: string): Sector {
  return SECTORS.find((s) => s.href === pathname) ?? SECTORS[0];
}
