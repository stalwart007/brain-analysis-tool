/**
 * WHITE-MATTER TRACTOGRAPHY.
 *
 * The point cloud gives you cortex — grey matter, the surface. What was
 * missing is the thing that actually makes a brain a brain: the wiring
 * underneath. These are the major association, projection and commissural
 * bundles, reconstructed as streamlines the way diffusion MRI renders them.
 *
 * Colour follows the **standard DTI directional convention**, which is not an
 * arbitrary palette — it is how every tractography figure in the literature is
 * read:
 *
 *     red    left ↔ right      (commissural — corpus callosum)
 *     green  anterior ↔ posterior (association — arcuate, ILF, cingulum)
 *     blue   superior ↔ inferior  (projection — corticospinal)
 *
 * So the image is legible to anyone who has seen a real tractogram: the red
 * band across the midline IS the corpus callosum, the blue column descending
 * to the brainstem IS the corticospinal tract. Getting this right is what
 * separates "sci-fi brain" from "brain".
 *
 * Everything is seeded, so the anatomy is identical every render and between
 * server and client.
 */

import * as THREE from "three";

function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export interface Bundle {
  name: string;
  /** control points of the bundle's core streamline, in brain space */
  spine: [number, number, number][];
  /** how far individual streamlines scatter around the spine */
  spread: number;
  /** streamlines in this bundle */
  count: number;
  /** true ⇒ also emit the mirrored left-hemisphere twin */
  mirror: boolean;
}

/**
 * The bundles. Coordinates share the cortex's frame: +x right, +y superior,
 * +z anterior, cerebrum radius ≈ 1.
 */
export const BUNDLES: Bundle[] = [
  {
    // Commissural: the great arc joining the hemispheres. The densest, most
    // recognisable structure in any tractogram.
    name: "corpus callosum",
    spine: [
      [-0.62, 0.06, 0.24],
      [-0.34, 0.36, 0.36],
      [-0.1, 0.5, 0.28],
      [0.1, 0.5, 0.28],
      [0.34, 0.36, 0.36],
      [0.62, 0.06, 0.24],
    ],
    spread: 0.085,
    count: 132,
    mirror: false,
  },
  {
    // Projection: motor cortex → internal capsule → brainstem. Reads blue.
    name: "corticospinal tract",
    spine: [
      [0.26, 0.74, 0.02],
      [0.22, 0.42, 0.02],
      [0.17, 0.1, -0.02],
      [0.1, -0.28, -0.1],
      [0.05, -0.7, -0.16],
    ],
    spread: 0.045,
    count: 58,
    mirror: true,
  },
  {
    // Association: the language loop, frontal ↔ temporal, arcing round the
    // Sylvian fissure.
    name: "arcuate fasciculus",
    spine: [
      [0.44, 0.2, 0.66],
      [0.56, 0.34, 0.2],
      [0.62, 0.1, -0.22],
      [0.62, -0.2, -0.12],
      [0.58, -0.3, 0.16],
    ],
    spread: 0.05,
    count: 74,
    mirror: true,
  },
  {
    // Association: limbic belt arching above the callosum.
    name: "cingulum",
    spine: [
      [0.14, 0.28, 0.68],
      [0.12, 0.56, 0.24],
      [0.12, 0.5, -0.3],
      [0.15, 0.16, -0.62],
      [0.2, -0.16, -0.5],
    ],
    spread: 0.038,
    count: 46,
    mirror: true,
  },
  {
    // Association: occipital ↔ temporal — the ventral "what" stream, which is
    // exactly the route the Content Lab claims to model.
    name: "inferior longitudinal fasciculus",
    spine: [
      [0.3, -0.12, -0.84],
      [0.46, -0.2, -0.5],
      [0.56, -0.28, -0.1],
      [0.58, -0.32, 0.22],
    ],
    spread: 0.05,
    count: 54,
    mirror: true,
  },
  {
    // Association: the hook joining orbitofrontal to anterior temporal — the
    // value/emotion link, i.e. the Studies sector's own anatomy.
    name: "uncinate fasciculus",
    spine: [
      [0.34, -0.34, 0.74],
      [0.42, -0.46, 0.5],
      [0.52, -0.4, 0.26],
      [0.56, -0.3, 0.06],
    ],
    spread: 0.036,
    count: 40,
    mirror: true,
  },
];

export interface Tractogram {
  /** flat xyz pairs for LineSegments */
  positions: Float32Array;
  /** per-vertex DTI directional colour */
  colors: Float32Array;
  /** sampled streamlines, kept for riding flow particles along them */
  streamlines: THREE.Vector3[][];
  segmentCount: number;
}

/** Standard DTI directional colour: |Δx|→R, |Δz|→G, |Δy|→B. */
function directionColor(d: THREE.Vector3, out: THREE.Color) {
  const l = d.length() || 1;
  const r = Math.abs(d.x) / l;
  const g = Math.abs(d.z) / l;
  const b = Math.abs(d.y) / l;
  // Gamma-lift the channels: raw normalised components render muddy against a
  // near-black field, and real tractograms are luminous.
  out.setRGB(Math.pow(r, 0.72), Math.pow(g, 0.72), Math.pow(b, 0.72));
}

/**
 * Build the tractogram.
 *
 * Each streamline is a Catmull-Rom through a jittered copy of its bundle's
 * spine. Jitter is applied to the CONTROL POINTS rather than to the sampled
 * curve — perturbing sampled points makes a frayed, noisy line, whereas
 * perturbing controls keeps every streamline smooth while letting the bundle
 * as a whole fan out, which is how real fibre bundles look.
 */
export function buildTracts(samples = 34): Tractogram {
  const rand = mulberry32(0x5eed_1a7);
  const pos: number[] = [];
  const col: number[] = [];
  const streamlines: THREE.Vector3[][] = [];
  const c = new THREE.Color();
  const d = new THREE.Vector3();

  const emit = (bundle: Bundle, side: 1 | -1) => {
    for (let s = 0; s < bundle.count; s++) {
      const controls = bundle.spine.map(([x, y, z]) => {
        const j = bundle.spread;
        return new THREE.Vector3(
          side * x + (rand() - 0.5) * j,
          y + (rand() - 0.5) * j,
          z + (rand() - 0.5) * j
        );
      });
      const curve = new THREE.CatmullRomCurve3(controls, false, "catmullrom", 0.4);
      const pts = curve.getPoints(samples);
      streamlines.push(pts);

      for (let i = 0; i < pts.length - 1; i++) {
        const a = pts[i];
        const b = pts[i + 1];
        d.subVectors(b, a);
        directionColor(d, c);
        // Taper the ends: real streamlines fade where tracking loses
        // confidence, and hard-terminated lines look like wireframe.
        const t = i / (pts.length - 2);
        const fade = Math.sin(Math.PI * t) * 0.75 + 0.25;
        pos.push(a.x, a.y, a.z, b.x, b.y, b.z);
        col.push(
          c.r * fade, c.g * fade, c.b * fade,
          c.r * fade, c.g * fade, c.b * fade
        );
      }
    }
  };

  for (const bundle of BUNDLES) {
    emit(bundle, 1);
    if (bundle.mirror) emit(bundle, -1);
  }

  return {
    positions: new Float32Array(pos),
    colors: new Float32Array(col),
    streamlines,
    segmentCount: pos.length / 6,
  };
}
