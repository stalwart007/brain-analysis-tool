/**
 * POINT SPRITES AND DEPTH-FADED LINES — the legibility layer of the 3D scene.
 *
 * Why this file exists at all:
 *
 * three's `PointsMaterial` with `sizeAttenuation` computes
 *
 *     gl_PointSize = size · (viewportHeight / 2) / −z
 *
 * with **no upper bound**. That is fine for a camera parked outside the object
 * and catastrophic for this one, which parks *inside* the tissue. At an
 * interior station the camera sits ~0.05–0.3 world units from the wiring, so a
 * 0.028-unit packet resolves to 60–350 CSS pixels — and because `PointsMaterial`
 * draws the whole point quad, it lands as a huge opaque SQUARE, additively
 * blended, directly over the reading column. That single default is what made
 * paragraphs unreadable behind drifting blocks.
 *
 * Everything here fixes that at the source rather than masking it afterwards:
 *
 *   · size is CLAMPED in pixels, so nothing can ever swell past a few px;
 *   · the sprite is a ROUND soft falloff with a bright core, so it reads as a
 *     travelling light rather than a tile;
 *   · a NEAR FADE removes anything closer than a threshold — which is also
 *     physically honest, since you cannot resolve what you are inside of;
 *   · a FAR FADE keeps the far hemisphere from accumulating into additive haze.
 *
 * The line material applies the same near/far reasoning to the tractogram, so a
 * streamline passing centimetres from the lens dims instead of slashing a bright
 * stripe across the masthead.
 *
 * Both materials keep the `uSize · (k / dist)` convention already used by the
 * cortex point shader in CortexNavigator, so the three point layers attenuate
 * identically and read as one scene.
 */

import * as THREE from "three";

/* ── the reading well ─────────────────────────────────────────────────────
   The second half of the legibility problem is not depth, it is COMPOSITION.
   Even correctly-sized sprites pile up in the middle of the viewport, which is
   exactly where the text column is.

   The previous answer was a full-screen scrim that dimmed a vertical band. That
   works, but it is a blunt instrument: it dims the anatomy you want to see just
   as much as the anatomy that is in the way, and the page ends up uniformly
   murky.

   This does it in screen space instead, inside the same shader that draws the
   thing. Every layer computes its own NDC radius and quiets itself toward the
   centre of the frame, keeping full strength at the periphery. The composition
   that falls out is the one you actually want — a luminous rim with a calm
   well in the middle — and the eye is led inward instead of being dragged to
   whatever is brightest. Nothing is hidden; it is redistributed.

   The ellipse is taller than it is wide (y is scaled down) because a reading
   column is tall and narrow, so the protected zone should be too. */
export const READING_WELL = /* glsl */ `
  uniform vec2  uWell;      // x: NDC radius fully quieted, y: radius fully clear
  uniform float uWellFloor; // brightness retained dead centre (never 0)
  uniform float uWellAmt;   // 0 = off (exterior), 1 = full (intracortical)

  float readingWell(vec4 clip) {
    // guard w: behind-camera vertices produce huge or negative w and would
    // otherwise fold the periphery back into the centre
    float w = max(abs(clip.w), 1e-4);
    vec2 ndc = clip.xy / w;
    // 0.55 makes the protected region an upright ellipse rather than a circle
    float r = length(vec2(ndc.x, ndc.y * 0.55));
    float edge = smoothstep(uWell.x, uWell.y, r);
    return mix(1.0, mix(uWellFloor, 1.0, edge), uWellAmt);
  }
`;

/** Uniform block every reading-well shader needs. */
export function wellUniforms(well: [number, number], floor: number) {
  return {
    uWell: { value: new THREE.Vector2(well[0], well[1]) },
    uWellFloor: { value: floor },
    uWellAmt: { value: 0 },
  };
}

/** Shared radial sprite: soft halo, tight bright core, fully transparent rim. */
const SPRITE_FRAG = /* glsl */ `
  precision mediump float;
  varying vec3 vColor;
  varying float vAlpha;

  void main() {
    // gl_PointCoord is 0‥1 across the quad; r is 0 at centre, 1 at the edge
    vec2 d = gl_PointCoord - 0.5;
    float r = length(d) * 2.0;
    if (r > 1.0) discard;

    // halo + core. The core is what makes a 3 px sprite still read as a
    // luminous packet rather than a grey speck.
    float halo = smoothstep(1.0, 0.0, r);
    float core = smoothstep(0.35, 0.0, r);
    float a = halo * halo * 0.55 + core * 0.75;

    gl_FragColor = vec4(vColor, a * vAlpha);
    if (gl_FragColor.a < 0.004) discard;
  }
`;

const SPRITE_VERT = /* glsl */ `
  ${READING_WELL}
  uniform float uSize;      // pixel size at uRef world units away
  uniform float uRef;
  uniform float uOpacity;
  uniform float uMinPx;
  uniform float uMaxPx;
  uniform vec2  uNearFade;  // x: fully hidden below, y: fully shown above
  uniform vec2  uFarFade;   // x: fully shown below, y: fully hidden above
  uniform vec3  uTint;      // multiplied into the vertex colour
  varying vec3 vColor;
  varying float vAlpha;

  void main() {
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    float dist = -mv.z;

    // THE CLAMP. Without the upper bound this is the bug the whole file exists
    // to fix; without the lower bound distant packets vanish into sub-pixel
    // flicker as the camera drifts.
    float ps = uSize * (uRef / max(dist, 0.04));
    gl_PointSize = clamp(ps, uMinPx, uMaxPx);

    // near fade: dissolve what the lens is practically touching
    float near = smoothstep(uNearFade.x, uNearFade.y, dist);
    // far fade: stop the far hemisphere piling up as additive fog
    float far = 1.0 - smoothstep(uFarFade.x, uFarFade.y, dist);

    // The color attribute only exists when the geometry supplies one, so it is
    // compiled in rather than assumed: a missing attribute silently reads as
    // garbage on some drivers.
    #ifdef USE_VCOLOR
      vColor = color * uTint;
    #else
      vColor = uTint;
    #endif
    gl_Position = projectionMatrix * mv;
    vAlpha = uOpacity * near * far * readingWell(gl_Position);
  }
`;

export interface SpriteOptions {
  /** pixel diameter at `ref` world units from the camera */
  size?: number;
  /** the reference distance the size is quoted at */
  ref?: number;
  opacity?: number;
  minPx?: number;
  /** the ceiling that makes this material safe inside the tissue */
  maxPx?: number;
  /** [hidden below, shown above] in world units */
  nearFade?: [number, number];
  /** [shown below, hidden above] in world units */
  farFade?: [number, number];
  tint?: THREE.ColorRepresentation;
  /** true when the geometry carries a color attribute */
  vertexColors?: boolean;
  /** [NDC radius fully quieted, radius fully clear] */
  well?: [number, number];
  /** brightness retained at dead centre */
  wellFloor?: number;
}

/**
 * A points material that cannot blow up at close range.
 *
 * Defaults are tuned for the cortex scene: a brain of radius ≈ 1 with interior
 * camera stations a few hundredths of a unit off the tissue.
 */
export function makeSpriteMaterial(opts: SpriteOptions = {}): THREE.ShaderMaterial {
  const {
    size = 3.4,
    ref = 1.0,
    opacity = 0.9,
    minPx = 0.75,
    maxPx = 4.5,
    nearFade = [0.16, 0.62],
    farFade = [3.2, 6.0],
    tint = "#ffffff",
    vertexColors = true,
    well = [0.34, 1.15],
    wellFloor = 0.18,
  } = opts;

  return new THREE.ShaderMaterial({
    vertexShader: SPRITE_VERT,
    fragmentShader: SPRITE_FRAG,
    defines: vertexColors ? { USE_VCOLOR: "" } : {},
    uniforms: {
      uSize: { value: size },
      uRef: { value: ref },
      uOpacity: { value: opacity },
      uMinPx: { value: minPx },
      uMaxPx: { value: maxPx },
      uNearFade: { value: new THREE.Vector2(nearFade[0], nearFade[1]) },
      uFarFade: { value: new THREE.Vector2(farFade[0], farFade[1]) },
      uTint: { value: new THREE.Color(tint) },
      ...wellUniforms(well, wellFloor),
    },
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    vertexColors,
  });
}

/* ── depth-faded lines ────────────────────────────────────────────────────
   `LineBasicMaterial` has the same close-range problem in a quieter form: it
   cannot vary opacity with distance, so a streamline 5 cm from the lens is
   drawn at exactly the same brightness as one across the skull. Inside the
   tissue that is what puts a hard luminous diagonal through a paragraph.

   This is the same near/far envelope as the sprite, applied per vertex. */

const LINE_VERT = /* glsl */ `
  ${READING_WELL}
  uniform float uOpacity;
  uniform vec2  uNearFade;
  uniform vec2  uFarFade;
  uniform vec3  uTint;
  varying vec3 vColor;
  varying float vAlpha;

  void main() {
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    float dist = -mv.z;

    float near = smoothstep(uNearFade.x, uNearFade.y, dist);
    float far = 1.0 - smoothstep(uFarFade.x, uFarFade.y, dist);

    #ifdef USE_VCOLOR
      vColor = color * uTint;
    #else
      vColor = uTint;
    #endif

    gl_Position = projectionMatrix * mv;
    // Per-vertex, so a streamline entering the reading well dims along its
    // length rather than switching off — the gradient IS the effect.
    vAlpha = uOpacity * near * far * readingWell(gl_Position);
  }
`;

const LINE_FRAG = /* glsl */ `
  precision mediump float;
  varying vec3 vColor;
  varying float vAlpha;
  void main() {
    if (vAlpha < 0.004) discard;
    gl_FragColor = vec4(vColor, vAlpha);
  }
`;

export interface FadedLineOptions {
  opacity?: number;
  nearFade?: [number, number];
  farFade?: [number, number];
  tint?: THREE.ColorRepresentation;
  /** true when the geometry carries a color attribute */
  vertexColors?: boolean;
  /** [NDC radius fully quieted, radius fully clear] */
  well?: [number, number];
  /** brightness retained at dead centre */
  wellFloor?: number;
}

export function makeFadedLineMaterial(
  opts: FadedLineOptions = {}
): THREE.ShaderMaterial {
  const {
    opacity = 0.2,
    nearFade = [0.2, 0.95],
    farFade = [3.4, 6.4],
    tint = "#ffffff",
    vertexColors = true,
    /* Lines get a slightly wider, deeper well than sprites: a streamline is a
       continuous stroke and the eye tracks it across the column, so it has to
       be quieter there than a scatter of discrete points does. */
    well = [0.38, 1.2],
    wellFloor = 0.1,
  } = opts;

  return new THREE.ShaderMaterial({
    vertexShader: LINE_VERT,
    fragmentShader: LINE_FRAG,
    defines: vertexColors ? { USE_VCOLOR: "" } : {},
    uniforms: {
      uOpacity: { value: opacity },
      uNearFade: { value: new THREE.Vector2(nearFade[0], nearFade[1]) },
      uFarFade: { value: new THREE.Vector2(farFade[0], farFade[1]) },
      uTint: { value: new THREE.Color(tint) },
      ...wellUniforms(well, wellFloor),
    },
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    vertexColors,
  });
}
