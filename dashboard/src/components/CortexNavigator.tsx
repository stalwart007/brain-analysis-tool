"use client";

/**
 * THE CORTEX NAVIGATOR — the brain that *is* the navigation.
 *
 * The camera is driven by a *station*: a position, a look-at target, and a
 * flag for whether it sits outside the skull or inside the tissue. Two things
 * supply stations, and the shell resolves between them:
 *
 *   · the route (the cortex rail) — each sector owns the lobe that performs
 *     its job, so navigating flies through the shell into that region;
 *   · the scroll journey (lib/journey) — within one route, scrolling walks
 *     the camera between anatomical stations, so reading the page is
 *     travelling through the brain.
 *
 * Hovering a nav item previews a region without travelling.
 *
 * Performance budget: three draw calls (cortex points, axon lines, sparks),
 * one custom shader, additive blending, no postprocessing, DPR capped at 1.75.
 */

import { useEffect, useMemo, useRef } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import * as THREE from "three";
import {
  buildCortex,
  regionCentroid,
  REGION_COLORS,
  RegionId,
  Station,
} from "@/lib/cortex";
import { buildTracts } from "@/lib/tracts";
import {
  makeFadedLineMaterial,
  makeSpriteMaterial,
  READING_WELL,
  wellUniforms,
} from "@/lib/sprites";

const VERT = /* glsl */ `
  ${READING_WELL}
  attribute float aRegion;
  attribute vec3 aNormal;
  uniform float uActive;
  uniform float uHover;
  uniform float uSize;
  uniform float uTime;
  uniform float uInside;
  uniform float uTravel;   // 0 at rest, 1 at peak of a flight
  varying vec3 vColor;
  varying float vAlpha;

  void main() {
    vec3 p = position;
    // slow cortical "breathing" — the tissue is alive, not a model
    p *= 1.0 + 0.011 * sin(uTime * 0.55 + p.y * 3.1 + p.z * 1.7);

    vec4 mv = modelViewMatrix * vec4(p, 1.0);
    float dist = -mv.z;

    // NB: 'active' is a reserved word in GLSL ES — do not name a variable that.
    float isSel = step(abs(aRegion - uActive), 0.5);
    float isHov = step(abs(aRegion - uHover), 0.5);
    float emph = max(isSel, isHov * 0.7);

    // Facing term: points whose normal turns away from the camera dim, which
    // is what gives the cloud read as a solid volume rather than a flat haze.
    vec3 vdir = normalize(-mv.xyz);
    vec3 nrm = normalize(mat3(modelViewMatrix) * aNormal);
    float facing = clamp(dot(nrm, vdir), -1.0, 1.0);
    // hard falloff so the far hemisphere genuinely recedes…
    float shade = mix(0.06, 1.0, smoothstep(-0.25, 0.85, facing));
    // …plus a rim light at grazing angles, which is what reads as a silhouette
    float rim = pow(1.0 - abs(facing), 4.0) * 0.55;

    // inside the head, unrelated tissue recedes so the target lobe reads
    float base = mix(0.9, 0.12, uInside);
    // emphasis is dialled back inside, where the lobe fills the whole view
    vAlpha = (base + emph * mix(0.5, 0.22, uInside)) * shade + rim * (1.0 - uInside);
    // flight brightens the field — the sense of rushing through tissue
    vAlpha *= 1.0 + uTravel * 0.9;

    // unemphasised tissue desaturates toward bone; the active lobe keeps hue
    vColor = mix(vec3(0.80, 0.78, 0.78), color, 0.28 + emph * 0.72);

    // NEAR FADE. Clamping point size alone was not enough: at an interior
    // station the camera sits centimetres from tissue, and a cell 10 cm away
    // still renders as a full-size additive blob directly over the reading
    // column. A real lens would have those out of focus and out of frame, so
    // they are faded out entirely below ~0.45 world units — which is also
    // physically right, since you cannot resolve what you are inside of.
    vAlpha *= smoothstep(0.14, 0.48, dist);

    // true size attenuation: uSize is the pixel size at one world unit away,
    // so points grow as the camera enters the tissue — but clamped, or near
    // cells become giant blobs that swamp the page content
    float ps = uSize * (1.0 + emph * 0.8) * (3.5 / max(dist, 0.22));
    gl_PointSize = clamp(ps, 0.6, 7.0);
    gl_Position = projectionMatrix * mv;

    // The reading well (see lib/sprites): grey matter keeps full brightness
    // around the rim of the frame and steps back through the middle, where the
    // text column lives. Applied last so it scales everything above it.
    vAlpha *= readingWell(gl_Position);
  }
`;

const FRAG = /* glsl */ `
  varying vec3 vColor;
  varying float vAlpha;
  void main() {
    vec2 d = gl_PointCoord - 0.5;
    float r2 = dot(d, d);
    if (r2 > 0.25) discard;
    float a = smoothstep(0.25, 0.0, r2);
    gl_FragColor = vec4(vColor, a * vAlpha);
  }
`;

/**
 * WHITE MATTER — the tractogram, plus signal running along it.
 *
 * Separated from the Cortex component because its geometry never changes with
 * route: the wiring is the same wiring wherever you stand, only the viewpoint
 * moves. Rebuilding it per navigation would be pure waste.
 *
 * Two draw calls: one LineSegments for the streamlines (vertex-coloured by DTI
 * direction) and one Points for the traffic riding them.
 */
function WhiteMatter({ inside }: { inside: 0 | 1 }) {
  const tracts = useMemo(() => buildTracts(34), []);

  const geometry = useMemo(() => {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(tracts.positions, 3));
    g.setAttribute("color", new THREE.BufferAttribute(tracts.colors, 3));
    g.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 3);
    return g;
  }, [tracts]);

  /* One travelling packet per streamline. Each carries the colour of the
     bundle it rides, so the traffic is colour-coded by fibre direction exactly
     like the tracts themselves — commissural traffic is red, projection
     traffic blue. */
  const flow = useMemo(() => {
    const n = tracts.streamlines.length;
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(n * 3), 3));
    const colors = new Float32Array(n * 3);
    const c = new THREE.Color();
    const d = new THREE.Vector3();
    tracts.streamlines.forEach((line, i) => {
      d.subVectors(line[line.length - 1], line[0]);
      const l = d.length() || 1;
      c.setRGB(
        Math.pow(Math.abs(d.x) / l, 0.6),
        Math.pow(Math.abs(d.z) / l, 0.6),
        Math.pow(Math.abs(d.y) / l, 0.6)
      );
      c.toArray(colors, i * 3);
    });
    g.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    g.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 3);
    // golden-ratio phase offsets so packets never bunch into a visible pulse
    const phases = new Float32Array(n);
    for (let i = 0; i < n; i++) phases[i] = (i * 0.6180339887) % 1;
    return { geometry: g, phases };
  }, [tracts]);

  /* Both materials are depth-aware (see lib/sprites). Two terms matter:

     NEAR FADE — inside the tissue the camera sits centimetres from the wiring,
     and an un-faded streamline or packet at that range lands as a luminous slab
     across the reading column. Fading what the lens is touching is also the
     physically honest reading: you cannot resolve what you are inside of.

     READING WELL — the screen-space term that keeps the periphery luminous and
     the middle calm, which is what lets these run at full brightness instead of
     being globally dimmed to protect the text. */
  const lineMat = useMemo(
    () =>
      makeFadedLineMaterial({
        /* From outside you are reading the wiring THROUGH the cortex, so it
           stays under the surface detail. Low per-line alpha on purpose: with
           ~680 additive streamlines the DENSITY makes the bundle, and a higher
           alpha just blows each bundle's core out to white and destroys the
           directional colour that makes it a tractogram rather than string. */
        opacity: 0.15,
        nearFade: [0.18, 0.85],
        farFade: [3.4, 6.4],
      }),
    []
  );

  const flowMat = useMemo(
    () =>
      makeSpriteMaterial({
        size: 3.0,
        opacity: 0.55,
        maxPx: 4.0,
        nearFade: [0.16, 0.62],
        farFade: [3.0, 6.0],
      }),
    []
  );

  /* Depth-dependent levels live in an effect rather than in the material
     factory, so crossing the cortical surface animates a uniform instead of
     recompiling two shaders mid-flight. */
  useEffect(() => {
    lineMat.uniforms.uOpacity.value = inside ? 0.34 : 0.15;
    flowMat.uniforms.uOpacity.value = inside ? 0.95 : 0.55;
  }, [inside, lineMat, flowMat]);

  /* Smoothed depth, so the well crosses the cortical surface on the same ramp
     the camera does rather than snapping. */
  const insideMix = useRef(inside);

  useFrame((_, dt) => {
    insideMix.current += (inside - insideMix.current) * Math.min(1, dt * 1.6);
    lineMat.uniforms.uWellAmt.value = insideMix.current;
    flowMat.uniforms.uWellAmt.value = insideMix.current;

    const arr = flow.geometry.attributes.position.array as Float32Array;
    const ph = flow.phases;
    for (let i = 0; i < tracts.streamlines.length; i++) {
      ph[i] += dt * (0.09 + (i % 7) * 0.014);
      if (ph[i] > 1) ph[i] -= 1;
      const line = tracts.streamlines[i];
      // sample the polyline at the packet's arc position
      const f = ph[i] * (line.length - 1);
      const k = Math.floor(f);
      const frac = f - k;
      const a = line[k];
      const b = line[Math.min(line.length - 1, k + 1)];
      arr[i * 3] = a.x + (b.x - a.x) * frac;
      arr[i * 3 + 1] = a.y + (b.y - a.y) * frac;
      arr[i * 3 + 2] = a.z + (b.z - a.z) * frac;
    }
    flow.geometry.attributes.position.needsUpdate = true;
  });

  return (
    <group>
      <lineSegments geometry={geometry} material={lineMat} frustumCulled={false} />
      <points geometry={flow.geometry} material={flowMat} frustumCulled={false} />
    </group>
  );
}

interface CortexProps {
  /** where the camera stands */
  station: Station;
  /** which structure is emphasised */
  region: RegionId;
  /** changing this triggers a flight; any stable string works */
  stationKey: string;
  hovered: RegionId | null;
  /** rendered inside the spinning brain group, so it shares the anatomy's frame */
  children?: React.ReactNode;
}

function Cortex({ station, region, stationKey, hovered, children }: CortexProps) {
  // 9k read as a sparse dust cloud next to the tractogram's ~680 streamlines;
  // grey matter should be the denser of the two, since it is.
  const cloud = useMemo(() => buildCortex(16000), []);
  const { camera } = useThree();

  const geometry = useMemo(() => {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(cloud.positions, 3));
    g.setAttribute("color", new THREE.BufferAttribute(cloud.colors, 3));
    g.setAttribute("aNormal", new THREE.BufferAttribute(cloud.normals, 3));
    const reg = new Float32Array(cloud.count);
    for (let i = 0; i < cloud.count; i++) reg[i] = cloud.regions[i];
    g.setAttribute("aRegion", new THREE.BufferAttribute(reg, 1));
    g.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 3);
    return g;
  }, [cloud]);

  const material = useMemo(
    () =>
      new THREE.ShaderMaterial({
        vertexShader: VERT,
        fragmentShader: FRAG,
        uniforms: {
          uActive: { value: 0 },
          uHover: { value: -1 },
          uSize: { value: 4.2 },
          uTime: { value: 0 },
          uInside: { value: 0 },
          uTravel: { value: 0 },
          /* Grey matter is the densest layer, so it gets the shallowest well —
             enough to stop it competing with type, not so much that the lobe
             you travelled into disappears from behind its own page. */
          ...wellUniforms([0.3, 1.1], 0.3),
        },
        transparent: true,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
        vertexColors: true,
      }),
    []
  );

  /* axon web + travelling sparks, rebuilt when the active region changes */
  const axons = useMemo(() => {
    const idx = cloud.regionIndex[region];
    const pts: THREE.Vector3[] = [];
    for (let i = 0; i < cloud.count && pts.length < 420; i++) {
      if (cloud.regions[i] === idx && i % 7 === 0) {
        pts.push(
          new THREE.Vector3(
            cloud.positions[i * 3],
            cloud.positions[i * 3 + 1],
            cloud.positions[i * 3 + 2]
          )
        );
      }
    }
    const seg: number[] = [];
    const pairs: [THREE.Vector3, THREE.Vector3][] = [];
    for (let i = 0; i < pts.length; i++) {
      for (let j = i + 1; j < pts.length; j++) {
        if (pts[i].distanceTo(pts[j]) < 0.17 && seg.length < 3600) {
          seg.push(pts[i].x, pts[i].y, pts[i].z, pts[j].x, pts[j].y, pts[j].z);
          if (pairs.length < 90) pairs.push([pts[i], pts[j]]);
        }
      }
    }
    // an empty position attribute makes computeBoundingSphere return NaN,
    // so always emit at least one (degenerate) segment
    if (seg.length === 0) seg.push(0, 0, 0, 0, 0, 0);
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(seg, 3));
    g.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 3);
    // phases live here, not in an effect: an effect runs *after* the first
    // frames, so useFrame would read undefined and write NaN into positions
    const phases = new Float32Array(pairs.length);
    for (let i = 0; i < pairs.length; i++) phases[i] = (i * 0.6180339887) % 1;
    return { geometry: g, pairs, phases };
  }, [cloud, region]);

  const sparkGeo = useMemo(() => {
    const g = new THREE.BufferGeometry();
    g.setAttribute(
      "position",
      new THREE.BufferAttribute(new Float32Array(Math.max(1, axons.pairs.length) * 3), 3)
    );
    // this buffer is rewritten every frame, so declare the bounds rather than
    // letting three derive them from a buffer that is mid-update
    g.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 3);
    return g;
  }, [axons]);

  /* The local axon web and its traffic. Single-hued (the active region's
     colour) rather than vertex-coloured, and depth-faded on the same envelope
     as the tractogram so the two layers recede together. Created once and
     re-tinted through a uniform — rebuilding a material per region change
     would recompile the shader on every navigation. */
  const axonMat = useMemo(
    () =>
      makeFadedLineMaterial({
        opacity: 0.05,
        vertexColors: false,
        nearFade: [0.14, 0.7],
        farFade: [3.2, 6.0],
      }),
    []
  );
  const sparkMat = useMemo(
    () =>
      makeSpriteMaterial({
        size: 3.2,
        opacity: 0.35,
        maxPx: 4.2,
        vertexColors: false,
        nearFade: [0.15, 0.6],
        farFade: [3.0, 6.0],
      }),
    []
  );

  useEffect(() => {
    const hex = REGION_COLORS[region];
    axonMat.uniforms.uTint.value.set(hex);
    sparkMat.uniforms.uTint.value.set(hex);
    axonMat.uniforms.uOpacity.value = station.inside ? 0.16 : 0.05;
    sparkMat.uniforms.uOpacity.value = station.inside ? 0.9 : 0.35;
  }, [region, station.inside, axonMat, sparkMat]);
  /* ── camera flight between stations ────────────────────────────────────
     A decaying lerp never actually *arrives*, so travel felt mushy. This is
     a timed flight: on every sector change we capture where the camera is,
     then drive a normalised 0→1 clock through an ease so the journey has a
     real launch, a peak, and an arrival. */
  const FLIGHT_S = 1.9;
  const target = useRef(new THREE.Vector3(...station.look));
  const flight = useRef({
    t: 1,
    fromPos: new THREE.Vector3(...station.cam),
    fromLook: new THREE.Vector3(...station.look),
  });
  const lastKey = useRef(stationKey);
  const goalPos = useRef(new THREE.Vector3());
  const goalLook = useRef(new THREE.Vector3());
  const stage = useRef<HTMLElement | null>(null);
  const group = useRef<THREE.Group>(null);
  const spin = useRef(0);

  useEffect(() => {
    if (lastKey.current === stationKey) return;
    lastKey.current = stationKey;
    flight.current.fromPos.copy(camera.position);
    flight.current.fromLook.copy(target.current);
    flight.current.t = 0;
    stage.current =
      stage.current ?? (document.querySelector(".cortex-stage") as HTMLElement | null);
  }, [stationKey, camera]);

  useFrame((state, dt) => {
    const t = state.clock.elapsedTime;
    const f = flight.current;
    material.uniforms.uTime.value = t;
    material.uniforms.uActive.value = cloud.regionIndex[region];
    material.uniforms.uHover.value =
      hovered !== null ? cloud.regionIndex[hovered] : -1;

    // advance the flight clock and ease it (smootherstep: gentle out, gentle in)
    f.t = Math.min(1, f.t + dt / FLIGHT_S);
    const e = f.t * f.t * f.t * (f.t * (f.t * 6 - 15) + 10);
    // a bell over the flight: 0 at both ends, 1 mid-journey
    const travel = Math.sin(Math.PI * f.t) * (f.t < 1 ? 1 : 0);
    material.uniforms.uTravel.value = travel;
    if (stage.current) stage.current.style.setProperty("--travel", travel.toFixed(3));

    // uInside crosses over mid-flight — the moment of passing through the shell
    material.uniforms.uInside.value +=
      (station.inside - material.uniforms.uInside.value) * Math.min(1, dt * 1.6);

    /* The reading well only exists inside the tissue, because that is the only
       place the problem exists: from outside, the brain is a distant object
       beside the masthead and nothing crosses the text. Riding uInside means it
       arrives on exactly the same crossfade as everything else about being
       inside — one event, not two animations that happen to overlap. */
    const wellAmt = material.uniforms.uInside.value;
    material.uniforms.uWellAmt.value = wellAmt;
    axonMat.uniforms.uWellAmt.value = wellAmt;
    sparkMat.uniforms.uWellAmt.value = wellAmt;

    // The brain spins in place while the camera holds a designed frame; an
    // orbiting camera slowly dragged the composition out of alignment.
    const s = station;
    if (s.inside === 0) spin.current += dt * 0.055;
    if (group.current) group.current.rotation.y = spin.current;

    if (s.inside === 0) {
      // fixed hero framing, with only a slight breathing drift
      goalPos.current.set(
        s.cam[0],
        s.cam[1] + Math.sin(t * 0.22) * 0.08,
        s.cam[2]
      );
      goalLook.current.set(...s.look);
    } else {
      // interior stations are defined in brain space, so carry them through
      // the brain's current rotation to keep the target lobe framed
      const c = Math.cos(spin.current), sn = Math.sin(spin.current);
      const rot = (v: readonly [number, number, number]) =>
        [v[0] * c + v[2] * sn, v[1], -v[0] * sn + v[2] * c] as const;
      const rc = rot(s.cam);
      const rl = rot(s.look);
      goalPos.current.set(
        rc[0] + Math.sin(t * 0.4) * 0.035,
        rc[1] + Math.cos(t * 0.31) * 0.028,
        rc[2] + Math.sin(t * 0.23) * 0.03
      );
      goalLook.current.set(rl[0], rl[1], rl[2]);
    }

    if (f.t < 1) {
      camera.position.copy(f.fromPos).lerp(goalPos.current, e);
      target.current.copy(f.fromLook).lerp(goalLook.current, e);
    } else {
      camera.position.copy(goalPos.current);
      target.current.copy(goalLook.current);
    }
    camera.lookAt(target.current);

    // sparks ride the axon web of the active region
    const arr = sparkGeo.attributes.position.array as Float32Array;
    const ph = axons.phases;
    for (let i = 0; i < axons.pairs.length; i++) {
      // signalling accelerates during a flight — the tissue rushes past
      ph[i] += dt * (0.35 + (i % 5) * 0.12) * (1 + travel * 4);
      if (ph[i] > 1) ph[i] -= 1;
      const [a, b] = axons.pairs[i];
      arr[i * 3] = a.x + (b.x - a.x) * ph[i];
      arr[i * 3 + 1] = a.y + (b.y - a.y) * ph[i];
      arr[i * 3 + 2] = a.z + (b.z - a.z) * ph[i];
    }
    sparkGeo.attributes.position.needsUpdate = true;
  });

  const hue = REGION_COLORS[region];
  const centroid = useMemo(() => regionCentroid(cloud, region), [cloud, region]);

  return (
    <group ref={group}>
      <points geometry={geometry} material={material} frustumCulled={false} />
      <lineSegments geometry={axons.geometry} material={axonMat} frustumCulled={false} />
      <points geometry={sparkGeo} material={sparkMat} frustumCulled={false} />
      {/* The region's core glow.
          It must be DIMMER and smaller from inside, not brighter: an interior
          station parks the camera a few centimetres away in brain space, where
          a 0.05-radius sphere subtends a huge angle and additive grey reads as
          a flat disc pasted over the page. From outside it is a distant ember
          marking the target lobe, which is when it should carry weight. */}
      <mesh position={centroid}>
        <sphereGeometry args={[station.inside ? 0.012 : 0.05, 16, 16]} />
        <meshBasicMaterial
          color={hue}
          transparent
          opacity={station.inside ? 0.22 : 0.2}
          blending={THREE.AdditiveBlending}
          depthWrite={false}
        />
      </mesh>
      {/* white matter lives in here so it inherits the brain's rotation —
          wiring that stayed still while the cortex turned would read as two
          unrelated objects rather than one anatomy */}
      {children}
    </group>
  );
}

export default function CortexNavigator({
  station,
  region,
  stationKey,
  hovered,
}: CortexProps) {
  return (
    <div
      aria-hidden
      /* z-0, not negative: a negative z-index would place the canvas behind
         the opaque body background and render it invisible */
      className="cortex-stage pointer-events-none fixed inset-0 z-0"
      data-inside={station.inside}
    >
      <Canvas
        camera={{ position: [0, 0.3, 3.5], fov: 55, near: 0.01, far: 40 }}
        dpr={[1, 1.75]}
        gl={{ antialias: true, alpha: true, powerPreference: "high-performance" }}
      >
        <Cortex
          station={station}
          region={region}
          stationKey={stationKey}
          hovered={hovered}
        >
          <WhiteMatter inside={station.inside} />
        </Cortex>
      </Canvas>
    </div>
  );
}
