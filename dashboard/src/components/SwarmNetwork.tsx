"use client";

/**
 * Live-streaming swarm graph. `verdicts` is a GROWING array — one entry per
 * agent that has reported in. Each new entry makes a node ignite out of the
 * dim "potential swarm" on the shell, take its verdict colour, and fly into its
 * verdict community (convert / continue / hesitate / abandon), wiring itself to
 * its cluster-siblings. So the graph literally assembles agent-by-agent as the
 * simulation streams.
 *
 * Perf: one instanced mesh (nodes) + additive glow points (shared positions) +
 * link LineSegments + starfield + packets + core. DPR capped; loop pauses and
 * snaps to a correct static frame when off-screen / hidden / reduced-motion.
 */

import { useLayoutEffect, useMemo, useRef } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import * as THREE from "three";
import { useCanvasLiveness } from "./vizHooks";
import { fibSphere, makeGlowTexture } from "./glow";

export const NODE_CAP = 260;

type Verdict = "convert" | "continue" | "hesitate" | "abandon";
const CLUSTERS: Record<Verdict, { color: THREE.Color; center: THREE.Vector3 }> = {
  convert: { color: new THREE.Color("#16d016"), center: new THREE.Vector3(2.7, 0.7, 0.5) },
  continue: { color: new THREE.Color("#3f8ff0"), center: new THREE.Vector3(-2.7, 0.7, 0.5) },
  hesitate: { color: new THREE.Color("#f2ad1f"), center: new THREE.Vector3(2.4, -0.9, -0.9) },
  abandon: { color: new THREE.Color("#f0605f"), center: new THREE.Vector3(-2.4, -0.9, -0.9) },
};
const VERDICT_ORDER: Verdict[] = ["convert", "continue", "hesitate", "abandon"];
const PENDING = new THREE.Color("#2f3d54");
const CORE_COLOR = new THREE.Color("#a8ccff");

function Graph({
  count,
  verdicts,
  live,
}: {
  count: number;
  verdicts?: string[];
  live: boolean;
}) {
  const group = useRef<THREE.Group>(null);
  const nodes = useRef<THREE.InstancedMesh>(null);
  const glow = useRef<THREE.Points>(null);
  const links = useRef<THREE.LineSegments>(null);
  const packets = useRef<THREE.Points>(null);
  const core = useRef<THREE.Mesh>(null);
  const stars = useRef<THREE.Points>(null);
  const { camera, pointer } = useThree();
  const invalidate = useThree((s) => s.invalidate);

  const glowTex = useMemo(() => makeGlowTexture(), []);
  const dummy = useMemo(() => new THREE.Object3D(), []);

  const shell = useMemo(() => fibSphere(count, 2.25, 0.6), [count]);
  const clusterOffsets = useMemo(
    () =>
      Object.fromEntries(
        VERDICT_ORDER.map((v) => [v, fibSphere(count, 0.72, 0.28)])
      ) as Record<Verdict, THREE.Vector3[]>,
    [count]
  );

  const state = useMemo(() => {
    const pos = new Float32Array(count * 3);
    const vel = new Float32Array(count * 3);
    shell.forEach((p, i) => {
      pos[i * 3] = p.x;
      pos[i * 3 + 1] = p.y;
      pos[i * 3 + 2] = p.z;
    });
    const curColor = Array.from({ length: count }, () => PENDING.clone());
    return { pos, vel, curColor };
  }, [count, shell]);

  const glowPositions = useMemo(() => new Float32Array(count * 3), [count]);
  const glowColors = useMemo(() => {
    const arr = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) PENDING.toArray(arr, i * 3);
    return arr;
  }, [count]);

  // per-node target + colour + arrived flag, derived from the (growing) verdicts.
  const layout = useMemo(() => {
    const targets = new Float32Array(count * 3);
    const targetColor = Array.from({ length: count }, () => PENDING.clone());
    const arrived = new Array<boolean>(count).fill(false);
    const slot: Record<Verdict, number> = { convert: 0, continue: 0, hesitate: 0, abandon: 0 };
    const buckets: Record<Verdict, number[]> = { convert: [], continue: [], hesitate: [], abandon: [] };
    const n = verdicts ? Math.min(verdicts.length, count) : 0;

    for (let i = 0; i < count; i++) {
      if (i < n) {
        const v = ((verdicts![i] as Verdict) in CLUSTERS ? verdicts![i] : "continue") as Verdict;
        const k = slot[v]++;
        const off = clusterOffsets[v][k % count];
        CLUSTERS[v].center.clone().add(off).toArray(targets, i * 3);
        targetColor[i] = CLUSTERS[v].color;
        arrived[i] = true;
        buckets[v].push(i);
      } else {
        targets[i * 3] = shell[i].x;
        targets[i * 3 + 1] = shell[i].y;
        targets[i * 3 + 2] = shell[i].z;
      }
    }
    const linkPairs: [number, number][] = [];
    for (const v of VERDICT_ORDER) {
      const m = buckets[v];
      for (let k = 0; k < m.length; k++) {
        if (m.length > 1) linkPairs.push([m[k], m[(k + 1) % m.length]]);
        if (m.length > 4) linkPairs.push([m[k], m[(k + 2) % m.length]]);
      }
    }
    return { targets, targetColor, arrived, linkPairs, n };
  }, [verdicts, count, shell, clusterOffsets]);

  const linkPositions = useMemo(
    () => new Float32Array(Math.max(1, layout.linkPairs.length) * 2 * 3),
    [layout]
  );

  const starPositions = useMemo(() => {
    const N = 950;
    const arr = new Float32Array(N * 3);
    for (let i = 0; i < N; i++)
      new THREE.Vector3()
        .setFromSphericalCoords(9 + Math.random() * 7, Math.acos(2 * Math.random() - 1), Math.random() * 6.283)
        .toArray(arr, i * 3);
    return arr;
  }, []);

  const PACKETS = 64;
  const packetPositions = useMemo(() => new Float32Array(PACKETS * 3), []);
  const packetSeeds = useMemo(
    () =>
      Array.from({ length: PACKETS }, (_, i) => ({
        node: (i * 7919) % count,
        t: ((i * 104729) % 1000) / 1000,
        speed: 0.4 + (((i * 15485863) % 1000) / 1000) * 0.6,
      })),
    [count]
  );

  function writeFrame(animated: boolean, tSec: number, dSec: number) {
    const mesh = nodes.current;
    if (!mesh) return;
    for (let i = 0; i < count; i++) {
      const on = layout.arrived[i];
      for (let a = 0; a < 3; a++) {
        const idx = i * 3 + a;
        const target = layout.targets[idx];
        if (animated) {
          state.vel[idx] += (target - state.pos[idx]) * (on ? 10 : 5) * dSec;
          state.vel[idx] *= 0.82;
          state.pos[idx] += state.vel[idx] * dSec;
        } else {
          state.pos[idx] = target;
          state.vel[idx] = 0;
        }
      }
      const col = layout.targetColor[i];
      if (animated) state.curColor[i].lerp(col, Math.min(1, dSec * 5));
      else state.curColor[i].copy(col);

      const scale = on
        ? 1.15
        : 0.5 + (animated ? Math.sin(tSec * 3 + i * 0.6) * 0.18 : 0);
      dummy.position.set(state.pos[i * 3], state.pos[i * 3 + 1], state.pos[i * 3 + 2]);
      dummy.scale.setScalar(scale * 0.09);
      dummy.updateMatrix();
      mesh.setMatrixAt(i, dummy.matrix);
      mesh.setColorAt(i, state.curColor[i]);
      glowPositions[i * 3] = state.pos[i * 3];
      glowPositions[i * 3 + 1] = state.pos[i * 3 + 1];
      glowPositions[i * 3 + 2] = state.pos[i * 3 + 2];
      state.curColor[i].toArray(glowColors, i * 3);
    }
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    if (glow.current) {
      (glow.current.geometry.attributes.position as THREE.BufferAttribute).needsUpdate = true;
      (glow.current.geometry.attributes.color as THREE.BufferAttribute).needsUpdate = true;
    }
    if (links.current) {
      for (let l = 0; l < layout.linkPairs.length; l++) {
        const [i, j] = layout.linkPairs[l];
        linkPositions.set(
          [state.pos[i * 3], state.pos[i * 3 + 1], state.pos[i * 3 + 2], state.pos[j * 3], state.pos[j * 3 + 1], state.pos[j * 3 + 2]],
          l * 6
        );
      }
      (links.current.geometry.attributes.position as THREE.BufferAttribute).needsUpdate = true;
    }
  }

  useLayoutEffect(() => {
    if (!live) {
      writeFrame(false, 0, 0);
      invalidate();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [layout, live, count]);

  useFrame((st, delta) => {
    const d = Math.min(delta, 0.05);
    camera.position.x += (pointer.x * 0.8 - camera.position.x) * Math.min(1, d * 2.5);
    camera.position.y += (0.6 + pointer.y * 0.4 - camera.position.y) * Math.min(1, d * 2.5);
    camera.lookAt(0, 0, 0);
    if (!live) {
      writeFrame(false, 0, 0);
      return;
    }
    const t = st.clock.elapsedTime;
    const streaming = layout.n > 0 && layout.n < count;
    if (group.current) group.current.rotation.y += d * (streaming ? 0.09 : 0.06);
    if (stars.current) stars.current.rotation.y -= d * 0.01;
    if (core.current) {
      const pulse = streaming ? 1 + Math.sin(t * 4.5) * 0.14 : 1 + Math.sin(t * 1.3) * 0.05;
      core.current.scale.setScalar(pulse);
      core.current.rotation.set(t * 0.15, -t * 0.3, 0);
    }

    writeFrame(true, t, d);

    if (packets.current) {
      const mat = packets.current.material as THREE.PointsMaterial;
      mat.opacity = THREE.MathUtils.lerp(mat.opacity, streaming ? 0.9 : 0, d * 4);
      if (streaming) {
        for (let p = 0; p < PACKETS; p++) {
          const s = packetSeeds[p];
          const target = layout.arrived[s.node] ? s.node : s.node % Math.max(1, layout.n);
          const prog = (s.t + t * s.speed) % 1;
          packetPositions[p * 3] = state.pos[target * 3] * prog;
          packetPositions[p * 3 + 1] = state.pos[target * 3 + 1] * prog;
          packetPositions[p * 3 + 2] = state.pos[target * 3 + 2] * prog;
        }
        (packets.current.geometry.attributes.position as THREE.BufferAttribute).needsUpdate = true;
      }
    }
  });

  return (
    <group ref={group}>
      <points ref={stars}>
        <bufferGeometry>
          <bufferAttribute attach="attributes-position" args={[starPositions, 3]} />
        </bufferGeometry>
        <pointsMaterial size={0.03} color="#8fb4e8" transparent opacity={0.5} sizeAttenuation depthWrite={false} />
      </points>

      {layout.linkPairs.length > 0 && (
        <lineSegments ref={links}>
          <bufferGeometry>
            <bufferAttribute attach="attributes-position" args={[linkPositions, 3]} />
          </bufferGeometry>
          <lineBasicMaterial color="#5a86c8" transparent opacity={0.32} blending={THREE.AdditiveBlending} depthWrite={false} />
        </lineSegments>
      )}

      <points ref={glow}>
        <bufferGeometry>
          <bufferAttribute attach="attributes-position" args={[glowPositions, 3]} />
          <bufferAttribute attach="attributes-color" args={[glowColors, 3]} />
        </bufferGeometry>
        <pointsMaterial map={glowTex} size={0.28} vertexColors transparent opacity={0.72} sizeAttenuation depthWrite={false} blending={THREE.AdditiveBlending} />
      </points>

      <instancedMesh ref={nodes} args={[undefined, undefined, count]} frustumCulled={false}>
        <sphereGeometry args={[1, 10, 10]} />
        <meshBasicMaterial toneMapped={false} />
      </instancedMesh>

      <mesh ref={core}>
        <icosahedronGeometry args={[0.22, 1]} />
        <meshBasicMaterial color={CORE_COLOR} wireframe transparent opacity={0.9} toneMapped={false} />
      </mesh>
      <mesh>
        <sphereGeometry args={[0.36, 24, 24]} />
        <meshBasicMaterial color={CORE_COLOR} transparent opacity={0.15} blending={THREE.AdditiveBlending} depthWrite={false} />
      </mesh>

      <points ref={packets}>
        <bufferGeometry>
          <bufferAttribute attach="attributes-position" args={[packetPositions, 3]} />
        </bufferGeometry>
        <pointsMaterial map={glowTex} size={0.13} color="#bcd6ff" transparent opacity={0} sizeAttenuation depthWrite={false} blending={THREE.AdditiveBlending} />
      </points>
    </group>
  );
}

export default function SwarmNetwork({
  count,
  verdicts,
}: {
  count: number;
  active?: boolean;
  verdicts?: string[];
}) {
  const holder = useRef<HTMLDivElement>(null);
  const live = useCanvasLiveness(holder);
  const shown = Math.max(16, Math.min(NODE_CAP, count || 60));

  return (
    <div ref={holder} className="relative h-full w-full">
      <Canvas
        camera={{ position: [0, 0.6, 6.2], fov: 52 }}
        dpr={[1, 1.6]}
        frameloop={live ? "always" : "demand"}
        gl={{ antialias: false, powerPreference: "low-power" }}
      >
        <Graph count={shown} verdicts={verdicts} live={live} />
      </Canvas>
      {count > NODE_CAP && (
        <span className="pointer-events-none absolute bottom-2 right-3 text-[10px] text-muted">
          showing {NODE_CAP} of {count} twins
        </span>
      )}
    </div>
  );
}
