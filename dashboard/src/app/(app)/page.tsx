"use client";

/**
 * THE SWARM PAGE — a descent through the cortex.
 *
 * Not a grid of panels but a journey: four stations, each sited in the
 * structure that actually performs that job, with scrolling flying the camera
 * between them. Arriving somewhere re-tints the interface, swaps the histology
 * behind the content, and grows the station's arrival mark.
 *
 * The anatomy is chosen functionally, not decoratively — and deliberately
 * visits three structures the top-level rail never does (parietal, temporal,
 * cerebellum), so the descent explores parts of the brain the nav doesn't:
 *
 *   SURFACE     whole cortex   the intact organ, before you go in
 *   PARIETAL    telemetry      where streams of sensory events become a
 *                              coherent picture of what someone did
 *   PREFRONTAL  the swarm      simulating other minds is what dorsolateral
 *                              PFC is for
 *   CEREBELLUM  calibration    the structure that learns from repetition by
 *                              minimising prediction error — which is exactly
 *                              what the run history is for
 */

import { useCallback, useEffect, useState } from "react";
import { api, SessionRow, SwarmRunRow } from "@/lib/api";
import SessionsPanel from "@/components/SessionsPanel";
import SwarmPanel from "@/components/SwarmPanel";
import RunsChart from "@/components/RunsChart";
import Station, { StationHeader, StationSpec } from "@/components/Station";
import { Electrode } from "@/components/Chrome";
import { useCountUp } from "@/components/vizHooks";

const SURFACE: StationSpec = {
  id: "swrm-surface",
  kicker: "Sector SWRM",
  index: "00",
  anatomy: "whole cortex",
  rationale: "the intact organ — telemetry in, simulated minds out",
  region: "prefrontal",
  histology: "none",
  // the art-directed exterior framing: brain right of the masthead, rotating
  // in place rather than the camera orbiting it
  camera: { cam: [3.4, 0.6, 2.6], look: [-1.1, -0.5, 0], inside: 0 },
};

const TELEMETRY: StationSpec = {
  id: "swrm-parietal",
  kicker: "Station 01",
  index: "01",
  anatomy: "posterior parietal cortex",
  rationale:
    "multisensory integration — where separate streams of events become one coherent picture of what a person did",
  region: "parietal",
  histology: "laminae",
  // inside, looking up into the parietal roof
  camera: { cam: [0.0, 0.28, 0.12], look: [0.06, 1.15, -0.04], inside: 1 },
};

const SIMULATION: StationSpec = {
  id: "swrm-prefrontal",
  kicker: "Station 02",
  index: "02",
  anatomy: "dorsolateral prefrontal cortex",
  rationale:
    "theory of mind and deliberation — modelling how someone else will react is the job this tissue evolved to do",
  region: "prefrontal",
  histology: "columns",
  camera: { cam: [0.05, 0.12, 0.15], look: [0.1, 0.16, 1.15], inside: 1 },
};

const CALIBRATION: StationSpec = {
  id: "swrm-cerebellum",
  kicker: "Station 03",
  index: "03",
  anatomy: "cerebellar cortex",
  rationale:
    "the brain's error-correction engine — it learns from repeated prediction error, which is precisely what a run history is for",
  region: "cerebellum",
  histology: "folia",
  camera: { cam: [0.0, -0.22, -0.12], look: [0.0, -0.9, -1.2], inside: 1 },
};

function Counted({ value }: { value: number }) {
  return <>{useCountUp(value)}</>;
}

export default function DashboardPage() {
  const [sessions, setSessions] = useState<SessionRow[]>([]);
  const [runs, setRuns] = useState<SwarmRunRow[]>([]);

  const refresh = useCallback(() => {
    api.sessions().then(setSessions).catch(() => {});
    api.swarmRuns().then(setRuns).catch(() => {});
  }, []);

  useEffect(() => {
    refresh();
    const t = setInterval(refresh, 15000);
    return () => clearInterval(t);
  }, [refresh]);

  const profiled = sessions.filter((s) => s.persona).length;
  const swarmRuns = runs.filter((r) => r.kind === "swarm");
  const lastRun = swarmRuns[0]?.result;

  return (
    <div className="mx-auto max-w-6xl px-6 pb-24">
      {/* ── SURFACE ──────────────────────────────────────────────────────
          Nothing but the organ and its vitals. No card, no panel — the whole
          viewport is the brain. */}
      <Station spec={SURFACE} full>
        <div className="tag mb-3 text-muted">{SURFACE.kicker}</div>
        <h1 className="display reg text-[clamp(3rem,11vw,8rem)] text-bone">Swarm</h1>
        <p className="mt-3 max-w-xl text-sm text-ink-2">
          Telemetry → cognitive models → personas → simulated reactions.
        </p>

        {/* electrode readings pinned to the tissue, not tiles stacked on it */}
        <div className="mt-12 flex flex-wrap gap-x-10 gap-y-6">
          <Electrode label="Segments" value={<Counted value={sessions.length} />} />
          <Electrode label="Personas" value={<Counted value={profiled} />} />
          <Electrode label="Runs" value={<Counted value={runs.length} />} />
          <Electrode
            label="Last intent"
            value={lastRun ? lastRun.mean_intent.toFixed(2) : "—"}
            live={!!lastRun}
          />
        </div>

        <div className="descend-cue mt-16 flex items-center gap-3">
          <span className="hud-label">Descend</span>
          <span className="text-lg leading-none text-ink-2">↓</span>
        </div>
      </Station>

      {/* ── STATION 01 · PARIETAL — telemetry arrives ────────────────── */}
      <Station spec={TELEMETRY}>
        <StationHeader
          spec={TELEMETRY}
          title="Telemetry"
          lede="Consent-gated kinematics, resolved into behavioural segments. Profiling one runs the full cognitive pipeline — drift-diffusion, HMM, information dynamics — before any LLM sees it."
        />
        <SessionsPanel sessions={sessions} onChanged={refresh} />
      </Station>

      {/* ── STATION 02 · PREFRONTAL — the forecast fires ─────────────── */}
      <Station spec={SIMULATION}>
        <StationHeader
          spec={SIMULATION}
          title="Forecast response"
          lede="Fan out one stimulus across the twin swarm. Twins are isolated — no inter-agent chatter — so the spread you see is genuine disagreement, not a herd."
        />
        <SwarmPanel personaCount={profiled} onRan={refresh} />
      </Station>

      {/* ── STATION 03 · CEREBELLUM — prediction error ───────────────── */}
      <Station spec={CALIBRATION}>
        <StationHeader
          spec={CALIBRATION}
          title="Calibration"
          lede="Engagement and intent across every run. Record real outcomes on the Experiments station and this becomes a measured error curve rather than a trend line."
        />
        <div className="card h-72 p-6">
          <RunsChart runs={swarmRuns} />
        </div>
      </Station>
    </div>
  );
}
