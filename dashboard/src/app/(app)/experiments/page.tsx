"use client";

/**
 * EXPERIMENTS — a descent through the frontal decision system.
 *
 * Same journey grammar as the swarm route: each panel sits in the structure
 * that performs its job, and scrolling flies the camera there.
 *
 *   dlPFC        weighing alternatives against each other
 *   parietal     sequencing a path and tracking where you are along it
 *   cerebellum   comparing prediction to outcome and correcting the model
 */

import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import ComparePanel from "@/components/ComparePanel";
import FlowPanel from "@/components/FlowPanel";
import CalibrationPanel from "@/components/CalibrationPanel";
import Station, { StationHeader, StationSpec } from "@/components/Station";

const CHOICE: StationSpec = {
  id: "expr-choice",
  kicker: "Sector EXPR",
  index: "01",
  anatomy: "dorsolateral prefrontal cortex",
  rationale:
    "value comparison — holding alternatives side by side and choosing between them is this region's defining job",
  region: "prefrontal",
  histology: "columns",
  camera: { cam: [0.05, 0.12, 0.15], look: [0.1, 0.16, 1.15], inside: 1 },
};

const SEQUENCE: StationSpec = {
  id: "expr-flow",
  kicker: "Station 02",
  index: "02",
  anatomy: "posterior parietal cortex",
  rationale:
    "path integration — tracking where you are in a sequence, and how far you still have to go",
  region: "parietal",
  histology: "laminae",
  camera: { cam: [0.0, 0.28, 0.12], look: [0.06, 1.15, -0.04], inside: 1 },
};

const ERROR: StationSpec = {
  id: "expr-calibration",
  kicker: "Station 03",
  index: "03",
  anatomy: "cerebellar cortex",
  rationale:
    "forward models and prediction error — the structure that only learns when the prediction was wrong",
  region: "cerebellum",
  histology: "folia",
  camera: { cam: [0.0, -0.22, -0.12], look: [0.0, -0.9, -1.2], inside: 1 },
};

export default function ExperimentsPage() {
  const [personaCount, setPersonaCount] = useState(0);

  useEffect(() => {
    api
      .sessions()
      .then((rows) => setPersonaCount(rows.filter((r) => r.persona).length))
      .catch(() => {});
  }, []);

  const seeded =
    personaCount > 0
      ? `${personaCount} persona${personaCount === 1 ? "" : "s"} available to every experiment.`
      : "Profile at least one session on the Swarm station first — experiments need seeded personas.";

  return (
    <div className="mx-auto max-w-5xl px-6 pb-24">
      <Station spec={CHOICE}>
        <div className="pt-16">
          <StationHeader
            spec={CHOICE}
            title="Compare"
            lede={`Zero-shot A/B across 2–8 variants, ranked with Beta-Binomial posteriors and an expected-loss stopping rule. ${seeded}`}
          />
        </div>
        <ComparePanel personaCount={personaCount} />
      </Station>

      <Station spec={SEQUENCE}>
        <StationHeader
          spec={SEQUENCE}
          title="Flow"
          lede="Walk the swarm through a multi-step journey. Twins' memory window shrinks with cognitive load, so late steps are judged with less context — as they would be by a tired person."
        />
        <FlowPanel personaCount={personaCount} />
      </Station>

      <Station spec={ERROR}>
        <StationHeader
          spec={ERROR}
          title="Calibration"
          lede="Record what actually happened and the simulation grades itself. Without real outcomes every number above is a prediction with no error bar on its accuracy."
        />
        <CalibrationPanel />
      </Station>
    </div>
  );
}
