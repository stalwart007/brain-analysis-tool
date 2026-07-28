"use client";

/**
 * STUDIES — a descent through the valuation and transmission systems.
 *
 *   orbitofrontal  what a thing is worth to you
 *   temporal       language, and the threat appraisal inside an objection
 *   hippocampus    what gets encoded well enough to be passed on
 */

import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import PricePanel from "@/components/PricePanel";
import ObjectionPanel from "@/components/ObjectionPanel";
import ViralityPanel from "@/components/ViralityPanel";
import Station, { StationHeader, StationSpec } from "@/components/Station";

const VALUE: StationSpec = {
  id: "stdy-price",
  kicker: "Sector STDY",
  index: "01",
  anatomy: "orbitofrontal cortex",
  rationale:
    "subjective value — this tissue encodes what something is worth to you, in a common currency across very different goods",
  region: "orbitofrontal",
  histology: "columns",
  camera: { cam: [0, 0.05, 0.25], look: [0.18, -0.42, 0.95], inside: 1 },
};

const OBJECTION: StationSpec = {
  id: "stdy-objection",
  kicker: "Station 02",
  index: "02",
  anatomy: "superior temporal cortex",
  rationale:
    "language and appraisal — an objection is a threat judgement wearing a sentence",
  region: "temporal",
  // Superior temporal cortex is six-layer neocortex — strongly laminated, and
  // nothing to do with the entorhinal grid this station used to paint. Using
  // the hippocampal field here was both anatomically wrong and a repeat of the
  // field the very next station needs.
  histology: "laminae",
  camera: { cam: [0.1, -0.05, 0.05], look: [1.2, -0.3, 0.1], inside: 1 },
};

const TRANSMISSION: StationSpec = {
  id: "stdy-virality",
  kicker: "Station 03",
  index: "03",
  anatomy: "hippocampal formation",
  rationale:
    "consolidation — nothing spreads that was not first encoded well enough to survive being retold",
  region: "hippocampus",
  histology: "grid",
  camera: { cam: [-0.05, 0.4, 0.55], look: [0.42, -0.22, -0.05], inside: 1 },
};

export default function StudiesPage() {
  const [personaCount, setPersonaCount] = useState(0);

  useEffect(() => {
    api
      .sessions()
      .then((rows) => setPersonaCount(rows.filter((r) => r.persona).length))
      .catch(() => {});
  }, []);

  const seeded =
    personaCount > 0
      ? `Running against ${personaCount} profiled persona${personaCount === 1 ? "" : "s"}.`
      : "Profile at least one session on the Swarm station first.";

  return (
    <div className="mx-auto max-w-5xl px-6 pb-24">
      <Station spec={VALUE}>
        <div className="pt-16">
          <StationHeader
            spec={VALUE}
            title="Price"
            lede={`Demand and revenue curves across candidate prices, with a log-log elasticity fit and a continuous revenue optimum that can fall between the prices you tested. ${seeded}`}
          />
        </div>
        <PricePanel personaCount={personaCount} />
      </Station>

      <Station spec={OBJECTION}>
        <StationHeader
          spec={OBJECTION}
          title="Objections"
          lede="Every twin's single biggest reason to say no, embedded and clustered into themes, ranked with the dealbreaker rate alongside."
        />
        <ObjectionPanel personaCount={personaCount} />
      </Station>

      <Station spec={TRANSMISSION}>
        <StationHeader
          spec={TRANSMISSION}
          title="Virality"
          lede="A Galton-Watson branching process seeded from twin share intent. R₀, extinction probability by PGF fixed point, and 2,000 simulated cascades — heterogeneity matters, so the mixture is not collapsed to its mean."
        />
        <ViralityPanel personaCount={personaCount} />
      </Station>
    </div>
  );
}
