"use client";

/**
 * OPTIMIZE — the generative sector.
 *
 * Every other route measures something you already made. These two make it:
 * one searches the space of copy, the other searches the space of orderings.
 * That difference is why they get their own sector rather than sitting under
 * Studies.
 */

import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import OptimizerPanel from "@/components/OptimizerPanel";
import SequencePanel from "@/components/SequencePanel";
import Station, { StationHeader, StationSpec } from "@/components/Station";

const SEARCH: StationSpec = {
  id: "optm-copy",
  kicker: "Sector OPTM",
  index: "01",
  anatomy: "rostral prefrontal cortex",
  rationale:
    "prospection — constructing candidate futures and holding them side by side is the most anterior thing the cortex does",
  region: "prefrontal",
  histology: "columns",
  camera: { cam: [0.02, 0.2, 0.1], look: [0.05, 0.1, 1.2], inside: 1 },
};

const ORDER: StationSpec = {
  id: "optm-sequence",
  kicker: "Station 02",
  index: "02",
  anatomy: "hippocampal formation",
  rationale:
    "temporal order memory — the hippocampus encodes not just what happened but the sequence it happened in, which is the whole question here",
  region: "hippocampus",
  histology: "grid",
  camera: { cam: [-0.05, 0.4, 0.55], look: [0.42, -0.22, -0.05], inside: 1 },
};

export default function OptimizePage() {
  const [personaCount, setPersonaCount] = useState(0);

  useEffect(() => {
    api
      .sessions()
      .then((rows) => setPersonaCount(rows.filter((r) => r.persona).length))
      .catch(() => {});
  }, []);

  const seeded =
    personaCount > 0
      ? `Searching against ${personaCount} profiled persona${personaCount === 1 ? "" : "s"}.`
      : "Profile at least one session on the Swarm station first — the search needs a swarm to score against.";

  return (
    <div className="mx-auto max-w-5xl px-6 pb-24">
      <Station spec={SEARCH}>
        <div className="pt-16">
          <StationHeader
            spec={SEARCH}
            title="Evolve"
            lede={`An evolutionary loop over copy: score, breed from the elites, reallocate budget to what's working. ${seeded}`}
          />
        </div>
        <OptimizerPanel personaCount={personaCount} />
      </Station>

      <Station spec={ORDER}>
        <StationHeader
          spec={ORDER}
          title="Order"
          lede="N! orderings is intractable, so precedence is estimated pairwise and the linear-ordering problem solved heuristically — with position effects reported apart from content effects."
        />
        <SequencePanel personaCount={personaCount} />
      </Station>
    </div>
  );
}
