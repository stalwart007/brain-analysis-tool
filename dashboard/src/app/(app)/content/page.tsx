"use client";

/**
 * CONTENT LAB.
 *
 * Two things used to be wrong with this page, and they compounded.
 *
 * The first was that the analysis stack under it is genuinely deep — a
 * permutation-tested inter-subject correlation, sup-t simultaneous confidence
 * bands, permutation-calibrated change-point detection, a per-twin peak-end
 * model, a Kaplan-Meier abandonment hazard — and NONE of that was legible
 * before running a study. A researcher arriving here could not tell whether
 * there was a statistical apparatus underneath or a language model producing
 * adjectives. `METHOD` exists to answer that in the four seconds before anyone
 * types anything, and every line in it names an estimator that actually runs.
 *
 * The second was that half the page was a stimulus preview: an HLS player
 * behind a URL box, which analysed nothing, connected to nothing, and sat
 * directly beneath the study panel implying that it did. Removing it is not a
 * cut of a feature so much as of a claim.
 *
 * Ingest leads now, because that was the actual blocker. A study you cannot
 * get content into is not a study.
 */

import { useEffect, useState } from "react";
import Reveal from "@/components/Reveal";
import ContentImpactPanel from "@/components/ContentImpactPanel";
import { api, SessionRow } from "@/lib/api";

/** What actually runs, named. Every entry is an estimator in `neuro.py` or
 *  `analytics.py`, not a description of an ambition. */
const METHOD: { stat: string; what: string }[] = [
  {
    stat: "synchrony",
    what: "inter-subject correlation of attention, against a 500-run beat-shuffle null",
  },
  {
    stat: "structure",
    what: "change-point detection, permutation-calibrated so a flat curve reports none",
  },
  {
    stat: "memory",
    what: "peak-end scored per twin, so a split audience shows as split",
  },
  {
    stat: "retention",
    what: "discrete-time abandonment hazard (Kaplan-Meier), with the risk set tracked",
  },
  {
    stat: "uncertainty",
    what: "sup-t simultaneous bands — valid read across the whole curve, not per beat",
  },
];

export default function ContentLabPage() {
  const [personaCount, setPersonaCount] = useState(0);

  useEffect(() => {
    api
      .sessions()
      .then((s: SessionRow[]) => setPersonaCount(s.filter((x) => x.persona).length))
      .catch(() => {});
  }, []);

  return (
    <div className="mx-auto max-w-5xl px-6 py-10">
      <Reveal>
        <div className="flex items-baseline gap-3">
          <span className="tag text-muted">Sector CNTL</span>
          <span className="h-px flex-1 bg-hairline" />
          <span className="panel-index">06</span>
        </div>
        <h1 className="display reg mt-2 text-[clamp(2.4rem,6vw,4.4rem)] text-bone">
          Content Lab
        </h1>
        <p className="mt-1 max-w-2xl text-sm text-muted">
          Any digital content — a YouTube link, a script, a static ad, a podcast,
          a landing page, a deck. The swarm experiences it beat by beat, and the
          analysis stack answers what it does to an audience.
        </p>

        {/* The apparatus, before anything is run. */}
        <dl className="mt-6 grid gap-x-6 gap-y-1.5 border-y border-hairline py-3 sm:grid-cols-2 lg:grid-cols-3">
          {METHOD.map((m) => (
            <div key={m.stat} className="flex gap-2">
              <dt className="hud-label shrink-0 text-accent">{m.stat}</dt>
              <dd className="text-[10px] leading-snug text-muted">{m.what}</dd>
            </div>
          ))}
        </dl>
        <p className="mt-1.5 font-mono text-[9px] leading-relaxed text-muted">
          Statistics a format cannot support are withheld with their reason
          rather than computed anyway — a static image has no timeline, so it
          gets no retention curve and says so.
        </p>

        <div className="mt-6">
          <ContentImpactPanel personaCount={personaCount} />
        </div>
      </Reveal>
    </div>
  );
}
