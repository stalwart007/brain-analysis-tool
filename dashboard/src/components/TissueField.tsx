"use client";

/**
 * THE HISTOLOGY LAYER — what makes arriving feel like being somewhere.
 *
 * Sits at z-1: above the cortical point cloud, below the content. That
 * sandwich is the whole trick — the field reads as matter suspended between
 * you and the far wall of the tissue, giving the page real depth instead of
 * a flat backdrop.
 *
 * Every field is mounted at once and crossfaded by opacity. Swapping
 * `background-image` instead would re-rasterise a full-viewport gradient on
 * arrival — a visible hitch at exactly the moment that has to feel smooth.
 */

import { useJourney } from "@/lib/journey";

const FIELDS = ["columns", "folia", "laminae", "grid"] as const;

export default function TissueField() {
  const { active } = useJourney();
  const current = active?.field;

  return (
    <div aria-hidden>
      {FIELDS.map((f) => (
        <div
          key={f}
          className="tissue-field"
          data-field={f}
          data-on={current === f ? "1" : "0"}
        />
      ))}
    </div>
  );
}
