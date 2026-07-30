"use client";

/**
 * WHAT SYNCHRONY LOOKS LIKE BY CHANCE, and where this content landed.
 *
 * The backend runs five hundred permutations to produce the p-value beside the
 * ISC figure. Every one of them is a complete re-run of the estimator on
 * beat-shuffled curves — by a wide margin the most expensive thing in the
 * study — and the entire result was being reported as `p=0.004`, a number that
 * is simultaneously the strongest evidence here and the one a reader is least
 * equipped to feel.
 *
 * Drawn, it stops being a claim. The grey mass is what agreement looks like
 * when the beats are shuffled and no shared structure survives; the marker is
 * where these twins actually landed. A reader who has never heard of a
 * permutation test can see whether the observation is inside the noise or
 * outside it, which is the whole question.
 *
 * The alternative — a bar labelled "ISC 0.42" — is the same number with the
 * evidence for it deleted.
 */

import { useMemo } from "react";
import ChartFrame from "@/components/charts/ChartFrame";
import { useChartCursor } from "@/components/charts/cursor";

export interface IscBlock {
  overall: number;
  grip: string;
  p_value?: number;
  significant?: boolean;
  ci?: [number, number] | number[] | null;
  null_mean_r?: number;
  n_permutations?: number;
  n_twins?: number;
  null_r_distribution?: number[];
  method: string;
}

const BINS = 28;

export default function SynchronyNull({ isc }: { isc: IscBlock }) {
  const nulls = isc.null_r_distribution ?? [];

  const { bins, lo, hi, peak } = useMemo(() => {
    if (!nulls.length) return { bins: [] as number[], lo: -1, hi: 1, peak: 1 };
    // The axis spans the null AND the observation. Fitting it to the null alone
    // puts a strongly synchronised result off the right edge, which is exactly
    // the case the chart exists to show.
    const values = [...nulls, isc.overall];
    const rawLo = Math.min(...values);
    const rawHi = Math.max(...values);
    const pad = Math.max(0.04, (rawHi - rawLo) * 0.08);
    const lo = rawLo - pad;
    const hi = rawHi + pad;
    const counts = new Array(BINS).fill(0);
    for (const v of nulls) {
      const i = Math.min(BINS - 1, Math.max(0, Math.floor(((v - lo) / (hi - lo)) * BINS)));
      counts[i] += 1;
    }
    return { bins: counts, lo, hi, peak: Math.max(...counts, 1) };
  }, [nulls, isc.overall]);

  const cursor = useChartCursor(
    // Its OWN domain. The null is indexed by correlation bin, not by beat, and
    // sharing the "beats" domain would light up bin 3 when the reader hovers
    // beat 3 elsewhere — a correspondence that does not exist.
    "isc-null",
    bins.length,
    "Permutation null distribution of synchrony",
    (i) => {
      const from = lo + ((hi - lo) * i) / BINS;
      const to = lo + ((hi - lo) * (i + 1)) / BINS;
      return `r ${from.toFixed(2)}–${to.toFixed(2)}: ${bins[i]} of ${nulls.length} shuffles`;
    }
  );

  if (!nulls.length) return null;

  const W = 320;
  const H = 96;
  const x = (r: number) => ((r - lo) / (hi - lo)) * W;
  const observedX = x(isc.overall);
  const beaten = nulls.filter((r) => r < isc.overall).length;

  return (
    <div className="rounded-xl border border-hairline bg-surface-2/60 p-3">
      <ChartFrame
        cursor={cursor}
        title="IS THIS BEYOND CHANCE?"
        categories={bins.map((_, i) => `r ${(lo + ((hi - lo) * i) / BINS).toFixed(2)}`)}
        series={[{ key: "null", label: "shuffled", color: "#6f6c6c", values: bins }]}
        footnote={`${isc.n_permutations ?? nulls.length} beat-shuffle permutations · grey = synchrony with the structure destroyed`}
        height={H}
      >
        <svg viewBox={`0 0 ${W} ${H}`} className="h-full w-full" preserveAspectRatio="none">
          {bins.map((count, i) => {
            const bw = W / BINS;
            const bh = (count / peak) * (H - 22);
            return (
              <rect
                key={i}
                x={i * bw + 0.5}
                y={H - 14 - bh}
                width={Math.max(0.5, bw - 1)}
                height={bh}
                fill="#6f6c6c"
                opacity={cursor.index === null ? 0.55 : cursor.isMarked(i) ? 0.95 : 0.22}
              />
            );
          })}

          {/* The observation. Deliberately the only saturated thing on the
              chart — colour is data, and this is the datum. */}
          <line
            x1={observedX}
            y1={4}
            x2={observedX}
            y2={H - 14}
            stroke="#3987e5"
            strokeWidth="2"
          />
          <circle cx={observedX} cy={4} r={3} fill="#3987e5" />
          <text
            x={Math.min(W - 4, Math.max(30, observedX))}
            y={H - 4}
            fontSize="7.5"
            fill="#3987e5"
            textAnchor={observedX > W - 60 ? "end" : "middle"}
            fontFamily="ui-monospace, monospace"
          >
            observed {isc.overall.toFixed(2)}
          </text>
        </svg>
      </ChartFrame>

      <p className="mt-1.5 font-mono text-[9px] leading-relaxed text-muted">
        {/* Stated as a count of shuffles rather than as "p < 0.05", because the
            count is the thing that was actually measured and the threshold is a
            convention laid over it. */}
        beat this audience in {nulls.length - beaten} of {nulls.length} shuffles
        {isc.p_value != null && <> · p = {isc.p_value.toFixed(3)}</>}
        {isc.ci && isc.ci.length === 2 && (
          <>
            {" "}
            · 95% CI [{Number(isc.ci[0]).toFixed(2)}, {Number(isc.ci[1]).toFixed(2)}]
          </>
        )}
        <br />
        {isc.significant ? (
          <span className="text-good">
            The agreement is real — shuffled beats do not reproduce it.
          </span>
        ) : (
          <span className="text-critical">
            Indistinguishable from chance. These twins are not reacting to shared
            structure, so read the attention curve as noise rather than as shape.
          </span>
        )}
      </p>
    </div>
  );
}
