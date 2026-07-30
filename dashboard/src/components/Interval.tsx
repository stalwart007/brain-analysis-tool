"use client";

/**
 * A confidence interval, or a stated reason there is not one.
 *
 * WHY THIS IS SHARED AND WHY IT MATTERS. `bootstrap_ci` was tightened to refuse
 * samples below eight — at n = 2 it used to return the sample RANGE and call it
 * 95%, whose measured coverage on Exponential(1) draws is 46.5%. That guard was
 * right, and it left roughly a dozen call sites returning `null` where they had
 * previously returned numbers, several of them on paths a single-persona run
 * reaches every time.
 *
 * Nothing crashed, which is the problem. Two of the three places that render an
 * interval used `{ci && …}`, so the line simply vanished; the third said
 * "interval n/a". Both read as A FIELD IS MISSING — a rendering gap, something
 * to go and look for in the payload — when the truth is a finding about the
 * run: it was too small to put an interval on. Those are opposite messages. One
 * sends a reader to check for a bug; the other tells them to add twins.
 *
 * `null` from `bootstrap_ci` has exactly ONE cause (`n < MIN_BOOTSTRAP_N`), so
 * this can name the reason precisely without being told the sample size.
 */

/** Mirrors `analytics.MIN_BOOTSTRAP_N`. Duplicated rather than plumbed through
 *  every payload because it is a property of the estimator, not of a run —
 *  and a wrong number here understates or overstates by a couple either way,
 *  where plumbing it would touch fifteen server call sites to say the same
 *  thing. Pinned by a test on both sides. */
export const MIN_BOOTSTRAP_N = 8;

export default function Interval({
  ci,
  /** Shown instead of the default when the caller knows something better —
   *  e.g. a bracket that is identified but not estimable. */
  absentReason,
}: {
  ci?: [number, number] | number[] | null;
  absentReason?: string;
}) {
  if (!ci || ci.length !== 2) {
    return (
      <span
        className="text-muted"
        title={
          absentReason ??
          `A bootstrap interval needs at least ${MIN_BOOTSTRAP_N} observations to ` +
            `resample. Below that the interval it would produce is not one — the ` +
            `estimator refuses rather than reporting a range as if it were 95% ` +
            `coverage. Run more twins for an interval on this number.`
        }
      >
        {" "}
        · <span className="underline decoration-dotted">no interval — under {MIN_BOOTSTRAP_N} observations</span>
      </span>
    );
  }
  const [lo, hi] = ci;
  if (hi - lo <= 0) {
    // A real, meaningful state and NOT a tight interval: every observation was
    // identical, so the bootstrap has no sampling spread to report at all.
    return (
      <span
        className="text-muted"
        title="Every observation was identical, so the bootstrap has no sampling spread to report — this is not a tight interval."
      >
        {" "}· no observed spread
      </span>
    );
  }
  return (
    <span className="text-muted">
      {" "}
      · [{lo.toFixed(3)}, {hi.toFixed(3)}]
    </span>
  );
}
