"use client";

/**
 * THE CONTROL ARM — the first thing on the screen, because it decides whether
 * anything below it means anything.
 *
 * A language model asked to deliberate will converge. It will converge on
 * statements it was never shown, on positions nobody argued, and on a motion it
 * has no stake in, because agreeing is what the training distribution rewards.
 * So some replicates are run as PLACEBO: the members are shown statements from a
 * DIFFERENT room and asked to respond. Any convergence there is the model being
 * agreeable, not the room being persuaded.
 *
 * If the placebo arm converged as hard as the real one, this study measured
 * nothing, and the screen says exactly that in the largest type on the page.
 * That is the selling point, not the disclaimer: a room simulator that cannot
 * fail its own control is a random sentence generator with a chart library.
 */

import type { RoomOrderEffect, RoomPlacebo } from "@/lib/api";
import { fmt2, signed } from "./scale";

const CRITICAL = "#f0605f";
const WARN = "#d95926";
const GOOD = "#16d016";

/**
 * DIRECTION, once, because it is the opposite of the intuitive one.
 *
 * A convergence figure here is `sd_final / sd_opening` — the spread REMAINING.
 * 0.2 means the room closed 80% of its disagreement; 1.0 means it closed none.
 * So the damning case is the control arm being LOWER than or equal to the real
 * arm, and the server's `convergence_delta` (placebo − real) is ≤ 0 exactly
 * then. Reading these as "bigger is more converged" inverts every verdict on
 * the screen, which is why the bars below plot the spread CLOSED (1 − ratio)
 * and say so.
 */
function closed(ratio: number | null): number | null {
  if (ratio === null || !Number.isFinite(ratio)) return null;
  return Math.max(0, Math.min(1, 1 - ratio));
}

export default function PlaceboContrast({
  placebo,
  orderEffect,
  placeboReplicates,
}: {
  placebo?: RoomPlacebo | null;
  orderEffect?: RoomOrderEffect | null;
  /** how many control replicates were requested, for the empty state */
  placeboReplicates: number;
}) {
  if (!placebo) {
    return (
      <section
        className="border border-dashed px-4 py-3"
        style={{ borderColor: "rgba(240,96,95,0.45)", background: "rgba(240,96,95,0.05)" }}
        role="note"
      >
        <div className="flex flex-wrap items-baseline gap-x-3">
          <span className="hud-label text-critical">NO CONTROL ARM</span>
          <span className="font-mono text-[10px] text-ink-2">
            {placeboReplicates > 0
              ? "the control replicates did not produce a comparison"
              : "no placebo replicate was run"}
          </span>
        </div>
        <p className="mt-1.5 text-[12px] leading-relaxed text-ink-2">
          Nothing on this screen separates a room that was persuaded from a
          language model that agrees with whatever it is shown. Every convergence
          number below is uncontrolled. Run at least one placebo replicate before
          treating any of it as a measurement.
        </p>
        {orderEffect && <OrderEffect orderEffect={orderEffect} />}
      </section>
    );
  }

  const realClosed = closed(placebo.real_convergence);
  const controlClosed = closed(placebo.placebo_convergence);
  // Prefer the server's own categorical read: it is computed against the 0.05
  // position quantisation, and re-deriving a threshold here is how a screen ends
  // up contradicting the verdict printed two lines below it.
  const sep = (placebo.separation ?? "").toLowerCase();
  const delta = placebo.convergence_delta;
  const damning = sep.startsWith("none")
    ? true
    : sep.startsWith("marginal") || sep.startsWith("clear")
      ? false
      : delta !== null && delta !== undefined
        ? delta <= 0
        : false;
  const partial = sep.startsWith("marginal")
    ? true
    : sep.startsWith("none") || sep.startsWith("clear")
      ? false
      : delta !== null && delta !== undefined && delta > 0 && delta < 0.05;

  const unresolvable = realClosed === null || controlClosed === null;
  const headlineColor = damning ? CRITICAL : partial ? WARN : unresolvable ? WARN : GOOD;
  const headline = damning
    ? "THE DELIBERATION MEASURED NOTHING"
    : partial
      ? "THE ARMS BARELY SEPARATE"
      : unresolvable
        ? "THE ARMS CANNOT BE COMPARED"
        : "THE ROOM DID WORK THE CONTROL DID NOT";

  const gapTop = Math.max(
    0.01,
    Math.abs(placebo.real_gap ?? 0),
    Math.abs(placebo.placebo_gap ?? 0)
  );
  const gapRatio =
    placebo.real_gap !== null && placebo.placebo_gap !== null && Math.abs(placebo.real_gap) > 1e-6
      ? Math.abs(placebo.placebo_gap / placebo.real_gap)
      : null;

  return (
    <section
      className="border"
      style={{
        borderColor: damning ? "rgba(240,96,95,0.5)" : "var(--color-hairline)",
        background: damning ? "rgba(240,96,95,0.06)" : "rgba(0,0,0,0.3)",
      }}
    >
      <header className="flex flex-wrap items-baseline gap-x-3 gap-y-1 border-b border-hairline px-4 py-2">
        <span className="hud-label" style={{ color: "rgb(var(--region-rgb))" }}>
          CONTROL ARM
        </span>
        <span className="font-mono text-[9px] text-muted">
          members shown statements from a different room — any convergence here is
          the model being agreeable
        </span>
      </header>

      <div className="px-4 py-3">
        <p
          className="font-display text-[clamp(1.05rem,2.6vw,1.5rem)] font-semibold leading-tight tracking-tight"
          style={{ color: headlineColor }}
        >
          {headline}
        </p>
        <p className="mt-1 font-mono text-[10px] leading-relaxed text-ink-2">
          {realClosed === null || controlClosed === null ? (
            <span style={{ color: headlineColor }}>
              at least one arm has no opening spread to converge from — there was
              nothing for the deliberation to close
            </span>
          ) : (
            <>
              the real room closed{" "}
              <span style={{ color: headlineColor }}>{Math.round(realClosed * 100)}%</span> of its
              opening disagreement; the control room closed{" "}
              <span style={{ color: headlineColor }}>{Math.round(controlClosed * 100)}%</span>{" "}
              reading statements from a different room entirely
              {delta !== null && delta !== undefined && (
                <span className="text-muted">
                  {" "}
                  · separation {signed(-delta)} on the spread ratio
                </span>
              )}
            </>
          )}
          {placebo.separation && (
            <span className="text-muted"> · {placebo.separation}</span>
          )}
        </p>

        {/* the four numbers, paired, on two shared scales */}
        <div className="mt-3 grid gap-x-6 gap-y-2 sm:grid-cols-2">
          <Pair
            label="DISAGREEMENT CLOSED"
            hint="1 − (final spread ÷ opening spread) — longer bar converged harder"
            real={realClosed}
            control={controlClosed}
            top={1}
            alarm={damning}
            format={(v) => `${Math.round(v * 100)}%`}
          />
          <Pair
            label="SAID − THOUGHT"
            hint="mean public minus private position produced by the arm"
            real={placebo.real_gap}
            control={placebo.placebo_gap}
            top={gapTop}
            alarm={gapRatio !== null && gapRatio >= 0.8}
            format={fmt2}
          />
        </div>

        <p className="verdict mt-3 text-[12px] text-ink-2">
          <span className="hud-label mr-2">VERDICT</span>
          {placebo.verdict}
        </p>

        {/* The gap's own control, in the server's words. A gap that survives in
            the placebo arm is a disposition of the model rather than politics in
            this room, and that distinction is the difference between a finding
            and a party trick. */}
        {placebo.gap_note && (
          <p
            className="mt-1.5 border-l-2 pl-2 text-[11px] leading-relaxed"
            style={{
              borderLeftColor:
                gapRatio !== null && gapRatio >= 0.8 ? "rgba(240,96,95,0.6)" : "var(--color-hairline)",
              color: gapRatio !== null && gapRatio >= 0.8 ? "#f0605f" : "var(--color-muted)",
            }}
          >
            {placebo.gap_note}
          </p>
        )}

        {orderEffect && <OrderEffect orderEffect={orderEffect} />}
      </div>
    </section>
  );
}

function Pair({
  label,
  hint,
  real,
  control,
  top,
  alarm,
  format,
}: {
  label: string;
  hint: string;
  real: number | null;
  control: number | null;
  top: number;
  alarm: boolean;
  format: (v: number) => string;
}) {
  return (
    <div>
      <div className="flex items-baseline gap-2">
        <span className="hud-label">{label}</span>
        <span className="font-mono text-[9px] text-muted">{hint}</span>
      </div>
      {[
        { k: "real room", v: real, color: "#4fb6ff" },
        { k: "control room", v: control, color: alarm ? CRITICAL : "rgba(223,217,217,0.4)" },
      ].map((row) => (
        <div key={row.k} className="mt-1 flex items-center gap-2">
          <span className="w-24 shrink-0 font-mono text-[9px] text-muted">{row.k}</span>
          <span className="h-2.5 flex-1 bg-white/[0.03]">
            {/* An unmeasurable arm is drawn hollow, never as a zero bar: "we
                could not measure it" and "it converged not at all" are
                different findings and must not share a mark. */}
            {row.v === null ? (
              <span className="block h-2.5 w-full border border-dashed border-hairline" />
            ) : (
              <span
                className="block h-2.5"
                style={{
                  width: `${Math.max(1, (Math.abs(row.v) / top) * 100)}%`,
                  background: row.color,
                }}
              />
            )}
          </span>
          <span className="w-12 shrink-0 text-right font-mono text-[10px] tabular-nums text-ink-2">
            {row.v === null ? <span className="text-muted">n/a</span> : format(row.v)}
          </span>
        </div>
      ))}
    </div>
  );
}

/**
 * The other validity threat: if reshuffling who speaks first moves the outcome,
 * then the room's verdict is partly an artefact of the seating, and a single
 * replicate would have reported it as a finding.
 */
function OrderEffect({ orderEffect }: { orderEffect: RoomOrderEffect }) {
  const big = Math.abs(orderEffect.effect_size) >= 0.1;
  const significant = orderEffect.p_value !== null && orderEffect.p_value <= 0.05;
  return (
    <p className="mt-2.5 border-t border-hairline pt-2 font-mono text-[9px] leading-relaxed text-muted">
      <span className="hud-label mr-2">SPEAKING ORDER</span>
      reshuffling who speaks first moved the outcome by{" "}
      <span style={{ color: big && significant ? WARN : "var(--color-ink-2)" }}>
        {signed(orderEffect.effect_size)}
      </span>
      {orderEffect.p_value === null ? (
        <> · no p-value — too few replicates to test it</>
      ) : (
        <> · p = {orderEffect.p_value.toPrecision(2)}</>
      )}
      {big && significant && (
        <span className="text-accent-2">
          {" "}
          — the room&apos;s conclusion depends on who went first, so read the
          consensus as one of several the same cast would have reached
        </span>
      )}
    </p>
  );
}
