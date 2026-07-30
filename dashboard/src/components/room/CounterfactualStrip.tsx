"use client";

/**
 * TAKE ONE PERSON OUT — where does the room land?
 *
 * The most decision-relevant thing a room model can say is not "the room
 * agreed", it is "the room agreed BECAUSE of her". Each row removes one member
 * from the fitted dynamics and re-solves for the fixed point, so the length of
 * the row is that person's authorship of the outcome.
 *
 * The full-room landing is a single rule down the whole strip, which is what
 * makes the rows comparable at a glance rather than eleven separate numbers.
 * When the room has no fixed point the rule is absent and the strip says so —
 * a delta from an anchor that does not exist is not a smaller finding, it is a
 * different one.
 */

import type { RoomCounterfactual, RoomMember } from "@/lib/api";
import type { ChartCursor } from "../charts/cursor";
import { fmt2, identityHue, positionInk, positionWord, signed } from "./scale";

function meanOf(xs: number[] | null | undefined): number | null {
  if (!xs || !xs.length) return null;
  const ok = xs.filter((v) => Number.isFinite(v));
  return ok.length ? ok.reduce((a, b) => a + b, 0) / ok.length : null;
}

export default function CounterfactualStrip({
  cast,
  counterfactuals,
  equilibrium,
  cursor,
}: {
  cast: RoomMember[];
  counterfactuals?: RoomCounterfactual[] | null;
  /** the full room's fixed point, per member — null when it has none */
  equilibrium?: number[] | null;
  cursor: ChartCursor;
}) {
  const fullMean = meanOf(equilibrium ?? null);

  if (!counterfactuals?.length) {
    return (
      <section className="border border-hairline bg-black/40">
        <Header fullMean={fullMean} />
        <p className="px-3 py-3 text-[12px] leading-relaxed text-ink-2">
          <span className="hud-label mr-2 text-critical">NOT COMPUTED</span>
          Removal counterfactuals are solved on the influence fit. This run has
          no fit that supports them, so there is no honest way to say what the
          room would have decided without any particular person.
        </p>
      </section>
    );
  }

  const rows = counterfactuals
    .map((cf) => {
      const idx = cast.findIndex((m) => m.name === cf.name);
      // Prefer the solved landing the server reports; then the mean of a
      // returned equilibrium vector; only then anchor the reported delta on the
      // full room — and never invent a landing with no anchor to hang it on.
      const landed = cf.landing ?? meanOf(cf.equilibrium);
      return {
        cf,
        idx,
        landing: landed ?? (fullMean !== null ? fullMean + cf.room_delta : null),
        solved: landed !== null,
        // The delta is measured against the room WITHOUT this member — the
        // surviving members' mean, which is not the full-room mean. Using the
        // full-room rule as the segment's origin would draw a length that
        // disagrees with the number printed beside it.
        anchor: cf.anchor ?? fullMean,
      };
    })
    .sort((a, b) => Math.abs(b.cf.room_delta) - Math.abs(a.cf.room_delta));

  const maxDelta = Math.max(0.02, ...rows.map((r) => Math.abs(r.cf.room_delta)));

  return (
    <section className="border border-hairline bg-black/40">
      <Header fullMean={fullMean} />

      {fullMean === null && (
        <p className="border-b border-hairline px-3 py-2 font-mono text-[9px] leading-relaxed text-critical">
          The full room has no equilibrium — the influence dynamics do not settle
          — so these are shown as pure deltas around zero rather than as
          landings. &ldquo;The room lands at X&rdquo; is not available here, only
          &ldquo;removing them moves it by X&rdquo;.
        </p>
      )}

      <ul className="divide-y divide-hairline/40">
        {rows.map(({ cf, idx, landing, solved, anchor }) => {
          const marked = idx >= 0 && cursor.isMarked(idx);
          const dim = cursor.index !== null && !marked;
          // With an anchor the track is the 0‥1 motion axis; without one it is a
          // ±maxDelta axis centred on "no change".
          const anchorPct = anchor !== null ? anchor * 100 : 50;
          const landPct =
            fullMean !== null
              ? landing === null
                ? null
                : Math.max(0, Math.min(1, landing)) * 100
              : 50 + (cf.room_delta / maxDelta) * 45;

          return (
            <li
              key={cf.name}
              onPointerEnter={() => idx >= 0 && cursor.setHovered(idx)}
              onPointerLeave={() => cursor.setHovered(null)}
              onClick={() => idx >= 0 && cursor.togglePin(idx)}
              className="cursor-pointer px-3 py-2 transition-colors"
              style={{
                opacity: dim ? 0.42 : 1,
                background: marked ? "rgb(var(--region-rgb) / 0.07)" : undefined,
              }}
            >
              <div className="flex flex-wrap items-baseline gap-x-2">
                <span
                  className="inline-block h-2 w-2 shrink-0"
                  style={{ background: idx >= 0 ? identityHue(idx) : "var(--color-muted)" }}
                  aria-hidden
                />
                <span className="font-mono text-[10px] text-ink-2">without {cf.name}</span>
                <span
                  className={`font-mono text-[10px] tabular-nums ${
                    Math.abs(cf.room_delta) >= 0.1 ? "text-critical" : "text-muted"
                  }`}
                >
                  Δ {signed(cf.room_delta)}
                </span>
                {landing !== null && (
                  <span
                    className="font-mono text-[10px] tabular-nums"
                    style={{ color: positionInk(landing) }}
                  >
                    lands at {fmt2(landing)} — {positionWord(landing)}
                  </span>
                )}
                <span className="ml-auto font-mono text-[9px] text-muted">
                  {cf.note
                    ? cf.note
                    : Math.abs(cf.room_delta) < 0.05
                      ? "no change beyond the position grid — the room decides the same thing without them"
                      : `their absence pulls the room ${
                          cf.room_delta > 0 ? "toward FOR" : "toward AGAINST"
                        }`}
                  {!solved && anchor !== null && " · landing inferred from Δ"}
                </span>
              </div>

              <div className="relative mt-1.5 h-4">
                {/* the track */}
                <div className="absolute inset-x-0 top-2 h-px bg-hairline" />
                {/* the anchor: the full room's landing, or zero change */}
                <div
                  className="absolute top-0 h-4 w-px"
                  style={{
                    left: `${anchorPct}%`,
                    background: "rgba(223,217,217,0.55)",
                  }}
                  aria-hidden
                />
                {/* the move */}
                {landPct !== null && (
                  <div
                    className="absolute top-[7px] h-[3px]"
                    style={{
                      left: `${Math.min(anchorPct, landPct)}%`,
                      width: `${Math.abs(landPct - anchorPct)}%`,
                      background: cf.room_delta > 0 ? "rgba(57,135,229,0.75)" : "rgba(217,89,38,0.75)",
                    }}
                    aria-hidden
                  />
                )}
                {/* where it lands */}
                {landPct !== null && (
                  <div
                    className="absolute top-[3px] h-[11px] w-[3px]"
                    style={{
                      left: `calc(${landPct}% - 1.5px)`,
                      background:
                        landing !== null ? positionInk(landing) : "rgba(223,217,217,0.7)",
                    }}
                    aria-hidden
                  />
                )}
              </div>
            </li>
          );
        })}
      </ul>

      <p className="border-t border-hairline px-3 py-2 font-mono text-[9px] leading-relaxed text-muted">
        Sorted by the size of the move. The bone rule is{" "}
        {fullMean === null
          ? "no change"
          : `the surviving members' mean with that person still in the room`}
        ; the coloured tick is where they settle once that person is not. Each
        row&apos;s baseline excludes the removed member, so the bar length is
        exactly the Δ printed beside it. These are counterfactuals on a FITTED
        model, not re-run rooms — they inherit every limit of the fit above, and
        a Δ under 0.05 is inside the grid a declared position lands on.
      </p>
    </section>
  );
}

function Header({ fullMean }: { fullMean: number | null }) {
  return (
    <header className="flex flex-wrap items-baseline gap-x-3 gap-y-1 border-b border-hairline px-3 py-2">
      <span className="hud-label" style={{ color: "rgb(var(--region-rgb))" }}>
        WHO IS CARRYING THE ROOM
      </span>
      {fullMean !== null ? (
        <span className="font-mono text-[9px] text-muted">
          full room lands at{" "}
          <span style={{ color: positionInk(fullMean) }}>
            {fmt2(fullMean)} — {positionWord(fullMean)}
          </span>
        </span>
      ) : (
        <span className="font-mono text-[9px] text-muted">no full-room equilibrium</span>
      )}
      <span className="ml-auto flex gap-3 font-mono text-[9px] text-muted">
        <span>← against</span>
        <span>for →</span>
      </span>
    </header>
  );
}
