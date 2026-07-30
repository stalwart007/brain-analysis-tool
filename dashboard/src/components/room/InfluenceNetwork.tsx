"use client";

/**
 * WHO MOVES WHOM — the fitted influence network, drawn on the SAME seating plan
 * as the table.
 *
 * Congruent geometry is the whole reason this is readable: the face at the head
 * of the table is at the head of this graph too, so "the CFO is the one everyone
 * bends toward" is a single glance across two pictures rather than a name lookup
 * between them.
 *
 * WHAT THE MARKS MEAN
 *   edge j → i     how much of i's next position is explained by j's last one
 *                  (row i of W is the LISTENER; the column is the source)
 *   node size      Perron centrality — whose opinion the consensus actually
 *                  weights, which is not the same as who talks most
 *   inner disc     self-weight: how much of their next position is their own
 *   dashed node    the row fitted poorly (r² < 0.2); its edges explain little
 *
 * REFUSAL IS A RESULT. When the backend sets `influence.refused` this panel
 * draws NO graph. A network fitted from too few observations is not a blurrier
 * version of a good one — it is a picture of noise with arrows on it, and it is
 * the single most persuasive wrong thing this screen could show. The reason is
 * rendered instead, at the size the graph would have been.
 */

import type { RoomCascades, RoomCentrality, RoomInfluence, RoomMember } from "@/lib/api";
import type { ChartCursor } from "../charts/cursor";
import { keyboardOnlyProps } from "../charts/cursor";
import { TABLE, clip, fmt2, identityHue, labelAnchor, maxOf, seatPlan } from "./scale";

/** Below this the arrow is indistinguishable from the fit's own noise floor. */
const EDGE_FLOOR = 0.03;
const R2_FLOOR = 0.2;

export default function InfluenceNetwork({
  cast,
  influence,
  centrality,
  cascades,
  cursor,
}: {
  cast: RoomMember[];
  influence?: RoomInfluence | null;
  centrality?: RoomCentrality | null;
  cascades?: RoomCascades | null;
  /** the shared "room-members" cursor */
  cursor: ChartCursor;
}) {
  const geo = TABLE;
  const plan = seatPlan(cast, geo);

  /* ── the refusals, first ─────────────────────────────────────────────── */

  if (!influence) {
    return (
      <Frame>
        <Refusal
          title="NOT COMPUTED"
          reason="This run carries no influence fit. Nothing here can say who moved whom."
        />
      </Frame>
    );
  }

  if (influence.refused) {
    return (
      <Frame method={influence.method} nObs={influence.n_obs_per_member}>
        <Refusal title="NOT IDENTIFIABLE" reason={influence.refused} />
      </Frame>
    );
  }

  // Keyboard + ARIA from the shared members cursor. Its `style` resets the
  // outline, which has to be dropped or it beats the focus ring on inline
  // specificity — the graph would then be focusable with nothing to show it.
  const { style: kbStyle, ...kb } = keyboardOnlyProps(cursor);

  const W = influence.W ?? [];
  const names = influence.names?.length ? influence.names : cast.map((m) => m.name);

  /** Map a row/column of W onto a cast index — the backend may order names
   *  differently from the cast, and silently mis-indexing arrows is the kind of
   *  bug that produces a confident, wrong picture. */
  const castIndex = (name: string) => cast.findIndex((m) => m.name === name);
  const rowToCast = names.map(castIndex);

  const edges: { from: number; to: number; w: number }[] = [];
  /** the diagonal of W: how much of a member's next position is still their own */
  const selfWeight: number[] = new Array(cast.length).fill(0);
  W.forEach((row, i) => {
    const target = rowToCast[i];
    if (target < 0 || !Array.isArray(row)) return;
    row.forEach((w, j) => {
      const source = rowToCast[j];
      if (source < 0 || !Number.isFinite(w)) return;
      if (source === target) {
        selfWeight[target] = w;
        return;
      }
      if (Math.abs(w) < EDGE_FLOOR) return;
      edges.push({ from: source, to: target, w });
    });
  });

  const maxW = maxOf(edges.map((e) => Math.abs(e.w)), 0.05);
  /**
   * THE BACKEND'S OWN INSTRUCTION, obeyed rather than noted.
   *
   * `w_entries_supported === false` means the individual entries of W were
   * measured to be worse than a uniform-row guess at this observation count, so
   * no single weight may be quoted as a number. What survived measurement is the
   * ORDERING within a row — who moved this member most, second most — so the
   * arrows stay and the numerals go. A panel that printed "0.42" under that
   * caveat would be quoting a number its own method section forbids.
   */
  const quoteWeights = influence.w_entries_supported !== false;

  const cen = centrality?.centrality ?? null;
  const maxCen = cen && cen.length ? maxOf(cen) : 1;
  const centralityFor = (castI: number) => {
    if (!cen) return null;
    const rowI = rowToCast.indexOf(castI);
    return rowI >= 0 ? cen[rowI] ?? null : null;
  };
  const r2For = (castI: number) => {
    const rowI = rowToCast.indexOf(castI);
    return rowI >= 0 ? influence.r2_per_member?.[rowI] ?? null : null;
  };
  const susceptibilityFor = (castI: number) => {
    const rowI = rowToCast.indexOf(castI);
    return rowI >= 0 ? influence.susceptibility?.[rowI] ?? null : null;
  };
  const cascadeFor = (castI: number) =>
    cascades?.per_member?.find((c) => c.name === cast[castI]?.name)?.score ?? null;

  const nodeR = (castI: number) => {
    const c = centralityFor(castI);
    if (c === null) return 17;
    return 11 + (Math.max(0, c) / maxCen) * 15;
  };

  const focus = cursor.index;
  const weakRows = cast
    .map((m, i) => ({ m, r2: r2For(i) }))
    .filter((x) => x.r2 !== null && (x.r2 as number) < R2_FLOOR);
  /** Rows the server itself declared unidentifiable: the regressors that member
   *  heard were collinear, so WHO moved them is not separable at any round
   *  count. Stronger than the r² heuristic, and it comes with a reason. */
  const unidentified = new Map(
    (influence.unidentified_rows ?? []).map((r) => [r.name, r.reason])
  );

  /* ── the readout for the focused member ─────────────────────────────── */

  const focusedIn =
    focus === null
      ? []
      : edges
          .filter((e) => e.to === focus)
          .sort((a, b) => Math.abs(b.w) - Math.abs(a.w))
          .slice(0, 3);
  const focusedOut =
    focus === null
      ? []
      : edges
          .filter((e) => e.from === focus)
          .sort((a, b) => Math.abs(b.w) - Math.abs(a.w))
          .slice(0, 3);

  return (
    <Frame
      method={influence.method}
      nObs={influence.n_obs_per_member}
      right={
        centrality ? (
          <span className="font-mono text-[9px] text-muted">
            {centrality.spectral_gap !== null && <>spectral gap {fmt2(centrality.spectral_gap)} · </>}
            {centrality.converges ? (
              <>
                converges
                {centrality.consensus_forecast !== null && (
                  <> → {fmt2(centrality.consensus_forecast)}</>
                )}
              </>
            ) : (
              <span className="text-critical">no single consensus</span>
            )}
          </span>
        ) : (
          <span className="font-mono text-[9px] text-muted">
            centrality not computed — nodes are uniform
          </span>
        )
      }
    >
      {/* The caveat the fit itself carries. Printed ABOVE the picture, because a
          reader who has already read a weight off an arrow cannot un-read it. */}
      {!quoteWeights && (
        <div
          className="border-b px-3 py-2"
          style={{ borderColor: "rgba(217,89,38,0.35)", background: "rgba(217,89,38,0.05)" }}
          role="note"
        >
          <span className="hud-label" style={{ color: "#d95926" }}>
            PATTERN ONLY — NO WEIGHT IS QUOTED
          </span>
          <p className="mt-1 font-mono text-[9px] leading-relaxed text-ink-2">
            {influence.w_entry_caveat ??
              "At this observation count the individual entries of W were measured to be worse than a uniform-row guess. The row ordering, the susceptibilities and the leave-one-out deltas did recover."}
          </p>
        </div>
      )}

      {/* No unique centrality is a room that has SPLIT, not a failed
          computation, and the estimator says which. */}
      {centrality?.note && (
        <p className="border-b border-hairline px-3 py-2 font-mono text-[9px] leading-relaxed text-muted">
          {centrality.note}
        </p>
      )}

      <div className="overflow-x-auto">
        <svg
          {...kb}
          viewBox={`0 0 ${geo.w} ${geo.h}`}
          className="h-auto w-full min-w-[560px] focus-visible:outline focus-visible:outline-1 focus-visible:outline-offset-[-2px]"
          style={{
            touchAction: kbStyle.touchAction,
            outlineColor: "rgb(var(--region-rgb) / 0.7)",
          }}
        >
          {/* the table's footprint, so the two pictures are the same room */}
          <ellipse
            cx={geo.cx}
            cy={geo.cy}
            rx={geo.tableRx}
            ry={geo.tableRy}
            fill="none"
            stroke="rgba(223,217,217,0.06)"
            strokeDasharray="3 5"
          />

          {edges.map((e, k) => {
            const s = plan[e.from];
            const t = plan[e.to];
            if (!s || !t) return null;
            const mag = Math.abs(e.w) / maxW;
            const involved = focus === null || e.from === focus || e.to === focus;
            // Direction becomes colour only when a member is focused — then it
            // answers the two questions worth asking: who moves them, and whom
            // do they move.
            const stroke =
              focus === null
                ? "rgb(var(--region-rgb))"
                : e.from === focus
                  ? "#3987e5"
                  : e.to === focus
                    ? "#d95926"
                    : "rgb(var(--region-rgb))";
            const bow = 0.26;
            const cx = s.x + (t.x - s.x) / 2 + (geo.cx - (s.x + t.x) / 2) * bow;
            const cy = s.y + (t.y - s.y) / 2 + (geo.cy - (s.y + t.y) / 2) * bow;
            const dx = t.x - cx;
            const dy = t.y - cy;
            const len = Math.hypot(dx, dy) || 1;
            const ux = dx / len;
            const uy = dy / len;
            const rT = nodeR(e.to);
            const tipX = t.x - ux * (rT + 2);
            const tipY = t.y - uy * (rT + 2);
            const backX = tipX - ux * 8;
            const backY = tipY - uy * 8;
            const px = -uy * 3.6;
            const py = ux * 3.6;
            const opacity = involved ? 0.2 + mag * 0.65 : 0.05;

            return (
              <g key={k} opacity={opacity} style={{ transition: "opacity 0.18s ease" }}>
                <path
                  d={`M ${s.x.toFixed(1)} ${s.y.toFixed(1)} Q ${cx.toFixed(1)} ${cy.toFixed(
                    1
                  )} ${tipX.toFixed(1)} ${tipY.toFixed(1)}`}
                  fill="none"
                  stroke={stroke}
                  strokeWidth={0.6 + mag * 3.2}
                />
                <polygon
                  points={`${tipX.toFixed(1)},${tipY.toFixed(1)} ${(backX + px).toFixed(1)},${(
                    backY + py
                  ).toFixed(1)} ${(backX - px).toFixed(1)},${(backY - py).toFixed(1)}`}
                  fill={stroke}
                />
              </g>
            );
          })}

          {cast.map((m, i) => {
            const seat = plan[i];
            if (!seat) return null;
            const lab = labelAnchor(seat);
            const r = nodeR(i);
            const marked = cursor.isMarked(i);
            const dim = focus !== null && !marked;
            const r2 = r2For(i);
            const notSeparable = unidentified.has(m.name);
            const weak = notSeparable || (r2 !== null && r2 < R2_FLOOR);
            const self = Math.max(0, Math.min(1, selfWeight[i] ?? 0));
            const c = centralityFor(i);

            return (
              <g
                key={m.name}
                onPointerEnter={() => cursor.setHovered(i)}
                onPointerLeave={() => cursor.setHovered(null)}
                onClick={() => cursor.togglePin(i)}
                style={{
                  cursor: "pointer",
                  opacity: dim ? 0.34 : 1,
                  transition: "opacity 0.18s ease",
                }}
              >
                <circle
                  cx={seat.x}
                  cy={seat.y}
                  r={r}
                  fill="rgb(var(--region-rgb) / 0.10)"
                  stroke={marked ? "rgba(223,217,217,0.8)" : "rgb(var(--region-rgb) / 0.5)"}
                  strokeWidth={marked ? 1.6 : 1}
                  strokeDasharray={weak ? "3 3" : undefined}
                />
                {/* self-weight: how much of them is still them */}
                {self > 0.02 && (
                  <circle
                    cx={seat.x}
                    cy={seat.y}
                    r={Math.max(1.5, r * Math.sqrt(self))}
                    fill="rgb(var(--region-rgb) / 0.42)"
                  />
                )}
                <rect
                  x={lab.anchor === "end" ? lab.x - 5 : lab.anchor === "start" ? lab.x : lab.x - 2.5}
                  y={lab.y - 12}
                  width={5}
                  height={5}
                  fill={identityHue(i)}
                  aria-hidden
                />
                <text
                  x={lab.x + (lab.anchor === "start" ? 9 : lab.anchor === "end" ? -9 : 0)}
                  y={lab.y}
                  textAnchor={lab.anchor}
                  style={{
                    fontSize: 10.5,
                    fontFamily: "var(--font-mono)",
                    fill: marked ? "var(--color-ink)" : "var(--color-ink-2)",
                  }}
                >
                  {clip(m.name, 16)}
                </text>
                <text
                  x={lab.x + (lab.anchor === "start" ? 9 : lab.anchor === "end" ? -9 : 0)}
                  y={lab.y + 11}
                  textAnchor={lab.anchor}
                  style={{ fontSize: 9, fontFamily: "var(--font-mono)", fill: "var(--color-muted)" }}
                >
                  {c === null ? "weight n/a" : `weight ${fmt2(c)}`}
                  {notSeparable ? " · not separable" : weak ? " · weak fit" : ""}
                </text>
              </g>
            );
          })}
        </svg>
      </div>

      {/* Reserved height, not grown: measured at 56px idle → 86px focused, so
          every hover pushed the cascade strip and the whole lower page down.
          The taller reserve is used only when this run HAS unidentified rows,
          because that is the only case where the extra reason line appears for
          some members and not others — sizing for it unconditionally would leave
          dead space on every run that fitted cleanly. */}
      <div
        className={`border-t border-hairline px-3 py-2 ${
          unidentified.size > 0 ? "min-h-[7.5rem]" : "min-h-[5.5rem]"
        }`}
      >
        {focus !== null && cast[focus] ? (
          // tabular figures: consensus weight, susceptibility, own weight, r²
          // and the cascade score all change as the cursor moves between faces
          <div className="font-mono text-[10px] leading-relaxed tabular-nums">
            <div className="flex flex-wrap items-baseline gap-x-2">
              <span
                className="inline-block h-2 w-2"
                style={{ background: identityHue(focus) }}
                aria-hidden
              />
              <span className="text-ink">{cast[focus].name}</span>
              <span className="text-muted">
                consensus weight{" "}
                <span className="text-ink-2">
                  {centralityFor(focus) === null ? "n/a" : fmt2(centralityFor(focus) as number)}
                </span>{" "}
                · susceptibility{" "}
                <span className="text-ink-2">
                  {susceptibilityFor(focus) === null
                    ? "n/a"
                    : fmt2(susceptibilityFor(focus) as number)}
                </span>{" "}
                {quoteWeights && (
                  <>
                    {" "}
                    · own weight <span className="text-ink-2">{fmt2(selfWeight[focus] ?? 0)}</span>
                  </>
                )}{" "}
                · r²{" "}
                <span className={r2For(focus) !== null && (r2For(focus) as number) < R2_FLOOR ? "text-critical" : "text-ink-2"}>
                  {r2For(focus) === null ? "n/a" : fmt2(r2For(focus) as number)}
                </span>
                {cascadeFor(focus) !== null && (
                  <>
                    {" "}
                    · cascade score <span className="text-ink-2">{fmt2(cascadeFor(focus) as number)}</span>
                  </>
                )}
              </span>
            </div>
            {unidentified.has(cast[focus].name) && (
              <p className="mt-1 border-l-2 border-dashed border-critical/50 pl-2 text-critical">
                {unidentified.get(cast[focus].name)}
              </p>
            )}
            {/* When the weights are not quotable, this is an ORDERED LIST and
                nothing more — which is exactly what the fit supports. */}
            <p className="mt-1 text-muted">
              <span style={{ color: "#d95926" }}>moved by</span>{" "}
              {focusedIn.length
                ? focusedIn
                    .map((e) =>
                      quoteWeights
                        ? `${cast[e.from]?.name ?? "?"} ${fmt2(e.w)}`
                        : (cast[e.from]?.name ?? "?")
                    )
                    .join(quoteWeights ? " · " : " › ")
                : "nobody above the noise floor"}
            </p>
            <p className="text-muted">
              <span style={{ color: "#3987e5" }}>moves</span>{" "}
              {focusedOut.length
                ? focusedOut
                    .map((e) =>
                      quoteWeights
                        ? `${cast[e.to]?.name ?? "?"} ${fmt2(e.w)}`
                        : (cast[e.to]?.name ?? "?")
                    )
                    .join(quoteWeights ? " · " : " › ")
                : "nobody above the noise floor"}
            </p>
            {!quoteWeights && (
              <p className="text-muted">
                listed strongest first · weights withheld, see the caveat above
              </p>
            )}
          </div>
        ) : (
          <p className="font-mono text-[10px] leading-relaxed text-muted">
            Arrows run from the speaker to the person they moved; width is the
            fitted weight. Node size is Perron centrality — how heavily the
            consensus weights that person, which is a different question from how
            often they spoke. The inner disc is how much of their next position is
            still their own. Hover a face to separate{" "}
            <span style={{ color: "#d95926" }}>who moves them</span> from{" "}
            <span style={{ color: "#3987e5" }}>whom they move</span>.
            {weakRows.length > 0 && (
              <>
                {" "}
                <span className="text-critical">
                  {weakRows.map((x) => x.m.name).join(", ")}{" "}
                  {weakRows.length === 1 ? "fits" : "fit"} poorly (r² &lt; {R2_FLOOR})
                </span>{" "}
                — drawn dashed, and their arrows should be read as gestures, not
                measurements.
              </>
            )}
            {edges.length === 0 && (
              <>
                {" "}
                <span className="text-critical">
                  No weight in this fit clears {EDGE_FLOOR} — the room is drawn
                  with no arrows because nobody measurably moved anybody.
                </span>
              </>
            )}
          </p>
        )}
      </div>

      {cascades?.per_member?.length ? (
        <div className="border-t border-hairline px-3 py-2">
          <span className="hud-label">CASCADE EXPOSURE</span>
          <span className="ml-2 font-mono text-[9px] text-muted">
            how much of the room moves when this one person moves first
          </span>
          <div className="mt-1.5 space-y-1">
            {[...cascades.per_member]
              .sort((a, b) => b.score - a.score)
              .map((c) => {
                const i = cast.findIndex((m) => m.name === c.name);
                const top = maxOf(cascades.per_member.map((x) => x.score));
                const marked = i >= 0 && cursor.isMarked(i);
                return (
                  <div
                    key={c.name}
                    onPointerEnter={() => i >= 0 && cursor.setHovered(i)}
                    onPointerLeave={() => cursor.setHovered(null)}
                    onClick={() => i >= 0 && cursor.togglePin(i)}
                    className="flex cursor-pointer items-center gap-2"
                  >
                    <span
                      className="w-28 shrink-0 truncate font-mono text-[9px]"
                      style={{ color: marked ? "var(--color-ink)" : "var(--color-muted)" }}
                    >
                      {c.name}
                    </span>
                    <span className="h-2 flex-1 bg-white/[0.03]">
                      <span
                        className="block h-2"
                        style={{
                          width: `${Math.max(1, (c.score / top) * 100)}%`,
                          background: "rgb(var(--region-rgb) / 0.6)",
                          opacity: focus === null || marked ? 1 : 0.3,
                        }}
                      />
                    </span>
                    <span className="w-9 shrink-0 text-right font-mono text-[9px] tabular-nums text-ink-2">
                      {fmt2(c.score)}
                    </span>
                  </div>
                );
              })}
          </div>
        </div>
      ) : (
        <p className="border-t border-hairline px-3 py-2 font-mono text-[9px] text-muted">
          cascade scores not computed for this run
        </p>
      )}
    </Frame>
  );
}

/* ── chrome ────────────────────────────────────────────────────────────── */

function Frame({
  children,
  method,
  nObs,
  right,
}: {
  children: React.ReactNode;
  method?: string;
  nObs?: number;
  right?: React.ReactNode;
}) {
  return (
    <section className="border border-hairline bg-black/40">
      <header className="flex flex-wrap items-baseline gap-x-3 gap-y-1 border-b border-hairline px-3 py-2">
        <span className="hud-label" style={{ color: "rgb(var(--region-rgb))" }}>
          WHO MOVES WHOM
        </span>
        {nObs !== undefined && (
          <span className="font-mono text-[9px] tabular-nums text-muted">
            {nObs} observation{nObs === 1 ? "" : "s"} per member
          </span>
        )}
        <span className="ml-auto">{right}</span>
      </header>
      {children}
      {method && (
        <p className="border-t border-hairline px-3 py-2 font-mono text-[9px] leading-relaxed text-muted">
          {method}
        </p>
      )}
    </section>
  );
}

function Refusal({ title, reason }: { title: string; reason: string }) {
  return (
    <div
      className="m-3 border border-dashed p-4"
      style={{ borderColor: "rgba(240,96,95,0.45)" }}
      role="note"
    >
      <div className="flex items-baseline gap-2">
        <span className="font-mono text-[10px] text-critical" aria-hidden>
          ▲
        </span>
        <span className="hud-label text-critical">{title}</span>
      </div>
      <p className="mt-1.5 text-[12px] leading-relaxed text-ink-2">{reason}</p>
      <p className="mt-2 font-mono text-[9px] leading-relaxed text-muted">
        No network is drawn from a refused fit. An influence graph estimated from
        too little data is not a rougher answer — it is a picture of noise with
        arrows on it, and it would be the most convincing wrong thing on this
        screen. Add rounds or replicates and the fit becomes identifiable.
      </p>
    </div>
  );
}
