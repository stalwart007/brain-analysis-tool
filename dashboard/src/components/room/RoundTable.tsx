"use client";

/**
 * THE TABLE — the product's best moment, and the reason this screen exists.
 *
 * Every seat carries two numbers. The FILL is what that person said out loud;
 * the RING around it is what they actually think. So the visible distance
 * between ring and fill IS their preference falsification, per face, without a
 * table of numbers and without reading anything. A room where the rings and the
 * fills agree is a room that meant it. A room where the ring is orange inside a
 * wall of blue fills is a room that just voted for something nobody wants.
 *
 * The ordering is honest and it is also the drama: private positions are NOT
 * streamed. During the run you can only watch the public conversation move —
 * rings sit dashed and empty, labelled "private held" — and the reveal lands
 * only when the terminal result arrives. The screen cannot show what it has not
 * been told, and pretending otherwise would be the one lie this whole product
 * is built to avoid.
 *
 * SYSTEM COMPLIANCE
 * · Colour is data: fill and ring are positions on one diverging scale anchored
 *   at 0.5. Chrome is bone/void plus the region's tint. Identity is a 5px chip.
 * · Never colour alone: the gap is ALSO an arc length on the ring, a signed
 *   number under the name, and a sentence in the ledger below.
 * · Zero radius on every instrument surface; the table itself is tissue, so it
 *   is the only curve in here.
 * · One clock: the speaking seat's halo is driven from --pulse, published by
 *   the shared physiological clock. There is no timer in this file.
 */

import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import type { CSSProperties } from "react";
import type { RoomMember } from "@/lib/api";
import type { ChartCursor } from "../charts/cursor";
import { keyboardOnlyProps } from "../charts/cursor";
import {
  TABLE,
  clip,
  fmt2,
  gapArc,
  gapWord,
  identityHue,
  labelAnchor,
  positionFill,
  positionInk,
  positionStroke,
  positionWord,
  seatPlan,
  signed,
} from "./scale";

/**
 * The three states a seat can be in, drawn at seat scale so the reader learns
 * the idiom from the picture rather than from the caption. The middle glyph is
 * the one that matters: a seat that says FOR while thinking AGAINST.
 */
function SeatLegend({ revealed }: { revealed: boolean }) {
  const glyphs: { pub: number | null; prv: number | null; caption: string }[] = [
    { pub: 0.86, prv: 0.82, caption: "means it — ring and fill agree" },
    { pub: 0.84, prv: 0.18, caption: "says for, thinks against — the arc is the gap" },
    revealed
      ? { pub: 0.5, prv: 0.5, caption: "undecided, and says so" }
      : { pub: 0.72, prv: null, caption: "private position not streamed yet" },
  ];
  return (
    <div className="flex flex-wrap items-center gap-x-5 gap-y-1 border-b border-hairline px-3 py-1.5">
      {glyphs.map((g, i) => {
        const seat = { x: 17, y: 17, r: 9, angle: 0, member: 0, side: "right" as const };
        const gap = g.pub !== null && g.prv !== null ? g.prv - g.pub : null;
        return (
          <span key={i} className="flex items-center gap-1.5">
            <svg viewBox="0 0 34 34" className="h-[26px] w-[26px] shrink-0" aria-hidden>
              {g.prv === null ? (
                <circle
                  cx={17}
                  cy={17}
                  r={13}
                  fill="none"
                  stroke="rgba(223,217,217,0.20)"
                  strokeDasharray="2 4"
                />
              ) : (
                <>
                  <circle cx={17} cy={17} r={13} fill="none" stroke={positionStroke(g.prv)} strokeWidth={2.2} />
                  {gap !== null && Math.abs(gap) >= 0.02 && (
                    <path
                      d={gapArc(seat, gap, 13)}
                      fill="none"
                      stroke={positionStroke(g.prv)}
                      strokeWidth={4}
                    />
                  )}
                </>
              )}
              {g.pub !== null && (
                <circle
                  cx={17}
                  cy={17}
                  r={9}
                  fill={positionFill(g.pub)}
                  stroke="rgba(223,217,217,0.28)"
                />
              )}
            </svg>
            <span className="font-mono text-[9px] leading-tight text-muted">{g.caption}</span>
          </span>
        );
      })}
    </div>
  );
}

export interface SeatDatum {
  /** index into the cast, and the members-cursor index */
  index: number;
  member: RoomMember;
  /** what they last said out loud; null before they have spoken */
  publicPosition: number | null;
  /** null while the run is in flight — the stream does not carry it */
  privatePosition: number | null;
  /** their place in this replicate's speaking order, 1-based */
  speakingIndex: number | null;
  /** talking right now */
  speaking: boolean;
  /**
   * TRUE while the shown position is the member's SEALED BALLOT — round 0, when
   * they are alone with the motion and nobody has spoken. It is a position on
   * the motion, so it fills the seat, but it is not something they said, and the
   * label says which.
   */
  ballot: boolean;
  statement: string | null;
  note: string | null;
  conceded: string[];
}

const CAMP_STROKE = "rgba(223,217,217,0.22)";

export default function RoundTable({
  seats,
  cursor,
  revealed,
  live,
  phase,
  armLabel,
  batonFrom,
  camps,
  roomGap,
}: {
  /** one per cast member, in cast order */
  seats: SeatDatum[];
  /** the shared "room-members" cursor — hovering a seat pins that member in
   *  every other chart on this screen */
  cursor: ChartCursor;
  /** the terminal result has landed and private positions are known */
  revealed: boolean;
  live: boolean;
  /** e.g. "replicate 2 · round 3" */
  phase: string;
  /** e.g. "REAL ARM" / "PLACEBO ARM" */
  armLabel: string;
  /** who spoke immediately before the current speaker, for the baton */
  batonFrom: number | null;
  camps: { members: string[]; mean_position: number }[] | null;
  /** the room's mean falsification gap, once known */
  roomGap: number | null;
}) {
  const reduce = useReducedMotion();
  const cast = seats.map((s) => s.member);
  const plan = seatPlan(cast);
  const geo = TABLE;
  // Keyboard + ARIA come from the shared cursor; its `style` carries
  // touch-action and an outline reset, and the reset has to be dropped here or
  // it wins over the focus ring by inline specificity.
  const { style: kbStyle, ...kb } = keyboardOnlyProps(cursor);

  const spoken = seats.filter((s) => s.publicPosition !== null);
  const roomPublic = spoken.length
    ? spoken.reduce((a, s) => a + (s.publicPosition ?? 0), 0) / spoken.length
    : null;

  const focus = cursor.index;
  const focused = focus !== null ? seats[focus] : null;

  const speaker = seats.find((s) => s.speaking) ?? null;
  const baton =
    live && speaker && batonFrom !== null && batonFrom !== speaker.index
      ? { from: plan[batonFrom], to: plan[speaker.index] }
      : null;

  // The falsification ledger: the ring/fill idiom said in words, sorted by how
  // badly each person is lying. This is the accessible equivalent of the seat
  // geometry, and it is also the table most researchers will screenshot.
  const ledger = seats
    .filter((s) => s.privatePosition !== null && s.publicPosition !== null)
    .map((s) => ({
      seat: s,
      // public − private, the server's convention. Positive is public support
      // with nothing behind it.
      gap: (s.publicPosition as number) - (s.privatePosition as number),
    }))
    .sort((a, b) => Math.abs(b.gap) - Math.abs(a.gap));

  return (
    <section className="border border-hairline bg-black/40">
      <header className="flex flex-wrap items-baseline gap-x-3 gap-y-1 border-b border-hairline px-3 py-2">
        <span className="hud-label" style={{ color: "rgb(var(--region-rgb))" }}>
          THE ROOM
        </span>
        <span className="flex items-center gap-1.5 font-mono text-[9px] uppercase tracking-[0.14em] text-muted">
          <span
            className={`neon-dot inline-block h-1.5 w-1.5 rounded-full ${
              live ? "pulse-dot bg-accent" : revealed ? "bg-good" : "bg-muted"
            }`}
          />
          {armLabel}
        </span>
        <span className="font-mono text-[9px] tabular-nums text-muted">{phase}</span>
        <span className="ml-auto flex flex-wrap items-center gap-x-3 gap-y-0.5 font-mono text-[9px] text-muted">
          <span>
            <span style={{ color: "rgba(57,135,229,0.95)" }}>■</span> for ·{" "}
            <span style={{ color: "rgba(217,89,38,0.95)" }}>■</span> against
          </span>
          <span>fill = said · ring = thought</span>
        </span>
      </header>

      {/* The idiom, taught before it has to be read. Three glyphs is cheaper
          than a paragraph and it is the only way the ring/fill distance becomes
          self-evident on first sight. */}
      <SeatLegend revealed={revealed} />

      {/* The SVG scrolls inside its own box on a narrow screen. The page never
          scrolls sideways — that is a hard constraint of the system. */}
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
          {/* ── the table top ─────────────────────────────────────────── */}
          <ellipse
            cx={geo.cx}
            cy={geo.cy}
            rx={geo.tableRx}
            ry={geo.tableRy}
            fill="rgb(var(--region-rgb) / 0.05)"
            stroke="rgb(var(--region-rgb) / 0.30)"
            strokeWidth={1}
          />
          <ellipse
            cx={geo.cx}
            cy={geo.cy}
            rx={geo.tableRx - 11}
            ry={geo.tableRy - 8}
            fill="none"
            stroke="rgba(223,217,217,0.08)"
            strokeWidth={1}
          />

          {/* camp arcs: which seats are sitting together, drawn as a chord
              through the members of each resolved coalition */}
          {camps?.map((camp, ci) => {
            const pts = camp.members
              .map((name) => seats.findIndex((s) => s.member.name === name))
              .filter((i) => i >= 0)
              .map((i) => plan[i]);
            if (pts.length < 2) return null;
            const d = pts
              .map((p, k) => `${k === 0 ? "M" : "L"} ${p.x.toFixed(1)} ${p.y.toFixed(1)}`)
              .join(" ");
            return (
              <path
                key={`camp-${ci}`}
                d={d}
                fill="none"
                stroke={CAMP_STROKE}
                strokeWidth={1}
                strokeDasharray="2 4"
                aria-hidden
              />
            );
          })}

          {/* ── where the room stands, in the middle of the table ────── */}
          {roomPublic !== null && (
            <g aria-hidden>
              <text
                x={geo.cx}
                y={geo.cy - 6}
                textAnchor="middle"
                className="font-display"
                style={{ fontSize: 30, fill: positionInk(roomPublic), fontWeight: 600 }}
              >
                {fmt2(roomPublic)}
              </text>
              <text
                x={geo.cx}
                y={geo.cy + 11}
                textAnchor="middle"
                style={{
                  fontSize: 9,
                  fill: "var(--color-muted)",
                  letterSpacing: "0.16em",
                  textTransform: "uppercase",
                  fontFamily: "var(--font-mono)",
                }}
              >
                stated · {positionWord(roomPublic)}
              </text>
              {revealed && roomGap !== null && (
                <text
                  x={geo.cx}
                  y={geo.cy + 30}
                  textAnchor="middle"
                  style={{
                    fontSize: 9.5,
                    fill: Math.abs(roomGap) >= 0.15 ? "#f0605f" : "var(--color-ink-2)",
                    fontFamily: "var(--font-mono)",
                  }}
                >
                  room gap {signed(roomGap)}
                </text>
              )}
            </g>
          )}

          {/* ── the baton: who is speaking, and who they answer ──────── */}
          <AnimatePresence>
            {baton && (
              <motion.line
                key={`baton-${baton.from.member}-${baton.to.member}`}
                x1={baton.from.x}
                y1={baton.from.y}
                x2={baton.to.x}
                y2={baton.to.y}
                stroke="rgb(var(--region-rgb) / 0.55)"
                strokeWidth={1}
                strokeDasharray="3 3"
                initial={reduce ? { opacity: 0.5 } : { pathLength: 0, opacity: 0.9 }}
                animate={{ pathLength: 1, opacity: 0.5 }}
                exit={{ opacity: 0 }}
                transition={{ duration: reduce ? 0 : 0.45, ease: "easeOut" }}
              />
            )}
          </AnimatePresence>

          {/* ── the seats ────────────────────────────────────────────── */}
          {seats.map((s) => {
            const seat = plan[s.index];
            const lab = labelAnchor(seat);
            const marked = cursor.isMarked(s.index);
            const dim = focus !== null && !marked;
            const pub = s.publicPosition;
            const prv = s.privatePosition;
            const gap = pub !== null && prv !== null ? pub - prv : null;
            const ringR = seat.r + 7;
            const circumference = 2 * Math.PI * ringR;

            return (
              <g
                key={s.member.name}
                onPointerEnter={() => cursor.setHovered(s.index)}
                onPointerLeave={() => cursor.setHovered(null)}
                onClick={() => cursor.togglePin(s.index)}
                style={{
                  cursor: "pointer",
                  opacity: dim ? 0.32 : 1,
                  transition: "opacity 0.18s ease",
                }}
              >
                {/* the speaking halo — one clock, no timer: its strength is the
                    shared cardiac envelope published on <html> */}
                {s.speaking && (
                  <circle
                    cx={seat.x}
                    cy={seat.y}
                    r={seat.r + 15}
                    fill="none"
                    stroke="rgb(var(--region-rgb))"
                    style={
                      {
                        opacity: "calc(0.18 + var(--pulse) * 0.62)",
                        strokeWidth: "calc(1px + var(--pulse) * 2.4px)",
                      } as CSSProperties
                    }
                    aria-hidden
                  />
                )}

                {/* PRIVATE — the ring. Dashed and colourless until the reveal:
                    "we have not been told" must not look like "neutral". */}
                {prv === null ? (
                  <circle
                    cx={seat.x}
                    cy={seat.y}
                    r={ringR}
                    fill="none"
                    stroke="rgba(223,217,217,0.20)"
                    strokeWidth={1}
                    strokeDasharray="2 4"
                  />
                ) : (
                  <motion.circle
                    cx={seat.x}
                    cy={seat.y}
                    r={ringR}
                    fill="none"
                    stroke={positionStroke(prv)}
                    strokeWidth={2.5}
                    strokeDasharray={circumference}
                    initial={reduce ? { strokeDashoffset: 0 } : { strokeDashoffset: circumference }}
                    animate={{ strokeDashoffset: 0 }}
                    transition={{
                      duration: reduce ? 0 : 0.7,
                      delay: reduce ? 0 : 0.12 + s.index * 0.07,
                      ease: "easeOut",
                    }}
                  />
                )}

                {/* the gap, as an ARC LENGTH on that ring */}
                {gap !== null && Math.abs(gap) >= 0.02 && (
                  <motion.path
                    d={gapArc(seat, gap, ringR)}
                    fill="none"
                    stroke={positionStroke(prv as number)}
                    strokeWidth={5}
                    strokeLinecap="butt"
                    initial={reduce ? { pathLength: 1 } : { pathLength: 0 }}
                    animate={{ pathLength: 1 }}
                    transition={{
                      duration: reduce ? 0 : 0.5,
                      delay: reduce ? 0 : 0.3 + s.index * 0.07,
                    }}
                  />
                )}

                {/* PUBLIC — the fill */}
                {pub === null ? (
                  <circle
                    cx={seat.x}
                    cy={seat.y}
                    r={seat.r}
                    fill="rgba(255,255,255,0.02)"
                    stroke="rgba(223,217,217,0.22)"
                    strokeWidth={1}
                    strokeDasharray="3 3"
                  />
                ) : (
                  <motion.circle
                    cx={seat.x}
                    cy={seat.y}
                    r={seat.r}
                    /* the attribute is the first-paint value framer animates
                       FROM — without it the seat flashes default black before
                       the first animation frame lands */
                    fill={positionFill(pub)}
                    animate={{ fill: positionFill(pub) }}
                    transition={{ duration: reduce ? 0 : 0.5, ease: "easeOut" }}
                    stroke={marked ? "rgba(223,217,217,0.75)" : "rgba(223,217,217,0.28)"}
                    strokeWidth={marked ? 1.5 : 1}
                  />
                )}

                {/* speaking order for this replicate, inside the seat */}
                <text
                  x={seat.x}
                  y={seat.y + 4}
                  textAnchor="middle"
                  style={{
                    fontSize: 11,
                    fontFamily: "var(--font-mono)",
                    fill: pub === null ? "var(--color-muted)" : "rgba(223,217,217,0.92)",
                    pointerEvents: "none",
                  }}
                >
                  {s.speakingIndex ?? "–"}
                </text>

                {/* identity chip — categorical, tiny, and the only place a
                    member's own colour appears on the table */}
                <rect
                  x={lab.anchor === "end" ? lab.x - 5 : lab.anchor === "start" ? lab.x : lab.x - 2.5}
                  y={lab.y - 12}
                  width={5}
                  height={5}
                  fill={identityHue(s.index)}
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
                  {clip(s.member.name, 16)}
                </text>
                <text
                  x={lab.x + (lab.anchor === "start" ? 9 : lab.anchor === "end" ? -9 : 0)}
                  y={lab.y + 11}
                  textAnchor={lab.anchor}
                  style={{
                    fontSize: 9,
                    fontFamily: "var(--font-mono)",
                    fill: "var(--color-muted)",
                  }}
                >
                  {pub === null ? (
                    "yet to speak"
                  ) : s.ballot ? (
                    <>
                      <tspan style={{ fill: "var(--color-muted)" }}>ballot </tspan>
                      <tspan style={{ fill: positionInk(pub) }}>{fmt2(pub)}</tspan>
                    </>
                  ) : (
                    <>
                      <tspan style={{ fill: positionInk(pub) }}>{fmt2(pub)}</tspan>
                      {prv !== null && (
                        <>
                          {" / "}
                          <tspan style={{ fill: positionInk(prv) }}>{fmt2(prv)}</tspan>
                          {gap !== null && Math.abs(gap) >= 0.05 && (
                            <tspan style={{ fill: "#f0605f" }}> {signed(gap)}</tspan>
                          )}
                        </>
                      )}
                    </>
                  )}
                </text>
              </g>
            );
          })}
        </svg>
      </div>

      {/* ── the readout: one focused member, in sentences ───────────────
          The height is RESERVED, not grown. Measured before this was: the box
          went 72px idle → 111px focused → 129px on the member with the longest
          statement, so every hover shoved the ledger, the trajectory and
          everything below it down the page and snapped it back on leave. The
          quote and the note are line-clamped for the same reason — an unbounded
          block of prose in a fixed strip is a row whose height is data. */}
      <div className="min-h-[9.25rem] border-t border-hairline px-3 py-2">
        {focused ? (
          <div>
            <div className="flex flex-wrap items-baseline gap-x-2">
              <span
                className="inline-block h-2 w-2 shrink-0"
                style={{ background: identityHue(focused.index) }}
                aria-hidden
              />
              <span className="text-[12px] text-ink">{focused.member.name}</span>
              <span className="font-mono text-[9px] uppercase tracking-[0.14em] text-muted">
                {focused.member.role}
                {focused.member.archetype ? ` · ${focused.member.archetype}` : ""}
              </span>
              {cursor.pinned !== null && (
                <button
                  onClick={cursor.clear}
                  className="ml-auto font-mono text-[9px] text-accent transition hover:text-ink"
                >
                  pinned ✕
                </button>
              )}
            </div>

            {/* tabular figures on the whole paragraph: every number in here
                changes as the cursor moves between members, and proportional
                digits make the words after them jitter sideways */}
            <p className="mt-1 font-mono text-[10px] leading-relaxed tabular-nums">
              {focused.publicPosition === null ? (
                <span className="text-muted">has not spoken yet in this view.</span>
              ) : focused.ballot ? (
                <>
                  <span className="text-muted">sealed ballot </span>
                  <span style={{ color: positionInk(focused.publicPosition) }}>
                    {fmt2(focused.publicPosition)} — {positionWord(focused.publicPosition)}
                  </span>
                  <span className="text-muted">
                    {" "}
                    · written alone, before anyone spoke — not a statement to the
                    room
                  </span>
                </>
              ) : (
                <>
                  <span className="text-muted">said </span>
                  <span style={{ color: positionInk(focused.publicPosition) }}>
                    {fmt2(focused.publicPosition)} — {positionWord(focused.publicPosition)}
                  </span>
                  {focused.privatePosition === null ? (
                    <span className="text-muted">
                      {" "}
                      · private position held until the run completes
                    </span>
                  ) : (
                    <>
                      <span className="text-muted"> · thinks </span>
                      <span style={{ color: positionInk(focused.privatePosition) }}>
                        {fmt2(focused.privatePosition)} — {positionWord(focused.privatePosition)}
                      </span>
                      <span className="text-critical">
                        {" "}
                        · Δ {signed(focused.publicPosition - focused.privatePosition)}
                      </span>
                      <span className="text-ink-2">
                        {" "}
                        — {gapWord(focused.publicPosition - focused.privatePosition)}
                      </span>
                    </>
                  )}
                </>
              )}
            </p>

            {focused.statement && (
              <p className="mt-1 line-clamp-2 border-l-2 pl-2 text-[11px] leading-snug text-ink-2"
                 style={{ borderLeftColor: "rgb(var(--region-rgb) / 0.55)" }}>
                “{focused.statement}”
              </p>
            )}
            {focused.note && (
              <p className="mt-1 line-clamp-2 border-l-2 border-dashed border-critical/50 pl-2 text-[11px] leading-snug text-muted">
                privately: {focused.note}
              </p>
            )}
            {/* Always present, because an OPTIONAL row is a row whose presence
                moves the page: this line existed for the members who conceded
                and not for the rest, so crossing between two faces jumped
                everything below by its height. "conceded to nobody" is also the
                more honest reading — a member who yielded to no one is a
                finding, not an absence. */}
            <p className="mt-1 truncate font-mono text-[9px] text-muted">
              conceded to{" "}
              {focused.conceded.length > 0 ? focused.conceded.join(", ") : "nobody"}
            </p>
            <p className="mt-1 font-mono text-[9px] tabular-nums text-muted">
              stake: {focused.member.stake || "unstated"} · openness{" "}
              {fmt2(focused.member.openness)} · assertiveness {fmt2(focused.member.assertiveness)}{" "}
              · seniority {fmt2(focused.member.seniority)} · risk{" "}
              {fmt2(focused.member.risk_appetite)}
              {focused.member.expertise?.length
                ? ` · ${focused.member.expertise.join(", ")}`
                : ""}
            </p>
          </div>
        ) : (
          <p className="font-mono text-[10px] leading-relaxed text-muted">
            Hover a seat to read that member, click to pin them across every
            chart on this screen; arrow keys step around the table.{" "}
            <span className="text-ink-2">Seats sit by seniority</span> — the most
            senior at the head, then alternating down each side, with seat size
            carrying the same rank. The numeral inside each seat is that
            member&apos;s place in this replicate&apos;s speaking order.
            {!revealed && (
              <>
                {" "}
                Rings are dashed because private positions are not streamed —
                they arrive with the result, and only then.
              </>
            )}
          </p>
        )}
      </div>

      {/* ── the ledger: the ring/fill gap as text, for everyone ───────── */}
      {ledger.length > 0 && (
        <div className="border-t border-hairline">
          <div className="flex flex-wrap items-baseline gap-x-3 px-3 pt-2">
            <span className="hud-label">PREFERENCE FALSIFICATION</span>
            <span className="font-mono text-[9px] text-muted">
              said minus thought, sorted by size · a positive Δ is public support
              with nothing behind it
            </span>
          </div>
          <div className="overflow-x-auto px-3 pb-2.5 pt-1">
            <table className="w-full min-w-[420px] border-collapse">
              <thead>
                <tr className="text-left">
                  {["member", "said", "thinks", "Δ", "reading"].map((h) => (
                    <th
                      key={h}
                      className="hud-label whitespace-nowrap py-1 pr-3 font-normal"
                      scope="col"
                    >
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {ledger.map(({ seat: s, gap }) => (
                  <tr
                    key={s.member.name}
                    onPointerEnter={() => cursor.setHovered(s.index)}
                    onPointerLeave={() => cursor.setHovered(null)}
                    onClick={() => cursor.togglePin(s.index)}
                    className="cursor-pointer border-t border-hairline/40 transition-colors"
                    style={{
                      background: cursor.isMarked(s.index)
                        ? "rgb(var(--region-rgb) / 0.08)"
                        : undefined,
                    }}
                  >
                    <td className="whitespace-nowrap py-1 pr-3 font-mono text-[10px] text-ink-2">
                      <span
                        className="mr-1.5 inline-block h-2 w-2 align-middle"
                        style={{ background: identityHue(s.index) }}
                        aria-hidden
                      />
                      {s.member.name}
                    </td>
                    <td
                      className="whitespace-nowrap py-1 pr-3 font-mono text-[10px] tabular-nums"
                      style={{ color: positionInk(s.publicPosition as number) }}
                    >
                      {fmt2(s.publicPosition as number)} {positionWord(s.publicPosition as number)}
                    </td>
                    <td
                      className="whitespace-nowrap py-1 pr-3 font-mono text-[10px] tabular-nums"
                      style={{ color: positionInk(s.privatePosition as number) }}
                    >
                      {fmt2(s.privatePosition as number)}{" "}
                      {positionWord(s.privatePosition as number)}
                    </td>
                    <td
                      className={`whitespace-nowrap py-1 pr-3 font-mono text-[10px] tabular-nums ${
                        Math.abs(gap) >= 0.15 ? "text-critical" : "text-ink-2"
                      }`}
                    >
                      {signed(gap)}
                    </td>
                    <td className="py-1 pr-1 text-[10px] leading-snug text-muted">
                      {gapWord(gap)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </section>
  );
}
