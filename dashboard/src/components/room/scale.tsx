/**
 * THE ROOM'S SHARED VOCABULARY — one scale, one seating plan, one set of words.
 *
 * Five visualisations read the same two numbers (a public position and a
 * private one) and they have to agree about what a colour means, where a face
 * sits, and what "0.68" is called out loud. Defining that once is the
 * difference between five views of one room and five unrelated pictures.
 *
 * RULES OF THE SYSTEM, applied here:
 * · Colour is data. The only saturated hues are positions on the motion, on a
 *   diverging scale anchored at 0.5 — the same validated pair used by the
 *   appraisal tensor (#3987e5 for / #d95926 against), CVD-checked on this
 *   surface. Identity gets a small categorical chip, never a fill.
 * · Never colour alone. Every position that is drawn is also said in words,
 *   because a diverging fill and a diverging ring are exactly the pair a
 *   colour-blind reader cannot separate.
 */

import type { RoomMember, RoomTurn } from "@/lib/api";

/* ── positions ─────────────────────────────────────────────────────────── */

/** 0 = against the motion, 1 = for it, 0.5 = genuinely undecided. */
export function positionFill(p: number): string {
  const t = Math.max(-1, Math.min(1, (p - 0.5) / 0.5));
  const mag = Math.abs(t);
  // Undecided is nearly colourless on purpose: a seat at 0.5 has taken no
  // position, and painting it a confident mid-tone would claim one.
  const alpha = 0.1 + mag * 0.78;
  return t >= 0 ? `rgba(57, 135, 229, ${alpha})` : `rgba(217, 89, 38, ${alpha})`;
}

/**
 * The same scale for a STROKE. A ring's presence is not data — only its hue is
 * — so the alpha floor is much higher: a private position of 0.5 must still
 * read as a drawn ring, or "undecided in private" becomes indistinguishable
 * from "we never learned what they thought".
 */
export function positionStroke(p: number): string {
  const t = Math.max(-1, Math.min(1, (p - 0.5) / 0.5));
  const mag = Math.abs(t);
  const alpha = 0.42 + mag * 0.56;
  return t >= 0 ? `rgba(57, 135, 229, ${alpha})` : `rgba(217, 89, 38, ${alpha})`;
}

/** Fully saturated, for type. Text needs contrast, not proportionality. */
export function positionInk(p: number): string {
  return p >= 0.5 ? "#4fb6ff" : "#e8703f";
}

/** What the number is called out loud — the readout that makes the scale
 *  legible without colour. */
export function positionWord(p: number): string {
  if (p >= 0.85) return "for";
  if (p >= 0.62) return "leaning for";
  if (p > 0.38) return "undecided";
  if (p > 0.15) return "leaning against";
  return "against";
}

/**
 * The gap, in words. This is the sentence the whole product is built to say.
 *
 * SIGN follows the server: gap = public − private. Positive is public support
 * that does not exist behind it; negative is agreement nobody voiced. The
 * 0.05 floor is the quantisation of a declared position — below it, a gap is
 * rounding rather than a finding.
 */
export function gapWord(gap: number): string {
  const m = Math.abs(gap);
  if (m < 0.05) return "says what they think";
  const dir =
    gap > 0 ? "more supportive out loud than in private" : "more supportive in private than out loud";
  if (m < 0.15) return `slightly ${dir}`;
  if (m < 0.35) return dir;
  return `much ${dir}`;
}

export const fmt2 = (v: number) => v.toFixed(2);
export const signed = (v: number) => `${v > 0 ? "+" : v < 0 ? "−" : ""}${Math.abs(v).toFixed(2)}`;

/* ── identity ──────────────────────────────────────────────────────────── */

/**
 * Categorical hues for member identity, drawn from the palette already in use
 * across the product. Identity is data too — the trajectory chart needs to say
 * WHOSE line this is — but it never fills a seat, because a seat's fill is
 * reserved for the position. Members get a 6px chip instead.
 */
const IDENTITY = [
  "#4fb6ff", "#f2ad1f", "#b07ff0", "#16d016",
  "#f0605f", "#5fd0c8", "#ff7fc4", "#9aa4b8",
];

export function identityHue(i: number): string {
  return IDENTITY[i % IDENTITY.length];
}

/** Longest name a seat label can hold before the ellipse eats the table. */
export function clip(s: string, n = 18): string {
  return s.length > n ? `${s.slice(0, n - 1)}…` : s;
}

/* ── the seating plan ──────────────────────────────────────────────────── */

export interface Seat {
  /** index into the cast */
  member: number;
  x: number;
  y: number;
  /** radians, measured from the table's centre */
  angle: number;
  /** seat radius, already scaled by seniority */
  r: number;
  /** which half of the table the seat is on — decides label anchoring */
  side: "left" | "right" | "top" | "bottom";
}

export interface TableGeometry {
  w: number;
  h: number;
  cx: number;
  cy: number;
  /** the table top itself */
  tableRx: number;
  tableRy: number;
}

/**
 * The box both the table and the network are drawn in. The width carries ~35px
 * of slack past the longest label a side seat can hold (16 mono chars at
 * 10.5px, measured) — a label that clips at the viewBox edge is the failure this
 * geometry was checked against, for every cast size from 2 to 8.
 */
export const TABLE: TableGeometry = {
  w: 800,
  h: 470,
  cx: 400,
  cy: 232,
  tableRx: 172,
  tableRy: 100,
};

/**
 * Seats around the table — most senior at the head, then alternating down each
 * side, which is how a room with a power gradient actually sits down. Seat
 * RADIUS also carries seniority, so the arrangement is legible even if you
 * never read the legend; it is never the only encoding of anything.
 *
 * The arrangement is a pure function of the cast, so it is stable across every
 * view (live, per-replicate, placebo) and identical in the influence network —
 * a face is in the same place in both pictures, which is what makes reading
 * across them possible at all.
 */
export function seatPlan(cast: RoomMember[], geo: TableGeometry = TABLE): Seat[] {
  const n = cast.length;
  if (n === 0) return [];

  const bySeniority = cast
    .map((m, i) => ({ i, s: Number.isFinite(m.seniority) ? m.seniority : 0.5 }))
    .sort((a, b) => b.s - a.s || a.i - b.i)
    .map((x) => x.i);

  // Slot 0 is the head of the table (12 o'clock); the rest fill outward,
  // alternating clockwise/anticlockwise so rank descends down both sides.
  const slots = Array.from({ length: n }, (_, k) => k);
  const ordered = slots
    .slice(1)
    .map((k) => ({ k, d: Math.min(k, n - k), side: k <= n - k ? 0 : 1 }))
    .sort((a, b) => a.d - b.d || a.side - b.side || a.k - b.k)
    .map((x) => x.k);

  const seatRx = geo.tableRx + 54;
  const seatRy = geo.tableRy + 40;
  const seats: Seat[] = [];

  [0, ...ordered].forEach((slot, rank) => {
    const member = bySeniority[rank];
    const angle = -Math.PI / 2 + (slot / n) * Math.PI * 2;
    const x = geo.cx + Math.cos(angle) * seatRx;
    const y = geo.cy + Math.sin(angle) * seatRy;
    const seniority = Number.isFinite(cast[member].seniority) ? cast[member].seniority : 0.5;
    // A tight range: seat size must never be the reason a label is unreadable.
    const r = 21 + seniority * 8;
    const cos = Math.cos(angle);
    const sin = Math.sin(angle);
    const side: Seat["side"] =
      Math.abs(cos) < 0.34 ? (sin < 0 ? "top" : "bottom") : cos > 0 ? "right" : "left";
    seats.push({ member, x, y, angle, r, side });
  });

  // Return in CAST order so callers can index by member without a lookup.
  return seats.sort((a, b) => a.member - b.member);
}

/** Where a seat's label goes, and how it is anchored. */
export function labelAnchor(seat: Seat): {
  x: number;
  y: number;
  anchor: "start" | "middle" | "end";
} {
  const pad = seat.r + 9;
  if (seat.side === "right") return { x: seat.x + pad, y: seat.y - 3, anchor: "start" };
  if (seat.side === "left") return { x: seat.x - pad, y: seat.y - 3, anchor: "end" };
  if (seat.side === "top") return { x: seat.x, y: seat.y - pad - 4, anchor: "middle" };
  return { x: seat.x, y: seat.y + pad + 11, anchor: "middle" };
}

/**
 * An arc on the seat's ring whose SWEEP is the falsification gap.
 *
 * The colour difference between ring and fill is the idiom, but a colour
 * difference is the one thing a reader may not be able to see. Length is not,
 * so the gap is drawn as an arc as well: clockwise from 12 o'clock when they
 * spoke MORE in favour than they privately are (gap = public − private > 0),
 * anticlockwise when they held support back. Same fact, two channels.
 */
export function gapArc(seat: Seat, gap: number, radius: number): string {
  const sweep = Math.min(Math.abs(gap), 1) * Math.PI * 1.5;
  if (sweep < 0.02) return "";
  const dir = gap >= 0 ? 1 : -1;
  const a0 = -Math.PI / 2;
  const a1 = a0 + dir * sweep;
  const x0 = seat.x + Math.cos(a0) * radius;
  const y0 = seat.y + Math.sin(a0) * radius;
  const x1 = seat.x + Math.cos(a1) * radius;
  const y1 = seat.y + Math.sin(a1) * radius;
  const large = sweep > Math.PI ? 1 : 0;
  return `M ${x0.toFixed(2)} ${y0.toFixed(2)} A ${radius} ${radius} 0 ${large} ${
    dir > 0 ? 1 : 0
  } ${x1.toFixed(2)} ${y1.toFixed(2)}`;
}

/* ── reductions over turns ─────────────────────────────────────────────── */

export interface Arm {
  placebo: boolean;
  /** null means "every replicate in this arm, averaged" */
  replicate: number | null;
}

export function armTurns(turns: RoomTurn[], arm: Arm): RoomTurn[] {
  return turns.filter(
    (t) => t.placebo === arm.placebo && (arm.replicate === null || t.replicate === arm.replicate)
  );
}

function mean(xs: number[]): number | null {
  return xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null;
}

/**
 * Per-member position at each round, averaged across whatever replicates the
 * view includes.
 *
 * ROUND 0 IS THE SEALED BALLOT, not a first statement: the member is alone with
 * the motion, nobody has spoken, and nobody will see what they write. So the
 * PUBLIC series is null there — the estimator ignores round 0's public fields
 * for exactly this reason, and drawing them as a stated position would invent
 * the one quantity this study measures. The private series does carry it: that
 * ballot IS the innate position every later gap is measured against.
 */
export function roundSeries(
  turns: RoomTurn[],
  names: string[],
  rounds: number,
  which: "public_position" | "private_position"
): (number | null)[][] {
  return names.map((name) =>
    Array.from({ length: rounds + 1 }, (_, r) => {
      if (r === 0 && which === "public_position") return null;
      const vals = turns
        .filter((t) => t.member === name && t.round === r)
        .map((t) => t[which])
        .filter((v): v is number => typeof v === "number" && Number.isFinite(v));
      return mean(vals);
    })
  );
}

/** The last thing each member said, and where they stood when they said it. */
export function latestByMember(turns: RoomTurn[], name: string): RoomTurn | null {
  let best: RoomTurn | null = null;
  for (const t of turns) {
    if (t.member !== name) continue;
    if (
      !best ||
      t.round > best.round ||
      (t.round === best.round && t.speaking_index > best.speaking_index)
    ) {
      best = t;
    }
  }
  return best;
}

/** Mean of a member's final-round positions across the replicates in view. */
export function finalPosition(
  turns: RoomTurn[],
  name: string,
  which: "public_position" | "private_position"
): number | null {
  const mine = turns.filter((t) => t.member === name);
  if (!mine.length) return null;
  const lastRound = Math.max(...mine.map((t) => t.round));
  return mean(
    mine
      .filter((t) => t.round === lastRound)
      .map((t) => t[which])
      .filter((v): v is number => typeof v === "number" && Number.isFinite(v))
  );
}

/**
 * Cross-member spread (population SD) of stated positions per round.
 *
 * SD, not variance, and averaged the way the server averages it — mean of the
 * per-replicate SDs — so this series is directly comparable to the estimator's
 * own `sd_by_round` and can be drawn on the same axes as it. Two spread
 * definitions on one chart is a chart that lies quietly.
 *
 * The SD is taken WITHIN each replicate before averaging. Pooling first would
 * fold the between-replicate spread — the effect of the speaking order — into a
 * number labelled "how much the members disagree", which is a different
 * quantity and makes every arm look more divided than it was.
 */
export function spreadByRound(turns: RoomTurn[], rounds: number): (number | null)[] {
  const reps = [...new Set(turns.map((t) => t.replicate))];
  return Array.from({ length: rounds + 1 }, (_, r) => {
    const perRep: number[] = [];
    for (const rep of reps) {
      const vals = turns
        .filter((t) => t.round === r && t.replicate === rep)
        // Round 0 is the sealed ballot, so its spread is the spread of what the
        // room walked in BELIEVING — the estimator reads the ballot's private
        // field for the same reason, and mixing the two definitions would put a
        // kink in the collapse curve that no deliberation caused.
        .map((t) => (r === 0 ? t.private_position : t.public_position))
        .filter((v): v is number => typeof v === "number" && Number.isFinite(v));
      if (vals.length < 2) continue;
      const m = vals.reduce((a, b) => a + b, 0) / vals.length;
      perRep.push(Math.sqrt(vals.reduce((a, b) => a + (b - m) * (b - m), 0) / vals.length));
    }
    return perRep.length ? perRep.reduce((a, b) => a + b, 0) / perRep.length : null;
  });
}

export function maxOf(xs: number[], floor = 1e-9): number {
  return Math.max(floor, ...xs.filter((v) => Number.isFinite(v)));
}
