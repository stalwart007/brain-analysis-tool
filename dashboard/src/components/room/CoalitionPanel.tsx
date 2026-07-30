"use client";

/**
 * CAMPS AND POWER — who ended up on whose side, and whose vote decides it.
 *
 * Two different quantities, deliberately shown together because they so often
 * disagree: a camp is where positions clustered, while Shapley-Shubik and
 * Banzhaf are how often a member is the one who turns a losing coalition into a
 * winning one. The person in the smaller camp is frequently the one with the
 * power index, and that gap is the most actionable thing in the panel.
 */

import type { RoomCoalitions, RoomMember } from "@/lib/api";
import type { ChartCursor } from "../charts/cursor";
import { fmt2, identityHue, positionFill, positionInk, positionWord, maxOf } from "./scale";

/** One set of camps — public or private — sharing the members cursor. */
function CampList({
  label,
  camps,
  cast,
  cursor,
  dashed,
}: {
  label: string;
  camps: { members: string[]; mean_position: number }[];
  cast: RoomMember[];
  cursor: ChartCursor;
  /** the private set is drawn dashed, the way private positions are everywhere */
  dashed?: boolean;
}) {
  if (!camps.length) return null;
  return (
    <div className="border-t border-hairline/60 first:border-t-0">
      <span className="block px-3 pt-2 hud-label">{label}</span>
      <ul className="divide-y divide-hairline/40">
        {[...camps]
          .sort((a, b) => b.members.length - a.members.length)
          .map((camp, ci) => (
            <li key={ci} className="px-3 py-2">
              <div className="flex flex-wrap items-baseline gap-x-2">
                <span
                  className="inline-block h-2.5 w-2.5 shrink-0"
                  style={{
                    background: dashed ? "transparent" : positionFill(camp.mean_position),
                    border: dashed
                      ? `1.5px solid ${positionInk(camp.mean_position)}`
                      : "1px solid var(--color-hairline)",
                  }}
                  aria-hidden
                />
                <span
                  className="font-mono text-[10px] tabular-nums"
                  style={{ color: positionInk(camp.mean_position) }}
                >
                  {fmt2(camp.mean_position)} — {positionWord(camp.mean_position)}
                </span>
                <span className="font-mono text-[9px] text-muted">
                  {camp.members.length} member{camp.members.length === 1 ? "" : "s"}
                </span>
              </div>
              <div className="mt-1 flex flex-wrap gap-1">
                {camp.members.map((name) => {
                  const i = cast.findIndex((m) => m.name === name);
                  const marked = i >= 0 && cursor.isMarked(i);
                  return (
                    <button
                      key={name}
                      onPointerEnter={() => i >= 0 && cursor.setHovered(i)}
                      onPointerLeave={() => cursor.setHovered(null)}
                      onClick={() => i >= 0 && cursor.togglePin(i)}
                      className={`flex items-center gap-1 border px-1.5 py-0.5 font-mono text-[9px] transition ${
                        marked ? "border-hairline text-ink" : "border-hairline/50 text-muted"
                      }`}
                    >
                      <span
                        className="inline-block h-1.5 w-1.5"
                        style={{ background: i >= 0 ? identityHue(i) : "var(--color-muted)" }}
                        aria-hidden
                      />
                      {name}
                    </button>
                  );
                })}
              </div>
            </li>
          ))}
      </ul>
    </div>
  );
}

export default function CoalitionPanel({
  coalitions,
  cast,
  cursor,
}: {
  coalitions?: RoomCoalitions | null;
  cast: RoomMember[];
  cursor: ChartCursor;
}) {
  const camps = coalitions?.camps ?? [];
  const privateCamps = coalitions?.private_camps ?? null;
  const power = coalitions?.power ?? null;
  /**
   * Power indices that are all equal are 1/n BY CONSTRUCTION — the weighted
   * voting game fell back to equal weights because no influence matrix was
   * identified. Ranking five identical numbers would present a construction as a
   * finding, so the panel says so and dims them instead.
   */
  const uninformativePower =
    !!power &&
    power.length > 1 &&
    Math.max(...power.map((p) => p.shapley_shubik)) -
      Math.min(...power.map((p) => p.shapley_shubik)) <
      1e-9;

  return (
    <section className="border border-hairline bg-black/40">
      <header className="flex flex-wrap items-baseline gap-x-3 gap-y-1 border-b border-hairline px-3 py-2">
        <span className="hud-label" style={{ color: "rgb(var(--region-rgb))" }}>
          CAMPS &amp; POWER
        </span>
        <span className="font-mono text-[9px] text-muted">
          where positions clustered · who is pivotal
        </span>
      </header>

      {!coalitions || camps.length === 0 ? (
        <p className="px-3 py-3 text-[12px] leading-relaxed text-ink-2">
          <span className="hud-label mr-2 text-critical">NOT COMPUTED</span>
          No camps were resolved for this room — either the positions did not
          separate into groups, or there were too few members to call anything a
          coalition.
        </p>
      ) : (
        <>
          {/* PUBLIC camps, then PRIVATE ones. Two pictures of the same room, and
              the difference between them is the finding: a room that shows one
              camp out loud and two in private has not agreed about anything. */}
          <CampList
            label={privateCamps ? "AS STATED" : "CAMPS"}
            camps={camps}
            cast={cast}
            cursor={cursor}
          />
          {privateCamps && (
            <CampList
              label="AS BELIEVED"
              camps={privateCamps}
              cast={cast}
              cursor={cursor}
              dashed
            />
          )}
          {coalitions.hidden_split !== null && coalitions.hidden_split !== undefined && (
            <p
              className="border-t border-hairline px-3 py-2 font-mono text-[9px] leading-relaxed"
              style={{ color: coalitions.hidden_split ? "#f0605f" : "var(--color-muted)" }}
            >
              {coalitions.hidden_split
                ? "HIDDEN SPLIT — the room's private camps are not its public ones. The agreement on the record does not exist off it."
                : "no hidden split — the public camps and the private camps are the same room"}
            </p>
          )}
        </>
      )}

      {power?.length ? (
        <div className="border-t border-hairline px-3 py-2">
          <div className="flex flex-wrap items-baseline gap-x-2">
            <span className="hud-label">PIVOTAL POWER</span>
            <span className="font-mono text-[9px] text-muted">
              Shapley-Shubik · Banzhaf — how often this member is the one who
              flips a losing bloc into a winning one
            </span>
          </div>
          {uninformativePower && (
            <p className="mt-1 font-mono text-[9px] leading-relaxed text-critical">
              Every index is identical — the voting game fell back to equal
              weights because no influence matrix was identified, so these are
              1/n by construction and say nothing about this room.
              {coalitions?.weight_source ? ` (${coalitions.weight_source})` : ""}
            </p>
          )}
          <div
            className="mt-1.5 space-y-1"
            style={{ opacity: uninformativePower ? 0.45 : 1 }}
          >
            {[...power]
              .sort((a, b) => b.shapley_shubik - a.shapley_shubik)
              .map((p) => {
                const i = cast.findIndex((m) => m.name === p.name);
                const marked = i >= 0 && cursor.isMarked(i);
                const top = maxOf(power.map((x) => x.shapley_shubik));
                return (
                  <div
                    key={p.name}
                    onPointerEnter={() => i >= 0 && cursor.setHovered(i)}
                    onPointerLeave={() => cursor.setHovered(null)}
                    onClick={() => i >= 0 && cursor.togglePin(i)}
                    className="flex cursor-pointer items-center gap-2"
                  >
                    <span
                      className="w-24 shrink-0 truncate font-mono text-[9px]"
                      style={{ color: marked ? "var(--color-ink)" : "var(--color-muted)" }}
                    >
                      {p.name}
                    </span>
                    <span className="h-2 flex-1 bg-white/[0.03]">
                      <span
                        className="block h-2"
                        style={{
                          width: `${Math.max(1, (p.shapley_shubik / top) * 100)}%`,
                          background: "rgb(var(--region-rgb) / 0.6)",
                          opacity: cursor.index === null || marked ? 1 : 0.3,
                        }}
                      />
                    </span>
                    <span className="w-20 shrink-0 text-right font-mono text-[9px] tabular-nums text-ink-2">
                      {fmt2(p.shapley_shubik)} · {fmt2(p.banzhaf)}
                    </span>
                  </div>
                );
              })}
          </div>
        </div>
      ) : (
        coalitions &&
        camps.length > 0 && (
          <p className="border-t border-hairline px-3 py-2 font-mono text-[9px] text-muted">
            power indices not computed — camps alone say where people stood, not
            who was pivotal
          </p>
        )
      )}
    </section>
  );
}
