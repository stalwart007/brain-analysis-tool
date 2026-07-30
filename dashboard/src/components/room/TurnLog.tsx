"use client";

/**
 * THE TRANSCRIPT — what was said, and what was thought while saying it.
 *
 * The statistics upstairs are reductions of this. Reading a single member's
 * public line beside their private note is where a researcher decides whether
 * the model produced a person or a press release, so the two sit on the same
 * row rather than in two tabs.
 *
 * Collapsed by default: it is the longest thing on the screen and the least
 * often needed, but when it is needed nothing else will do.
 */

import { useState } from "react";
import type { RoomMember, RoomTurn } from "@/lib/api";
import type { ChartCursor } from "../charts/cursor";
import { fmt2, identityHue, positionInk, positionWord, signed } from "./scale";

export default function TurnLog({
  turns,
  cast,
  cursor,
  label,
}: {
  /** already filtered to the arm/replicate in view */
  turns: RoomTurn[];
  cast: RoomMember[];
  cursor: ChartCursor;
  label: string;
}) {
  const [open, setOpen] = useState(false);

  if (!turns.length) return null;

  const rounds = Array.from(new Set(turns.map((t) => t.round))).sort((a, b) => a - b);
  const focus = cursor.index;
  const focusedName = focus !== null ? cast[focus]?.name : null;

  return (
    <section className="border border-hairline bg-black/30">
      <button
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        className="flex w-full flex-wrap items-baseline gap-x-3 px-3 py-2 text-left transition hover:bg-white/[0.02]"
      >
        <span className="hud-label">{open ? "▾" : "▸"} TRANSCRIPT — {label}</span>
        <span className="font-mono text-[9px] text-muted">
          {turns.length} turn{turns.length === 1 ? "" : "s"} across {rounds.length} round
          {rounds.length === 1 ? "" : "s"}
          {focusedName && <span className="text-accent"> · {focusedName} pinned</span>}
        </span>
      </button>

      {open && (
        <div className="border-t border-hairline">
          {rounds.map((r) => (
            <div key={r}>
              <div className="flex items-baseline gap-2 border-b border-hairline/40 bg-white/[0.015] px-3 py-1">
                <span className="hud-label">ROUND {r}</span>
                {r === 0 && (
                  <span className="font-mono text-[9px] text-muted">
                    sealed ballot — written alone, before anyone spoke
                  </span>
                )}
              </div>
              {turns
                .filter((t) => t.round === r)
                .sort(
                  (a, b) => a.replicate - b.replicate || a.speaking_index - b.speaking_index
                )
                .map((t, k) => {
                  const idx = cast.findIndex((m) => m.name === t.member);
                  const marked = idx >= 0 && cursor.isMarked(idx);
                  const dim = focus !== null && !marked;
                  // public − private, the server's sign. NaN when the private
                  // half is genuinely absent, which is never in a terminal
                  // result and always in a live frame.
                  const gap =
                    Number.isFinite(t.private_position) && Number.isFinite(t.public_position)
                      ? t.public_position - t.private_position
                      : null;
                  return (
                    <div
                      key={`${t.replicate}-${t.member}-${k}`}
                      onPointerEnter={() => idx >= 0 && cursor.setHovered(idx)}
                      onPointerLeave={() => cursor.setHovered(null)}
                      onClick={() => idx >= 0 && cursor.togglePin(idx)}
                      className="cursor-pointer border-b border-hairline/30 px-3 py-2 transition-colors"
                      style={{
                        opacity: dim ? 0.45 : 1,
                        background: marked ? "rgb(var(--region-rgb) / 0.06)" : undefined,
                      }}
                    >
                      <div className="flex flex-wrap items-baseline gap-x-2">
                        <span className="font-mono text-[9px] tabular-nums text-muted">
                          {t.speaking_index}
                        </span>
                        <span
                          className="inline-block h-2 w-2 shrink-0"
                          style={{
                            background: idx >= 0 ? identityHue(idx) : "var(--color-muted)",
                          }}
                          aria-hidden
                        />
                        <span className="text-[11px] text-ink-2">{t.member}</span>
                        <span
                          className="font-mono text-[10px] tabular-nums"
                          style={{ color: positionInk(t.public_position) }}
                        >
                          {t.round === 0 && <span className="text-muted">ballot </span>}
                          {fmt2(t.public_position)} {positionWord(t.public_position)}
                        </span>
                        {gap !== null && (
                          <span
                            className={`font-mono text-[9px] tabular-nums ${
                              Math.abs(gap) >= 0.15 ? "text-critical" : "text-muted"
                            }`}
                          >
                            privately {fmt2(t.private_position)} · Δ {signed(gap)}
                          </span>
                        )}
                        {t.conceded_to?.length > 0 && (
                          <span className="font-mono text-[9px] text-muted">
                            conceded to {t.conceded_to.join(", ")}
                          </span>
                        )}
                        {turns.some((x) => x.replicate !== t.replicate) && (
                          <span className="ml-auto font-mono text-[9px] text-muted">
                            rep {t.replicate + 1}
                          </span>
                        )}
                      </div>
                      {t.public_statement && (
                        <p className="mt-1 text-[12px] leading-snug text-ink">
                          {t.public_statement}
                        </p>
                      )}
                      {t.private_note && (
                        <p className="mt-1 border-l-2 border-dashed border-critical/40 pl-2 text-[11px] leading-snug text-muted">
                          {t.private_note}
                        </p>
                      )}
                    </div>
                  );
                })}
            </div>
          ))}
        </div>
      )}
    </section>
  );
}
