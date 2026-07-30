"use client";

/**
 * THE CAST — describe the people in prose, then argue with what comes back.
 *
 * A form with nine fields per character is a form nobody fills in, so the input
 * is a paragraph: "our CFO who has been burned by two failed migrations, a
 * head of sales who wants it yesterday, a staff engineer nobody outranks". The
 * backend casts that into parameterised members.
 *
 * APPROVE BEFORE SPEND, exactly as the audience panel and the asset input do.
 * The cast the researcher reads is the cast that runs — every trait is editable
 * here, because a room the researcher cannot correct is a room whose result they
 * have no reason to believe. Re-casting or changing the seat count revokes the
 * approval, since nobody has read the new people yet.
 */

import { useState } from "react";
import { motion } from "framer-motion";
import type { RoomMember } from "@/lib/api";
import { adaptMember } from "./adapt";
import { identityHue } from "./scale";

const TRAITS: { key: keyof RoomMember; label: string; hint: string }[] = [
  { key: "openness", label: "openness", hint: "0 will not move · 1 updates on any decent argument" },
  { key: "assertiveness", label: "assertiveness", hint: "0 waits to be asked · 1 talks over people" },
  { key: "seniority", label: "seniority", hint: "0 junior · 1 the room reports to them" },
  { key: "risk_appetite", label: "risk appetite", hint: "0 wants proof · 1 wants to ship" },
];

function blankMember(n: number): RoomMember {
  return {
    name: `Member ${n}`,
    role: "",
    archetype: "",
    stake: "",
    // Left empty deliberately: a hand-added seat has no authored conditioning
    // prompt, the card says so, and one is synthesised from these fields at
    // dispatch rather than faked here.
    system_prompt: "",
    openness: 0.5,
    assertiveness: 0.5,
    seniority: 0.5,
    risk_appetite: 0.5,
    expertise: [],
  };
}

export default function CastComposer({
  brief,
  onBriefChange,
  motionText,
  seats,
  cast,
  onCast,
  approved,
  onApprove,
  disabled,
  castPath,
}: {
  brief: string;
  onBriefChange: (v: string) => void;
  /** the motion, sent as context so the cast has stakes in THIS decision */
  motionText: string;
  seats: number;
  cast: RoomMember[];
  onCast: (c: RoomMember[]) => void;
  approved: boolean;
  onApprove: (v: boolean) => void;
  disabled?: boolean;
  /** proxied backend path, e.g. /api/cs/room/cast */
  castPath: string;
}) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [open, setOpen] = useState(true);
  const [note, setNote] = useState<string | null>(null);

  const ready = brief.trim().length >= 20;

  async function castRoom() {
    setBusy(true);
    setError(null);
    try {
      const res = await fetch(castPath, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ brief, motion: motionText, seats }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => null);
        throw new Error(body?.detail ?? `Casting failed (${res.status})`);
      }
      const data = await res.json();
      // Accept either key: this endpoint is being written concurrently, and a
      // rename must not present as an empty room.
      const members = (data?.cast ?? data?.members ?? []) as unknown[];
      if (!Array.isArray(members) || members.length === 0) {
        throw new Error("The cast came back empty — try naming the people more explicitly.");
      }
      onCast(members.map(adaptMember));
      setNote(typeof data?.note === "string" ? data.note : null);
      onApprove(false);
      setOpen(true);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not cast the room");
    } finally {
      setBusy(false);
    }
  }

  function patch(i: number, next: Partial<RoomMember>) {
    onCast(cast.map((m, k) => (k === i ? { ...m, ...next } : m)));
  }

  function remove(i: number) {
    onCast(cast.filter((_, k) => k !== i));
    onApprove(false);
  }

  function add() {
    onCast([...cast, blankMember(cast.length + 1)]);
    onApprove(false);
  }

  return (
    <div className="mb-4 rounded-xl border border-hairline bg-surface-2/40 p-3">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <button
          onClick={() => setOpen((o) => !o)}
          aria-expanded={open}
          className="hud-label text-ink-2 transition hover:text-ink"
        >
          THE CAST {open ? "▾" : "▸"}
        </button>
        <span className="font-mono text-[10px] text-muted">
          {cast.length === 0 ? (
            "nobody cast yet — describe the room below"
          ) : approved ? (
            <span className="text-good">
              ● approved · {cast.length} member{cast.length === 1 ? "" : "s"} will run
            </span>
          ) : (
            <span className="text-accent-2">
              ○ {cast.length} member{cast.length === 1 ? "" : "s"} — not approved yet
            </span>
          )}
        </span>
      </div>

      <textarea
        value={brief}
        onChange={(e) => onBriefChange(e.target.value)}
        disabled={disabled || busy}
        placeholder="Describe the people. “Our CFO, burned by two failed migrations and three months from a board review. A head of sales who wants it yesterday and has the CEO's ear. A staff engineer nobody outranks who thinks the whole plan is backwards.”"
        className="mt-2.5 min-h-24 w-full resize-y rounded-xl border border-hairline bg-surface-2 p-3 text-sm outline-none placeholder:text-muted focus:border-accent/60 disabled:opacity-50"
      />

      <div className="mt-2 flex flex-wrap items-center gap-2">
        <motion.button
          whileTap={{ scale: 0.97 }}
          onClick={castRoom}
          disabled={disabled || busy || !ready}
          title={ready ? undefined : "describe the room in at least 20 characters"}
          className="rounded-lg border border-hairline px-3 py-1.5 text-xs text-ink-2 transition hover:border-accent/50 hover:text-ink disabled:opacity-40"
        >
          {busy ? "casting…" : cast.length ? "re-cast the room" : "cast the room"}
        </motion.button>
        {cast.length > 0 && (
          <>
            <button
              onClick={add}
              disabled={disabled || busy}
              className="rounded-lg border border-dashed border-hairline px-3 py-1.5 text-xs text-muted transition hover:border-accent/50 hover:text-ink-2 disabled:opacity-40"
            >
              + add a seat
            </button>
            <motion.button
              whileTap={{ scale: 0.97 }}
              onClick={() => onApprove(!approved)}
              disabled={disabled || busy}
              aria-pressed={approved}
              className={`rounded-lg border px-3 py-1.5 text-xs transition disabled:opacity-40 ${
                approved
                  ? "border-good/60 text-good"
                  : "border-accent/60 bg-accent/[0.10] text-ink hover:brightness-125"
              }`}
            >
              {approved ? "✓ cast approved" : "approve this cast"}
            </motion.button>
          </>
        )}
        {note && <span className="font-mono text-[9px] text-muted">{note}</span>}
        {!motionText.trim() && (
          <span className="font-mono text-[9px] text-accent-2">
            no motion stated yet — cast now and their stakes will be generic
          </span>
        )}
      </div>

      {error && <p className="mt-2 text-[11px] text-critical">{error}</p>}

      {open && cast.length > 0 && (
        <ul className="mt-3 grid gap-2 lg:grid-cols-2">
          {cast.map((m, i) => (
            <li key={i} className="rounded-lg border border-hairline/60 bg-black/20 p-2.5">
              <div className="flex items-start gap-2">
                <span
                  className="mt-1.5 inline-block h-2 w-2 shrink-0"
                  style={{ background: identityHue(i) }}
                  aria-hidden
                />
                <div className="min-w-0 flex-1 space-y-1.5">
                  <input
                    value={m.name}
                    onChange={(e) => patch(i, { name: e.target.value })}
                    disabled={disabled || busy}
                    aria-label={`Member ${i + 1} name`}
                    className="w-full rounded border border-hairline bg-surface-2 px-2 py-1 text-xs text-ink outline-none focus:border-accent/60"
                  />
                  <div className="flex gap-1.5">
                    <input
                      value={m.role}
                      onChange={(e) => patch(i, { role: e.target.value })}
                      disabled={disabled || busy}
                      placeholder="role"
                      aria-label={`${m.name} role`}
                      className="min-w-0 flex-1 rounded border border-hairline bg-surface-2 px-2 py-1 font-mono text-[10px] text-ink-2 outline-none placeholder:text-muted focus:border-accent/60"
                    />
                    <input
                      value={m.archetype}
                      onChange={(e) => patch(i, { archetype: e.target.value })}
                      disabled={disabled || busy}
                      placeholder="archetype"
                      aria-label={`${m.name} archetype`}
                      className="min-w-0 flex-1 rounded border border-hairline bg-surface-2 px-2 py-1 font-mono text-[10px] text-ink-2 outline-none placeholder:text-muted focus:border-accent/60"
                    />
                  </div>
                  <input
                    value={m.stake}
                    onChange={(e) => patch(i, { stake: e.target.value })}
                    disabled={disabled || busy}
                    placeholder="what they stand to gain or lose"
                    aria-label={`${m.name} stake`}
                    className="w-full rounded border border-hairline bg-surface-2 px-2 py-1 text-[11px] text-ink-2 outline-none placeholder:text-muted focus:border-accent/60"
                  />
                  <input
                    value={(m.expertise ?? []).join(", ")}
                    onChange={(e) =>
                      patch(i, {
                        expertise: e.target.value
                          .split(",")
                          .map((s) => s.trim())
                          .filter(Boolean),
                      })
                    }
                    disabled={disabled || busy}
                    placeholder="expertise, comma separated"
                    aria-label={`${m.name} expertise`}
                    className="w-full rounded border border-hairline bg-surface-2 px-2 py-1 font-mono text-[10px] text-muted outline-none placeholder:text-muted focus:border-accent/60"
                  />

                  {/* An authored conditioning prompt is what makes a member a
                      character rather than a name with four numbers on it. When
                      one is missing the card says so instead of the run quietly
                      playing a generic person. */}
                  {!m.system_prompt.trim() && (
                    <p className="font-mono text-[9px] leading-relaxed text-accent-2">
                      no conditioning prompt — this seat will be played from its
                      role, archetype and stake alone
                    </p>
                  )}

                  {TRAITS.map((t) => {
                    const v = Number(m[t.key] ?? 0.5);
                    return (
                      <label key={t.key} className="flex items-center gap-2" title={t.hint}>
                        <span className="w-24 shrink-0 font-mono text-[9px] uppercase tracking-[0.1em] text-muted">
                          {t.label}
                        </span>
                        <input
                          type="range"
                          min={0}
                          max={1}
                          step={0.05}
                          value={Number.isFinite(v) ? v : 0.5}
                          onChange={(e) => patch(i, { [t.key]: Number(e.target.value) } as Partial<RoomMember>)}
                          disabled={disabled || busy}
                          className="h-1 min-w-0 flex-1"
                          style={{ accentColor: "rgb(var(--region-rgb))" }}
                        />
                        <span className="w-8 shrink-0 text-right font-mono text-[10px] tabular-nums text-ink-2">
                          {(Number.isFinite(v) ? v : 0.5).toFixed(2)}
                        </span>
                      </label>
                    );
                  })}
                </div>
                <button
                  onClick={() => remove(i)}
                  disabled={disabled || busy}
                  aria-label={`Remove ${m.name}`}
                  className="shrink-0 rounded px-1 text-xs text-muted transition hover:text-critical"
                >
                  ×
                </button>
              </div>
            </li>
          ))}
        </ul>
      )}

      {cast.length > 0 && !approved && (
        <p className="mt-2 font-mono text-[9px] leading-relaxed text-muted">
          Nothing is spent until this cast is approved. Traits are the parameters
          the members are actually simulated from, so an edit here changes the
          run — and a cast you disagree with is a result you have no reason to
          believe. Re-casting or changing the seat count revokes the approval.
        </p>
      )}
    </div>
  );
}

