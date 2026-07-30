"use client";

/**
 * THE WHITE ROOM — a deliberation study.
 *
 * Describe the people in prose. They are cast into parameterised members, sat
 * around a table, and made to argue a motion for several rounds. Each round each
 * member says something PUBLICLY, with a position on the motion, and holds a
 * PRIVATE position the others never see. The distance between those two numbers
 * is preference falsification, and measuring it is the point of the whole
 * screen: rooms do not fail because people disagree, they fail because people
 * agree out loud and disagree afterwards.
 *
 * FOUR THINGS MAKE THIS A MEASUREMENT RATHER THAN A DEMO.
 *
 * 1. Replication. The speaking order is seeded and the room is re-run with
 *    different orders, so "the room converged on 0.8" can be separated from "the
 *    room converged on whoever spoke first".
 * 2. A CONTROL ARM. Some replicates are placebo: members are shown statements
 *    from a DIFFERENT room. A language model asked to deliberate will converge
 *    on almost anything, and the only way to know whether this deliberation did
 *    work is to run one that cannot have. If the control converged as hard, this
 *    screen says the study measured nothing, in the largest type on the page.
 * 3. Refusals. Every analysis field arrives nullable, because the backend
 *    refuses to fit what the data cannot support. A refused influence fit
 *    renders as a stated reason at the size the network would have been — never
 *    as a network with thin arrows.
 * 4. Honest ordering. Private positions are NOT streamed. While the run is in
 *    flight you can only watch the public conversation move; the rings around
 *    the seats stay dashed and empty, and the reveal lands with the result. The
 *    screen cannot show what it has not been told.
 *
 * ONE CLOCK: every pulse here derives from --pulse, published on <html> by the
 * shared physiological clock. There is not a single timer in this feature.
 */

import { useMemo, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import type { RoomFalsification, RoomMember, RoomResult, RoomTurn } from "@/lib/api";
import { streamRun } from "@/lib/stream";
import { ChartCursorProvider, useChartCursor } from "./charts/cursor";
import FindingsPanel, { ResearchQuestion } from "./FindingsPanel";
import { adaptMember, adaptRoomResult, withConditioning } from "./room/adapt";
import CastComposer from "./room/CastComposer";
import CoalitionPanel from "./room/CoalitionPanel";
import ConformityChart from "./room/ConformityChart";
import CounterfactualStrip from "./room/CounterfactualStrip";
import InfluenceNetwork from "./room/InfluenceNetwork";
import OpinionTrajectory from "./room/OpinionTrajectory";
import PlaceboContrast from "./room/PlaceboContrast";
import RoundTable, { type SeatDatum } from "./room/RoundTable";
import TurnLog from "./room/TurnLog";
import {
  armTurns,
  finalPosition,
  fmt2,
  identityHue,
  latestByMember,
  positionInk,
  positionWord,
  roundSeries,
  signed,
  spreadByRound,
} from "./room/scale";

/* ── the wire ──────────────────────────────────────────────────────────────
   Both paths live here and nowhere else. `/api/cs/*` is the Next proxy that
   injects the headless API key server-side and forwards to /v1/* on the
   FastAPI backend; streamRun prefixes /api/cs itself, so its path is relative
   to that. If the backend lands these under /v1/studies/room/* instead, this
   block is the only edit. */
const CAST_PATH = "/api/cs/room/cast";
const STREAM_PATH = "/room/stream";

const SAMPLE_MOTION =
  "We freeze all new feature work for six weeks and migrate the billing system before the next enterprise renewal cycle.";

const SAMPLE_BRIEF =
  "Our CFO, burned by two failed migrations and three months out from a board review. A head of sales who wants it yesterday and has the CEO's ear. A staff engineer nobody outranks who thinks the plan is backwards. A support lead who takes every outage call personally and has never been asked to a meeting like this.";

interface LiveTurn {
  replicate: number;
  placebo: boolean;
  round: number;
  member: string;
  speaking_index: number;
  public_statement: string;
  public_position: number;
}

interface View {
  placebo: boolean;
  /** null = every replicate in this arm, averaged */
  replicate: number | null;
}

export default function RoomPanel() {
  return (
    <ChartCursorProvider>
      <RoomInner />
    </ChartCursorProvider>
  );
}

function RoomInner() {
  const [brief, setBrief] = useState("");
  const [motionText, setMotionText] = useState("");
  const [question, setQuestion] = useState("");
  // Defaults and bounds mirror RoomRequest in schemas.py exactly. A control that
  // offers a value the server rejects is a 422 the researcher has to decode.
  const [seats, setSeats] = useState(5);
  const [rounds, setRounds] = useState(4);
  // 5, matching RoomRequest. Three replicates put the influence fit at exactly
  // its 1.5 observations-per-parameter refusal boundary — one dropped replicate
  // and the whole fit refuses — and floored the speaking-order permutation test
  // at p = 2/3! = 0.333, so that control could never fire. Five gives 2.5
  // obs/param and a 0.017 floor. Replicates are the cheap axis: each one buys
  // (rounds − 1) fresh observations per member.
  const [replicates, setReplicates] = useState(5);
  const [placeboReps, setPlaceboReps] = useState(1);
  const [cast, setCast] = useState<RoomMember[]>([]);
  const [approved, setApproved] = useState(false);

  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [stage, setStage] = useState<string | null>(null);
  const [total, setTotal] = useState(0);
  const [liveTurns, setLiveTurns] = useState<LiveTurn[]>([]);
  const [failedTurns, setFailedTurns] = useState(0);
  const [skipped, setSkipped] = useState<string[]>([]);
  const [streamRounds, setStreamRounds] = useState<number | null>(null);
  const [result, setResult] = useState<RoomResult | null>(null);
  const [view, setView] = useState<View>({ placebo: false, replicate: null });
  /** how the cast came to exist, and whether it can disagree at all — both
   *  arrive on the stream's own `cast` frame, before any turn */
  const [provenance, setProvenance] = useState<RoomResult["provenance"]>(null);
  const [diversity, setDiversity] = useState<RoomResult["diversity"]>(null);

  /** The cast that is actually on screen: the backend's, once it has spoken. */
  const roomCast = result?.cast?.length ? result.cast : cast;

  /* ── the two cursors ──────────────────────────────────────────────────
     MEMBERS is one axis; ROUNDS is another. They get separate domains, so
     pinning a face never lights up "round 3" somewhere and imply a link that
     does not exist. Every member-indexed view on this screen — the table, the
     network, the counterfactual strip, the camps, the transcript — shares the
     members domain, which is what makes hovering a seat pin that person
     everywhere. */
  const memberCursor = useChartCursor(
    "room-members",
    roomCast.length,
    "The people at the table",
    (i) => describeMember(i, roomCast, result, view, liveTurns)
  );

  const liveArm = liveTurns.length ? liveTurns[liveTurns.length - 1] : null;
  const currentSpeaker = busy && liveArm ? liveArm.member : null;
  const previousSpeakerIndex = useMemo(() => {
    if (!busy || liveTurns.length < 2) return null;
    const prev = liveTurns[liveTurns.length - 2];
    const cur = liveTurns[liveTurns.length - 1];
    if (prev.replicate !== cur.replicate || prev.placebo !== cur.placebo) return null;
    const i = roomCast.findIndex((m) => m.name === prev.member);
    return i >= 0 ? i : null;
  }, [busy, liveTurns, roomCast]);

  /* ── what the table is showing ─────────────────────────────────────── */

  const turnsInView: RoomTurn[] = useMemo(() => {
    if (!result) return [];
    return armTurns(result.turns ?? [], view);
  }, [result, view]);

  const liveInView: LiveTurn[] = useMemo(() => {
    if (!liveArm) return [];
    return liveTurns.filter(
      (t) => t.replicate === liveArm.replicate && t.placebo === liveArm.placebo
    );
  }, [liveTurns, liveArm]);

  /**
   * How many rounds the round axis spans.
   *
   * With a result in hand this is the run's OWN extent — padding it out to the
   * requested round count would draw empty columns for rounds that never
   * happened, which reads as missing data rather than as a shorter run. While
   * the run is live it is the requested count, so the axis does not grow a
   * column at a time under the reader.
   */
  const roundsInView = useMemo(() => {
    if (result) {
      const fromTurns = Math.max(0, ...(result.turns ?? []).map((t) => t.round));
      return Math.max(result.rounds ?? 0, fromTurns, 1);
    }
    const fromLive = liveTurns.length ? Math.max(...liveTurns.map((t) => t.round)) : 0;
    return Math.max(streamRounds ?? 0, rounds, fromLive);
  }, [result, liveTurns, streamRounds, rounds]);

  const seatData: SeatDatum[] = useMemo(() => {
    // Prefer the backend's own falsification numbers when the view is the
    // pooled real arm — it computed them, and two paths to one number is how a
    // screen ends up disagreeing with itself.
    const pooledReal = !view.placebo && view.replicate === null;
    const perMember = pooledReal ? result?.falsification?.per_member : undefined;

    return roomCast.map((member, index) => {
      if (result) {
        const row = perMember?.find((p) => p.name === member.name);
        const last = latestByMember(turnsInView, member.name);
        const pub =
          row?.public ?? finalPosition(turnsInView, member.name, "public_position");
        const prv =
          row?.private ?? finalPosition(turnsInView, member.name, "private_position");
        return {
          index,
          member,
          publicPosition: pub,
          privatePosition: prv,
          // Speaking order differs by replicate, so a numeral would be a lie
          // when several replicates are pooled.
          speakingIndex: view.replicate === null ? null : last?.speaking_index ?? null,
          speaking: false,
          // A finished run always shows a deliberation round, never the ballot.
          ballot: false,
          statement: last?.public_statement ?? null,
          note: last?.private_note ?? null,
          conceded: last?.conceded_to ?? [],
        };
      }

      const mine = liveInView.filter((t) => t.member === member.name);
      const last = mine.length ? mine[mine.length - 1] : null;
      return {
        index,
        member,
        publicPosition: last ? last.public_position : null,
        privatePosition: null,
        speakingIndex: last ? last.speaking_index : null,
        speaking: currentSpeaker === member.name,
        // Round 0 is the sealed ballot: a position, but not one they said.
        ballot: last?.round === 0,
        statement: last && last.round > 0 ? last.public_statement : null,
        note: null,
        conceded: [],
      };
    });
  }, [roomCast, result, turnsInView, liveInView, currentSpeaker, view]);

  const publicSeries = useMemo(() => {
    const names = roomCast.map((m) => m.name);
    if (result) return roundSeries(turnsInView, names, roundsInView, "public_position");
    // Live: the same reduction over the frames that have arrived — and the same
    // rule about round 0, which is a sealed ballot rather than a statement.
    return names.map((name) =>
      Array.from({ length: roundsInView + 1 }, (_, r) => {
        if (r === 0) return null;
        const hit = liveInView.filter((t) => t.member === name && t.round === r);
        return hit.length
          ? hit.reduce((a, t) => a + t.public_position, 0) / hit.length
          : null;
      })
    );
  }, [roomCast, result, turnsInView, liveInView, roundsInView]);

  const privateSeries = useMemo(() => {
    const names = roomCast.map((m) => m.name);
    if (!result) return names.map(() => Array.from({ length: roundsInView + 1 }, () => null));
    return roundSeries(turnsInView, names, roundsInView, "private_position");
  }, [roomCast, result, turnsInView, roundsInView]);

  const spreadReal = useMemo(
    () =>
      result
        ? spreadByRound(armTurns(result.turns ?? [], { placebo: false, replicate: null }), roundsInView)
        : [],
    [result, roundsInView]
  );
  const spreadPlacebo = useMemo(() => {
    if (!result) return null;
    const placeboTurns = armTurns(result.turns ?? [], { placebo: true, replicate: null });
    return placeboTurns.length ? spreadByRound(placeboTurns, roundsInView) : null;
  }, [result, roundsInView]);

  const replicateList = useMemo(() => {
    if (!result) return { real: [] as number[], placebo: [] as number[] };
    const real = new Set<number>();
    const plac = new Set<number>();
    (result.turns ?? []).forEach((t) => (t.placebo ? plac : real).add(t.replicate));
    return {
      real: [...real].sort((a, b) => a - b),
      placebo: [...plac].sort((a, b) => a - b),
    };
  }, [result]);

  /* ── cost, and the gate ───────────────────────────────────────────── */

  const seatCount = cast.length || seats;
  // One call per member for the private baseline, then one per member per
  // round — for every replicate, real and placebo alike. The control arm is
  // not free, and pretending otherwise is how a budget readout becomes a trap.
  const cost = seatCount * (1 + rounds) * (replicates + placeboReps);

  // Mirrors the server's own refusals so the button explains itself instead of
  // round-tripping to a 422: three seats is the minimum for a deliberation (two
  // members is a negotiation), and the motion has a 10-character floor.
  const blocked =
    cast.length < 3
      ? "cast at least three members — two is a negotiation, not a deliberation"
      : !approved
        ? "approve the cast first"
        : motionText.trim().length < 10
          ? "write the motion they are deliberating"
          : null;

  async function run() {
    setBusy(true);
    setError(null);
    setResult(null);
    setLiveTurns([]);
    setFailedTurns(0);
    setSkipped([]);
    setStreamRounds(null);
    setTotal(0);
    setStage("seating the room…");
    setView({ placebo: false, replicate: null });
    memberCursor.clear();
    try {
      await streamRun(
        STREAM_PATH,
        {
          // Field names are RoomRequest's, not this component's: `cast_brief`
          // and `members`, not `brief` and `cast`.
          motion: motionText,
          cast_brief: brief,
          members: withConditioning(cast),
          seats: cast.length || seats,
          rounds,
          replicates,
          placebo_replicates: placeboReps,
          research_question: question,
        },
        (evt) => {
          if (evt.type === "start") {
            setTotal(Number(evt.total ?? 0) || 0);
            setStreamRounds(Number(evt.rounds ?? 0) || null);
            setStage("the room is deliberating…");
          } else if (evt.type === "cast") {
            // The cast arrives on its OWN frame, after start — with the
            // provenance and the degeneracy check that decide how much any of
            // this is worth.
            const streamed = Array.isArray(evt.members) ? evt.members : [];
            if (streamed.length) setCast(streamed.map(adaptMember));
            setProvenance(
              (evt.provenance as RoomResult["provenance"]) ?? null
            );
            setDiversity((evt.diversity as RoomResult["diversity"]) ?? null);
          } else if (evt.type === "turn") {
            setLiveTurns((t) => [
              ...t,
              {
                replicate: Number(evt.replicate ?? 0),
                placebo: Boolean(evt.placebo),
                round: Number(evt.round ?? 0),
                member: String(evt.member ?? ""),
                speaking_index: Number(evt.speaking_index ?? 0),
                // The live frame calls it `statement`; the terminal record calls
                // it `public_statement`. Both are read.
                public_statement: String(evt.statement ?? evt.public_statement ?? ""),
                public_position: Number(evt.public_position ?? 0.5),
              },
            ]);
          } else if (evt.type === "turn_failed") {
            setFailedTurns((n) => n + 1);
          } else if (evt.type === "replicate_start") {
            setStage(
              `replicate ${Number(evt.replicate ?? 0) + 1}${
                evt.placebo ? " · control arm" : ""
              } — sealed ballot`
            );
          } else if (evt.type === "replicate_skipped") {
            setSkipped((s) => [
              ...s,
              String(evt.reason ?? evt.detail ?? `replicate ${Number(evt.replicate ?? 0) + 1}`),
            ]);
          } else if (evt.type === "round_done") {
            setStage(
              `round ${Number(evt.round ?? 0)} closed · replicate ${
                Number(evt.replicate ?? 0) + 1
              }`
            );
          } else if (evt.type === "replicate_done") {
            setStage(`replicate ${Number(evt.replicate ?? 0) + 1} complete`);
          } else if (evt.type === "stage") {
            setStage(String(evt.detail ?? evt.stage ?? ""));
          } else if (evt.type === "done") {
            // ONE adapter, so the wire's shapes and signs are reconciled in a
            // single documented place rather than in nine components.
            setResult(adaptRoomResult(evt.result));
          } else if (evt.type === "error") {
            setError(String(evt.detail));
          }
        }
      );
    } catch (e) {
      setError(e instanceof Error ? e.message : "The room study failed");
    } finally {
      setBusy(false);
      setStage(null);
    }
  }

  const armLabel = result
    ? view.placebo
      ? "CONTROL ARM"
      : "REAL ARM"
    : liveArm?.placebo
      ? "CONTROL ARM"
      : "REAL ARM";

  const phase = busy
    ? liveArm
      ? `replicate ${liveArm.replicate + 1} · round ${liveArm.round} · ${liveTurns.length}${
          total ? `/${total}` : ""
        } turns`
      : "seating"
    : result
      ? view.replicate === null
        ? `${
            view.placebo ? replicateList.placebo.length : replicateList.real.length
          } replicate(s), pooled · orders differ, so no seat numbers`
        : `${view.placebo ? "control" : "real"} replicate ${
            (view.placebo
              ? replicateList.placebo.indexOf(view.replicate)
              : replicateList.real.indexOf(view.replicate)) + 1
          } · seeded speaking order`
      : "idle";

  const viewLabel = result
    ? `${view.placebo ? "control" : "real"} · ${
        view.replicate === null
          ? "pooled"
          : `replicate ${
              (view.placebo
                ? replicateList.placebo.indexOf(view.replicate)
                : replicateList.real.indexOf(view.replicate)) + 1
            }`
      }`
    : "live";

  return (
    <div className="card p-6" data-live={busy ? "1" : "0"}>
      <div className="mb-1 flex flex-wrap items-baseline justify-between gap-2">
        <h2>The white room</h2>
        <span className="text-xs text-muted">
          what a room decides, and what it privately thinks of the decision
        </span>
      </div>
      <p className="lede mb-4">
        Cast the people, hand them a motion, and let them argue it out over
        several rounds. Everything said is public and carries a position; every
        member also holds a private one nobody at the table sees. Runs are
        replicated with different seeded speaking orders, and some replicates are
        run as a control — the members are shown statements from a different room
        entirely, because a model that agrees with anything will happily agree
        with this too.
      </p>

      {/* ── input, in the order the work actually happens ─────────────
          The MOTION first: casting sends it as context, so people are cast with
          stakes in this specific decision rather than in general. A composer
          above the motion would be a composer that always fires with an empty
          one. */}

      <label className="hud-label mb-1 block" htmlFor="room-motion">
        THE MOTION
      </label>
      <textarea
        id="room-motion"
        value={motionText}
        onChange={(e) => setMotionText(e.target.value)}
        disabled={busy}
        placeholder="One decision, stated so it can be argued for or against. “We freeze feature work for six weeks and migrate billing before the renewal cycle.”"
        className="min-h-20 w-full resize-y rounded-xl border border-hairline bg-surface-2 p-3.5 text-sm outline-none placeholder:text-muted focus:border-accent/60 disabled:opacity-50"
      />
      <p className="mt-1 font-mono text-[9px] leading-relaxed text-muted">
        Positions run 0 (against) to 1 (for), so the motion has to be something a
        reasonable room could split on. A motion nobody can oppose measures
        nothing.
      </p>

      {!motionText && (
        <div className="mt-2 flex flex-wrap gap-1.5">
          <button
            onClick={() => {
              setMotionText(SAMPLE_MOTION);
              if (!brief) setBrief(SAMPLE_BRIEF);
            }}
            disabled={busy}
            className="rounded-lg border border-dashed border-hairline px-3 py-1.5 text-xs text-muted transition hover:border-accent/50 hover:text-ink-2 disabled:opacity-40"
          >
            use a sample room
          </button>
        </div>
      )}

      <div className="mt-4">
        <CastComposer
          brief={brief}
          onBriefChange={setBrief}
          motionText={motionText}
          seats={seats}
          cast={cast}
          onCast={setCast}
          approved={approved}
          onApprove={setApproved}
          disabled={busy}
          castPath={CAST_PATH}
        />
      </div>

      <div>
        <ResearchQuestion
          value={question}
          onChange={setQuestion}
          disabled={busy}
          examples={[
            "e.g. will anyone say out loud that this plan is doomed?",
            "does the room actually agree, or is it deferring to the CFO?",
            "who has to be in the room for this to pass?",
            "is the consensus an artefact of who spoke first?",
          ]}
        />
      </div>

      <div className="mt-3 flex flex-wrap items-center gap-3 text-xs text-muted">
        <NumberField
          label="Seats"
          value={seats}
          min={3}
          max={9}
          disabled={busy || cast.length > 0}
          onChange={setSeats}
          hint={
            cast.length > 0
              ? "the cast decides the seat count now — add or remove members above"
              : "how many people to cast (3–9)"
          }
        />
        <NumberField
          label="Rounds"
          value={rounds}
          min={2}
          max={8}
          disabled={busy}
          onChange={setRounds}
          hint="deliberation rounds after the private opening ballot — the influence fit needs enough updates per member to be identifiable at all"
        />
        <NumberField
          label="Replicates"
          value={replicates}
          min={1}
          max={8}
          disabled={busy}
          onChange={setReplicates}
          hint="re-runs of the same room with different seeded speaking orders — order is a confound, not a setting"
        />
        <NumberField
          label="Placebo"
          value={placeboReps}
          min={0}
          max={4}
          disabled={busy}
          onChange={setPlaceboReps}
          hint="control replicates: each member is shown statements from a DIFFERENT room"
        />
        <span
          className="soma"
          title="seats × (1 + rounds) × (replicates + placebo) — the +1 is the private baseline before anyone speaks"
        >
          ≈{cost} twin calls
        </span>
        {placeboReps === 0 && (
          <span className="font-mono text-[9px] text-critical">
            no control arm — nothing will separate deliberation from agreeableness
          </span>
        )}
        <motion.button
          whileTap={{ scale: 0.97 }}
          disabled={busy || blocked !== null}
          onClick={run}
          title={blocked ?? undefined}
          className="ml-auto bg-accent px-5 py-2 disabled:opacity-40"
        >
          {busy ? "Deliberating…" : "Convene the room"}
        </motion.button>
      </div>

      {blocked && !busy && (
        <p className="mt-2 font-mono text-[9px] text-muted">{blocked} to convene.</p>
      )}

      {stage && (
        <p className="mt-3 font-mono text-[11px] text-accent">
          <span className="pulse-dot mr-1.5 inline-block h-1.5 w-1.5 rounded-full bg-accent align-middle" />
          {stage}
          {total > 0 && liveTurns.length > 0 && (
            <span className="text-muted">
              {" "}
              · {liveTurns.length}/{total} turns
            </span>
          )}
          {failedTurns > 0 && (
            <span className="text-critical"> · {failedTurns} turn(s) failed</span>
          )}
        </p>
      )}
      {error && <p className="mt-3 text-sm text-critical">{error}</p>}

      {/* THE CAST'S OWN CONTROL, and it belongs before the results rather than
          under them: a room whose members were built to agree will converge no
          matter what the deliberation does, so a degenerate cast invalidates the
          convergence finding the same way a matched placebo does. */}
      {(provenance ?? result?.provenance) && (
        <p className="mt-3 font-mono text-[9px] leading-relaxed text-muted">
          <span className="hud-label mr-2">CAST</span>
          {(provenance ?? result?.provenance)?.provenance}
          {(provenance ?? result?.provenance)?.note && (
            <> — {(provenance ?? result?.provenance)?.note}</>
          )}
        </p>
      )}
      {(diversity ?? result?.diversity)?.note && (
        <p
          className="mt-1.5 border-l-2 pl-2 text-[11px] leading-relaxed"
          style={{
            borderLeftColor: (diversity ?? result?.diversity)?.degenerate
              ? "rgba(240,96,95,0.6)"
              : "var(--color-hairline)",
            color: (diversity ?? result?.diversity)?.degenerate
              ? "#f0605f"
              : "var(--color-muted)",
          }}
        >
          {(diversity ?? result?.diversity)?.note}
        </p>
      )}
      {skipped.length > 0 && (
        <p className="mt-1.5 font-mono text-[9px] leading-relaxed text-critical">
          {skipped.length} replicate(s) skipped — {skipped.join(" · ")}
        </p>
      )}

      {/* ── the room, live and after ─────────────────────────────────── */}

      {(busy || result) && roomCast.length > 0 && (
        <div className="mt-5 space-y-5">
          {result?.motion && (
            <p className="verdict border-l-2 pl-3 text-ink-2" style={{ borderLeftColor: "rgb(var(--region-rgb) / 0.6)" }}>
              <span className="hud-label mr-2">MOTION</span>
              {result.motion}
            </p>
          )}

          {/* THE CONTROL LEADS. Everything below is only worth reading if the
              placebo arm failed to converge — so it goes first, not in a
              footnote. */}
          {result && (
            <PlaceboContrast
              placebo={result.placebo}
              orderEffect={result.order_effect}
              placeboReplicates={placeboReps}
            />
          )}

          {result && <FindingsPanel findings={result.findings} />}

          {/* which arm and which replicate the table is showing */}
          {result && (
            <div
              role="group"
              aria-label="Which arm and replicate the table and charts show"
              className="flex flex-wrap items-center gap-1.5"
            >
              <span className="hud-label mr-1">SHOWING</span>
              <ViewButton
                active={!view.placebo && view.replicate === null}
                onClick={() => setView({ placebo: false, replicate: null })}
                label="real · pooled"
              />
              {replicateList.real.map((r) => (
                <ViewButton
                  key={`real-${r}`}
                  active={!view.placebo && view.replicate === r}
                  onClick={() => setView({ placebo: false, replicate: r })}
                  label={`R${r + 1}`}
                />
              ))}
              {replicateList.placebo.length > 0 && (
                <>
                  <span className="mx-1 h-3 w-px bg-hairline" />
                  <ViewButton
                    active={view.placebo && view.replicate === null}
                    onClick={() => setView({ placebo: true, replicate: null })}
                    label="control · pooled"
                  />
                  {/* Placebo replicate indices continue the real arm's numbering
                      on the wire, so they are labelled by their position in the
                      control arm — "C1" is the first control run, not the
                      fourth replicate. */}
                  {replicateList.placebo.map((r, i) => (
                    <ViewButton
                      key={`plac-${r}`}
                      active={view.placebo && view.replicate === r}
                      onClick={() => setView({ placebo: true, replicate: r })}
                      label={`C${i + 1}`}
                    />
                  ))}
                </>
              )}
              <span className="ml-auto font-mono text-[9px] text-muted">
                switching replicates is how the speaking-order effect becomes
                visible — the same cast, a different order, watch the seats move
              </span>
            </div>
          )}

          <RoundTable
            seats={seatData}
            cursor={memberCursor}
            revealed={!!result}
            live={busy}
            phase={phase}
            armLabel={armLabel}
            batonFrom={previousSpeakerIndex}
            camps={result?.coalitions?.camps ?? null}
            roomGap={result?.falsification?.room_gap ?? null}
          />

          {/* the floor, while it is being held */}
          {busy && <LiveFloor turns={liveTurns} cast={roomCast} />}

          {result?.falsification && (
            <RoomGapReadout falsification={result.falsification} />
          )}

          {result && (
            <>
              <OpinionTrajectory
                cast={roomCast}
                publicSeries={publicSeries}
                privateSeries={privateSeries}
                rounds={roundsInView}
                memberCursor={memberCursor}
                falsification={result.falsification}
                viewLabel={viewLabel}
              />

              <InfluenceNetwork
                cast={roomCast}
                influence={result.influence}
                centrality={result.centrality}
                cascades={result.cascades}
                cursor={memberCursor}
              />

              <div className="grid gap-5 lg:grid-cols-2">
                <ConformityChart
                  conformity={result.conformity}
                  spreadReal={spreadReal}
                  spreadPlacebo={spreadPlacebo}
                  rounds={roundsInView}
                />
                <div className="space-y-5">
                  <CounterfactualStrip
                    cast={roomCast}
                    counterfactuals={result.counterfactuals}
                    equilibrium={result.equilibrium}
                    cursor={memberCursor}
                  />
                  <CoalitionPanel
                    coalitions={result.coalitions}
                    cast={roomCast}
                    cursor={memberCursor}
                  />
                </div>
              </div>

              <TurnLog
                turns={turnsInView}
                cast={roomCast}
                cursor={memberCursor}
                label={viewLabel}
              />

              {/* THE REFUSAL LEDGER. Every estimator that declined, with its
                  reason — collected in one place because "this run did not
                  support that" is a result, and a screen that only shows what
                  computed is a screen that flatters itself. */}
              {(result.refusals?.length || result.dropped_replicates?.length || result.turns_failed) && (
                <section className="border border-hairline bg-black/30">
                  <header className="flex flex-wrap items-baseline gap-x-3 border-b border-hairline px-3 py-2">
                    <span className="hud-label">WHAT THIS RUN COULD NOT SUPPORT</span>
                    <span className="font-mono text-[9px] text-muted">
                      {result.refusals?.length ?? 0} estimator refusal
                      {(result.refusals?.length ?? 0) === 1 ? "" : "s"}
                      {result.dropped_replicates?.length
                        ? ` · ${result.dropped_replicates.length} replicate(s) dropped whole`
                        : ""}
                      {result.turns_failed ? ` · ${result.turns_failed} turn(s) failed` : ""}
                    </span>
                  </header>
                  <ul className="divide-y divide-hairline/40">
                    {(result.refusals ?? []).map((r, i) => (
                      <li key={i} className="px-3 py-1.5">
                        <span className="font-mono text-[9px] uppercase tracking-[0.12em] text-critical">
                          {r.what}
                        </span>
                        <p className="text-[11px] leading-snug text-muted">{r.reason}</p>
                      </li>
                    ))}
                  </ul>
                </section>
              )}

              <div className="grid gap-x-6 gap-y-2 border-t border-hairline pt-3 sm:grid-cols-2">
                <div>
                  <span className="hud-label">HOW THIS WAS MEASURED</span>
                  <p className="mt-0.5 font-mono text-[10px] leading-relaxed text-muted">
                    {result.method}
                  </p>
                </div>
                <div>
                  <span className="hud-label">WHAT THIS CANNOT SHOW</span>
                  <p className="mt-0.5 font-mono text-[10px] leading-relaxed text-muted">
                    {result.limits}
                  </p>
                </div>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}

/* ── small parts ───────────────────────────────────────────────────────── */

function NumberField({
  label,
  value,
  min,
  max,
  onChange,
  disabled,
  hint,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  onChange: (v: number) => void;
  disabled?: boolean;
  hint: string;
}) {
  return (
    <label className="flex items-center gap-2" title={hint}>
      {label}
      <input
        type="number"
        min={min}
        max={max}
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(Math.min(max, Math.max(min, +e.target.value || min)))}
        className="w-14 border border-hairline bg-surface-2 px-2 py-1.5 text-sm tabular-nums text-ink outline-none disabled:opacity-40"
      />
    </label>
  );
}

function ViewButton({
  active,
  onClick,
  label,
}: {
  active: boolean;
  onClick: () => void;
  label: string;
}) {
  return (
    <button
      onClick={onClick}
      aria-pressed={active}
      className={`border px-2 py-0.5 font-mono text-[9px] uppercase tracking-[0.12em] transition ${
        active
          ? "border-transparent bg-bone text-void"
          : "border-hairline text-muted hover:border-accent/50 hover:text-ink-2"
      }`}
    >
      {label}
    </button>
  );
}

/**
 * The room's mean falsification, with its interval. A gap without an interval is
 * a number that cannot be argued with, and this one is the study's headline —
 * so it never appears without one.
 */
function RoomGapReadout({ falsification }: { falsification: RoomFalsification }) {
  const { room_gap: gap, room_gap_ci: ci, p_value: p } = falsification;

  if (falsification.refused) {
    return (
      <div className="border border-dashed border-hairline px-4 py-3" role="note">
        <span className="hud-label text-critical">GAP NOT MEASURED</span>
        <p className="mt-1 text-[12px] leading-relaxed text-ink-2">
          {falsification.refused}
        </p>
      </div>
    );
  }

  // 0.05 is the grid a declared position lands on; below it a gap is rounding.
  const material = Math.abs(gap) >= 0.05;
  const resolved = ci !== null && (ci[0] > 0 || ci[1] < 0);
  return (
    <div
      className="border px-4 py-3"
      style={{
        borderColor: material && resolved ? "rgba(240,96,95,0.45)" : "var(--color-hairline)",
        background: material && resolved ? "rgba(240,96,95,0.05)" : "rgba(0,0,0,0.25)",
      }}
    >
      <div className="flex flex-wrap items-baseline gap-x-3">
        <span className="hud-label">SAID − THOUGHT, ROOM MEAN</span>
        <span
          className="font-display text-[26px] font-semibold leading-none tabular-nums"
          style={{ color: material ? "#f0605f" : "var(--color-ink)" }}
        >
          {signed(gap)}
        </span>
        {ci ? (
          <span className="font-mono text-[10px] tabular-nums text-muted">
            95% CI [{ci[0].toFixed(2)}, {ci[1].toFixed(2)}]
          </span>
        ) : (
          <span className="font-mono text-[10px] text-muted">
            {falsification.gap_ci_note ?? "no interval — too few members to bootstrap one"}
          </span>
        )}
        {p !== null && (
          <span className="font-mono text-[10px] tabular-nums text-muted">
            p = {p.toPrecision(2)}
          </span>
        )}
        {falsification.supported === false && (
          <span className="font-mono text-[9px] uppercase tracking-[0.14em] text-muted">
            not supported
          </span>
        )}
      </div>
      {/* SIGN: gap = public − private. */}
      <p className="mt-1 text-[12px] leading-relaxed text-ink-2">
        {!resolved
          ? "The interval spans zero: this room is consistent with everyone saying exactly what they think."
          : gap > 0
            ? "The room is publicly MORE in favour than it privately is. The motion has more public support than actual support — the failure mode that only shows up after the decision is made."
            : "The room is privately MORE in favour than it says out loud: the opposition being voiced is louder than the opposition being held."}
      </p>
      {falsification.gap_ci_note && ci && (
        <p className="mt-0.5 font-mono text-[9px] leading-relaxed text-muted">
          {falsification.gap_ci_note}
        </p>
      )}
    </div>
  );
}

/** The last few things said, newest first — the floor while it is being held. */
function LiveFloor({ turns, cast }: { turns: LiveTurn[]; cast: RoomMember[] }) {
  const recent = turns.slice(-3).reverse();
  if (!recent.length) return null;
  return (
    <div className="border border-hairline bg-black/30">
      <div className="flex items-baseline gap-2 border-b border-hairline px-3 py-1.5">
        <span className="hud-label">THE FLOOR</span>
        <span className="font-mono text-[9px] text-muted">
          public statements only — private positions arrive with the result
        </span>
      </div>
      <AnimatePresence initial={false}>
        {recent.map((t, k) => {
          const i = cast.findIndex((m) => m.name === t.member);
          return (
            <motion.div
              key={`${t.replicate}-${t.round}-${t.member}-${t.speaking_index}`}
              initial={{ opacity: 0, y: -6 }}
              animate={{ opacity: k === 0 ? 1 : 0.55, y: 0 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.22 }}
              className="border-b border-hairline/40 px-3 py-2 last:border-b-0"
            >
              <div className="flex flex-wrap items-baseline gap-x-2">
                <span
                  className="inline-block h-2 w-2 shrink-0"
                  style={{ background: i >= 0 ? identityHue(i) : "var(--color-muted)" }}
                  aria-hidden
                />
                <span className="text-[11px] text-ink-2">{t.member}</span>
                <span
                  className="font-mono text-[10px] tabular-nums"
                  style={{ color: positionInk(t.public_position) }}
                >
                  {fmt2(t.public_position)} {positionWord(t.public_position)}
                </span>
                <span className="ml-auto font-mono text-[9px] text-muted">
                  {t.round === 0 ? "sealed ballot" : `round ${t.round}`} ·{" "}
                  {t.placebo ? "control" : "real"} arm
                </span>
              </div>
              {t.public_statement && (
                <p className="mt-0.5 text-[12px] leading-snug text-ink">{t.public_statement}</p>
              )}
            </motion.div>
          );
        })}
      </AnimatePresence>
    </div>
  );
}

/**
 * What a screen reader hears when the members cursor lands on a seat. The
 * ring/fill gap is a colour difference, so it has to exist as a sentence too —
 * this is that sentence, and it is also the chart's aria-label.
 */
function describeMember(
  i: number,
  cast: RoomMember[],
  result: RoomResult | null,
  view: View,
  liveTurns: LiveTurn[]
): string {
  const m = cast[i];
  if (!m) return "no member";
  if (result) {
    const turns = armTurns(result.turns ?? [], view);
    const row = result.falsification?.per_member?.find((p) => p.name === m.name);
    const pub = row?.public ?? finalPosition(turns, m.name, "public_position");
    const prv = row?.private ?? finalPosition(turns, m.name, "private_position");
    if (pub === null) return `${m.name}, ${m.role}, did not speak`;
    if (prv === null) return `${m.name}, ${m.role}, said ${pub.toFixed(2)}`;
    return `${m.name}, ${m.role}, said ${pub.toFixed(2)} ${positionWord(
      pub
    )}, privately ${prv.toFixed(2)} ${positionWord(prv)}, gap ${(prv - pub).toFixed(2)}`;
  }
  const mine = liveTurns.filter((t) => t.member === m.name);
  const last = mine.length ? mine[mine.length - 1] : null;
  return last
    ? `${m.name}, ${m.role}, currently saying ${last.public_position.toFixed(2)} ${positionWord(
        last.public_position
      )}; private position not yet known`
    : `${m.name}, ${m.role}, has not spoken`;
}
