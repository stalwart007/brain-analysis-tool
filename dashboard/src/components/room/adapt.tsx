"use client";

/**
 * THE ONE PLACE THE WIRE MEETS THE SCREEN.
 *
 * The room endpoint returns an orchestration envelope from boardroom.py with
 * every estimator nested under `analysis` (room.py's `room_analysis`). Several
 * keys are named differently from the shapes the five visualisations want, and
 * two of them carry the OPPOSITE sign convention from the obvious reading. Nine
 * components each doing their own `?? fallback` chain is how a UI ends up
 * disagreeing with itself about what a number means, so all of it happens here,
 * once, with the mapping written down.
 *
 * MAPPINGS, wire → view:
 *   motion_preview            → motion            (the full motion is not returned)
 *   analysis.absence[]        → counterfactuals[]  (delta_room_mean → room_delta,
 *                                                   room_mean_without → landing,
 *                                                   room_mean_with   → anchor)
 *   analysis.cascade          → cascades           (cascade_r2_gain → score)
 *   analysis.equilibrium      → equilibrium.positions
 *   analysis.falsification    → falsification      (per_member_gap + roster's
 *                                                   final_public/final_private)
 *   analysis.conformity       → conformity         (observed_rate_per_round,
 *                                                   benchmark_rate_per_round,
 *                                                   conformity_ratio, sd_by_round)
 *   analysis.coalitions       → coalitions         (public_camps → camps, centre →
 *                                                   mean_position; power indices
 *                                                   only exist with a decision
 *                                                   threshold)
 *   analysis.placebo          → placebo            (convergence_real/placebo,
 *                                                   gap_real/placebo, separation)
 *   analysis.order_effect     → order_effect       (eta_squared → effect_size)
 *   analysis.n_rounds         → rounds − 1         (the wire COUNTS rounds and
 *                                                   counts the sealed ballot as
 *                                                   one of them; this screen
 *                                                   indexes them from 0)
 *
 * SIGNS, stated once so nothing downstream has to guess:
 *   gap        = public − private. Positive is public support that is not real.
 *   convergence = sd_final / sd_opening, i.e. spread REMAINING. Lower converged
 *                harder, so the damning placebo case is placebo ≤ real.
 *
 * Every read is defensive: a field that moves shows up as a refusal or an absent
 * panel, never as a crash or as a zero pretending to be a measurement. The
 * flat shape from the original written contract is also accepted, so this keeps
 * working if the endpoint is the one that changes.
 */

import type {
  RoomCascades,
  RoomCentrality,
  RoomCoalitions,
  RoomConformity,
  RoomCounterfactual,
  RoomFalsification,
  RoomInfluence,
  RoomMember,
  RoomOrderEffect,
  RoomPlacebo,
  RoomResult,
  RoomTurn,
} from "@/lib/api";
import type { FindingsBlock } from "@/lib/api";

type Dict = Record<string, unknown>;

const isDict = (v: unknown): v is Dict => typeof v === "object" && v !== null && !Array.isArray(v);
const arr = (v: unknown): unknown[] => (Array.isArray(v) ? v : []);
const num = (v: unknown): number | null =>
  typeof v === "number" && Number.isFinite(v) ? v : null;
const str = (v: unknown, fallback = ""): string => (typeof v === "string" ? v : fallback);
const dict = (v: unknown): Dict => (isDict(v) ? v : {});

/** Clamp a declared disposition into the 0‥1 the seat geometry assumes. */
function unit(v: unknown, fallback = 0.5): number {
  const n = typeof v === "number" ? v : Number(v);
  return Number.isFinite(n) ? Math.max(0, Math.min(1, n)) : fallback;
}

export function adaptMember(raw: unknown): RoomMember {
  const m = dict(raw);
  return {
    name: str(m.name, "unnamed"),
    role: str(m.role),
    archetype: str(m.archetype),
    stake: str(m.stake),
    // Carried through untouched: the server requires it on a supplied cast, and
    // the researcher has no business editing a conditioning prompt by hand.
    system_prompt: str(m.system_prompt),
    openness: unit(m.openness),
    assertiveness: unit(m.assertiveness),
    seniority: unit(m.seniority),
    risk_appetite: unit(m.risk_appetite),
    expertise: arr(m.expertise).map((e) => String(e)),
  };
}

/**
 * Every member the server receives must carry a conditioning prompt. A seat the
 * researcher added by hand has none, so one is written from the fields they did
 * fill in — visibly, at dispatch, rather than being invented in the editor where
 * it would look authored.
 */
export function withConditioning(cast: RoomMember[]): RoomMember[] {
  return cast.map((m) =>
    m.system_prompt.trim()
      ? m
      : {
          ...m,
          system_prompt: [
            `You are ${m.name}${m.role ? `, ${m.role}` : ""}.`,
            m.archetype ? `${m.archetype}.` : "",
            m.stake ? `What you stand to gain or lose here: ${m.stake}.` : "",
            m.expertise.length ? `You are credible in ${m.expertise.join(", ")}.` : "",
            "Speak in character, in your own voice, and hold your own counsel.",
          ]
            .filter(Boolean)
            .join(" "),
        }
  );
}

function adaptTurn(raw: unknown): RoomTurn {
  const t = dict(raw);
  return {
    replicate: num(t.replicate) ?? 0,
    // `placebo` on the record; `arm: "placebo"` on the analysis payload.
    placebo: Boolean(t.placebo) || str(t.arm) === "placebo",
    round: num(t.round) ?? 0,
    member: str(t.member),
    speaking_index: num(t.speaking_index) ?? num(t.order) ?? 0,
    public_statement: str(t.public_statement) || str(t.statement),
    public_position: num(t.public_position) ?? 0.5,
    private_position: num(t.private_position) ?? Number.NaN,
    private_note: str(t.private_note),
    conceded_to: arr(t.conceded_to).map((n) => String(n)),
  };
}

function adaptInfluence(a: Dict, names: string[]): RoomInfluence | null {
  const inf = a.influence;
  if (!isDict(inf)) return null;
  const W = arr(inf.W).map((row) => arr(row).map((v) => num(v) ?? 0));
  const refused = typeof inf.refused === "string" ? inf.refused : null;
  return {
    W,
    susceptibility: arr(inf.susceptibility).map((v) => num(v) ?? 0),
    names: arr(inf.names).length ? arr(inf.names).map((n) => String(n)) : names,
    r2_per_member: arr(inf.r2_per_member).map((v) => num(v)),
    n_obs_per_member: num(inf.n_obs_per_member) ?? 0,
    // No W is a refusal even when the server did not phrase one — drawing a
    // network from an absent matrix is the failure this guard exists for.
    refused: refused ?? (W.length ? null : "The influence matrix was not returned."),
    method: str(inf.method),
    unidentified_rows: arr(inf.unidentified_rows).map((r) => {
      const row = dict(r);
      return {
        name: str(row.name),
        index: num(row.index) ?? -1,
        reason: str(row.reason),
      };
    }),
    // Absent means "not stated"; only an explicit false suppresses the weights.
    w_entries_supported:
      typeof inf.w_entries_supported === "boolean" ? inf.w_entries_supported : undefined,
    w_entry_caveat: typeof inf.w_entry_caveat === "string" ? inf.w_entry_caveat : null,
    susceptibility_supported: Array.isArray(inf.susceptibility_supported)
      ? inf.susceptibility_supported.map(Boolean)
      : null,
  };
}

function adaptCentrality(a: Dict): RoomCentrality | null {
  const c = a.centrality;
  if (!isDict(c)) return null;
  const centrality = arr(c.centrality).map((v) => num(v) ?? 0);
  const note = typeof c.note === "string" ? c.note : null;
  // A null centrality WITH a note is a finding ("the room has split"), so the
  // block survives; a null centrality with nothing to say does not.
  if (!centrality.length && !note) return null;
  return {
    centrality: centrality.length ? centrality : null,
    spectral_gap: num(c.spectral_gap),
    converges: Boolean(c.converges),
    consensus_forecast: num(c.consensus_forecast),
    note,
  };
}

function adaptEquilibrium(a: Dict): number[] | null {
  const eq = a.equilibrium;
  if (Array.isArray(eq)) return eq.map((v) => num(v) ?? 0);
  if (!isDict(eq)) return null;
  const positions = arr(eq.positions).map((v) => num(v) ?? 0);
  return positions.length ? positions : null;
}

function adaptCounterfactuals(a: Dict): RoomCounterfactual[] | null {
  const rows = arr(a.absence ?? a.counterfactuals);
  if (!rows.length) return null;
  return rows.map((r) => {
    const row = dict(r);
    return {
      name: str(row.name),
      equilibrium: Array.isArray(row.equilibrium)
        ? row.equilibrium.map((v) => num(v) ?? 0)
        : null,
      room_delta: num(row.delta_room_mean) ?? num(row.room_delta) ?? 0,
      landing: num(row.room_mean_without),
      anchor: num(row.room_mean_with),
      note: typeof row.note === "string" ? row.note : null,
    };
  });
}

function adaptFalsification(a: Dict, names: string[]): RoomFalsification | null {
  const f = a.falsification;
  if (!isDict(f)) return null;
  if (typeof f.refused === "string" && f.refused) {
    return {
      per_member: [],
      room_gap: Number.NaN,
      room_gap_ci: null,
      p_value: null,
      by_round: [],
      refused: f.refused,
      method: str(f.method) || null,
    };
  }

  const gaps = arr(f.per_member_gap).map((v) => num(v) ?? 0);
  const roster = arr(a.roster).map(dict);
  const labels = arr(f.names).length ? arr(f.names).map((n) => String(n)) : names;

  // Already a list of rows (the originally written contract) — take it as-is.
  const given = arr(f.per_member)
    .map(dict)
    .filter((r) => typeof r.name === "string");

  const per_member = given.length
    ? given.map((r) => ({
        name: str(r.name),
        gap: num(r.gap) ?? 0,
        public: num(r.public) ?? 0,
        private: num(r.private) ?? 0,
      }))
    : labels.map((name, i) => {
        const row = roster.find((r) => str(r.name) === name) ?? {};
        const pub = num(row.final_public);
        const prv = num(row.final_private);
        return {
          name,
          gap: num(row.gap) ?? gaps[i] ?? 0,
          public: pub ?? 0.5,
          private: prv ?? 0.5,
        };
      });

  const ci = arr(f.gap_ci ?? f.room_gap_ci);
  const byRound = arr(f.gap_by_round);

  return {
    per_member,
    room_gap: num(f.room_mean_gap) ?? num(f.room_gap) ?? 0,
    room_gap_ci:
      ci.length === 2 && num(ci[0]) !== null && num(ci[1]) !== null
        ? [num(ci[0]) as number, num(ci[1]) as number]
        : null,
    p_value: num(f.p_value),
    // `gap_by_round` covers the DELIBERATION rounds only — the sealed ballot is
    // held out as the innate baseline — so entry i is round i + 1 on this
    // screen's axis, where 0 is the ballot. Off by one here would attribute
    // every round's gap to the round before it.
    by_round: byRound.map((g, r) => ({ round: r + 1, gap: num(g) ?? 0 })),
    supported: typeof f.supported === "boolean" ? f.supported : undefined,
    direction: typeof f.direction === "string" ? f.direction : null,
    gap_ci_note: typeof f.gap_ci_note === "string" ? f.gap_ci_note : null,
    method: str(f.method) || null,
    refused: null,
  };
}

function adaptConformity(a: Dict): RoomConformity | null {
  const c = a.conformity;
  if (!isDict(c)) return null;
  const refused = typeof c.refused === "string" ? c.refused : null;
  const observed = num(c.observed_rate_per_round) ?? num(c.observed_rate);
  const bayes = num(c.benchmark_rate_per_round) ?? num(c.bayes_rate);
  const ratio = num(c.conformity_ratio) ?? num(c.ratio);
  // sd_by_round covers the deliberation rounds; the ballot's spread is reported
  // separately as sd_innate. Prepending it makes the series axis-aligned — index
  // = round — so the chart can plot the collapse from where it actually started.
  const sdRounds = arr(c.sd_by_round).map((v) => num(v));
  const sd = sdRounds.length ? [num(c.sd_innate), ...sdRounds] : [];
  if (refused || observed === null || bayes === null || ratio === null) {
    return refused ? {
      observed_rate: Number.NaN,
      bayes_rate: Number.NaN,
      ratio: Number.NaN,
      verdict: str(c.verdict),
      sd_by_round: sd.length ? sd : null,
      refused,
    } : null;
  }
  return {
    observed_rate: observed,
    bayes_rate: bayes,
    ratio,
    verdict: str(c.verdict),
    sd_by_round: sd.length ? sd : null,
    refused: null,
  };
}

function adaptCascades(a: Dict): RoomCascades | null {
  const c = a.cascade ?? a.cascades;
  if (!isDict(c)) return null;
  const rows = arr(c.per_member)
    .map(dict)
    .map((r) => ({
      name: str(r.name),
      score: num(r.cascade_r2_gain) ?? num(r.score),
    }))
    // A refused row carries a null gain. It is not a zero — zero means "spoke
    // first and moved nobody", which is a finding.
    .filter((r): r is { name: string; score: number } => r.score !== null);
  return rows.length ? { per_member: rows } : null;
}

function adaptCoalitions(a: Dict, names: string[]): RoomCoalitions | null {
  const c = a.coalitions;
  if (!isDict(c)) return null;
  const asCamps = (v: unknown) =>
    arr(v)
      .map(dict)
      .map((camp) => ({
        members: arr(camp.members).map((m) => String(m)),
        mean_position: num(camp.centre) ?? num(camp.mean_position) ?? 0.5,
      }))
      .filter((camp) => camp.members.length > 0);

  const camps = asCamps(c.public_camps ?? c.camps);
  const privateCamps = asCamps(c.private_camps);

  /**
   * Power indices arrive as rows of `{name, index}` — where `index` is the
   * VALUE, not the member's position in the cast. A list of bare numbers or a
   * name → value map are both accepted too.
   */
  const asMap = (v: unknown): Map<string, number> => {
    const out = new Map<string, number>();
    if (Array.isArray(v)) {
      v.forEach((x, i) => {
        if (isDict(x)) {
          const name = str(x.name) || names[i];
          const value = num(x.index) ?? num(x.value) ?? num(x.power);
          if (name && value !== null) out.set(name, value);
          return;
        }
        const n = num(x);
        if (n !== null && names[i]) out.set(names[i], n);
      });
    } else if (isDict(v)) {
      Object.entries(v).forEach(([k, x]) => {
        const n = num(x);
        if (n !== null) out.set(k, n);
      });
    }
    return out;
  };
  const ss = asMap(c.shapley_shubik);
  const bz = asMap(c.banzhaf);
  const power = arr(c.power).length
    ? arr(c.power)
        .map(dict)
        .map((p) => ({
          name: str(p.name),
          shapley_shubik: num(p.shapley_shubik) ?? 0,
          banzhaf: num(p.banzhaf) ?? 0,
        }))
    : [...ss.keys()].map((name) => ({
        name,
        shapley_shubik: ss.get(name) ?? 0,
        banzhaf: bz.get(name) ?? 0,
      }));

  const vote = dict(c.vote);

  if (!camps.length && !power.length) return null;
  return {
    camps,
    private_camps: privateCamps.length ? privateCamps : null,
    hidden_split: typeof c.hidden_split === "boolean" ? c.hidden_split : null,
    power: power.length ? power : null,
    power_note: typeof c.power_note === "string" ? c.power_note : null,
    weight_source: typeof vote.weight_source === "string" ? vote.weight_source : null,
  };
}

function adaptOrderEffect(a: Dict): RoomOrderEffect | null {
  const o = a.order_effect;
  if (!isDict(o)) return null;
  const effect = num(o.eta_squared) ?? num(o.effect_size);
  if (effect === null) return null;
  return { effect_size: effect, p_value: num(o.eta_p_value) ?? num(o.p_value) };
}

function adaptPlacebo(a: Dict): RoomPlacebo | null {
  const p = a.placebo;
  if (!isDict(p)) return null;
  // `ran: false` is the server saying "no control arm", which the UI must render
  // as loudly as a failed control rather than as a missing panel.
  if (p.ran === false) return null;
  const real = num(p.convergence_real) ?? num(p.real_convergence);
  const control = num(p.convergence_placebo) ?? num(p.placebo_convergence);
  if (real === null && control === null && !str(p.verdict)) return null;
  return {
    real_convergence: real,
    placebo_convergence: control,
    real_gap: num(p.gap_real) ?? num(p.real_gap),
    placebo_gap: num(p.gap_placebo) ?? num(p.placebo_gap),
    verdict: str(p.verdict),
    separation: typeof p.separation === "string" ? p.separation : null,
    convergence_delta: num(p.convergence_delta),
    gap_delta: num(p.gap_delta),
    gap_note: typeof p.gap_note === "string" ? p.gap_note : null,
    ran: p.ran !== false,
    refused: typeof p.refused === "string" ? p.refused : null,
  };
}

/**
 * The whole envelope. `raw` is whatever came back on the `done` frame.
 */
export function adaptRoomResult(raw: unknown): RoomResult {
  const env = dict(raw);
  // Nested under `analysis` by the orchestrator; accept a flat envelope too.
  const a = isDict(env.analysis) ? env.analysis : env;

  const cast = arr(env.cast ?? env.members).map(adaptMember);
  const names = arr(a.names).length
    ? arr(a.names).map((n) => String(n))
    : cast.map((m) => m.name);

  const turns = arr(env.turns).map(adaptTurn);
  // `n_rounds` counts the DELIBERATION rounds; the sealed ballot is round 0 on
  // this screen's axis and is held out of that count. So the largest round index
  // equals n_rounds, and the axis has n_rounds + 1 slots.
  const roundCount = num(a.n_rounds) ?? num(env.rounds);
  const maxTurnRound = turns.length ? Math.max(...turns.map((t) => t.round)) : 0;

  const findings = (isDict(env.findings) ? env.findings : isDict(a.findings) ? a.findings : null) as
    | FindingsBlock
    | null;

  const provenance = isDict(env.cast_provenance)
    ? {
        provenance: str(env.cast_provenance.provenance),
        members: num(env.cast_provenance.members) ?? cast.length,
        note: typeof env.cast_provenance.note === "string" ? env.cast_provenance.note : null,
      }
    : null;

  const diversity = isDict(env.cast_diversity)
    ? {
        seats: num(env.cast_diversity.seats) ?? undefined,
        distinct_archetypes: num(env.cast_diversity.distinct_archetypes) ?? undefined,
        distinct_roles: num(env.cast_diversity.distinct_roles) ?? undefined,
        mean_archetype_overlap: num(env.cast_diversity.mean_archetype_overlap) ?? undefined,
        degenerate: Boolean(env.cast_diversity.degenerate),
        note: typeof env.cast_diversity.note === "string" ? env.cast_diversity.note : null,
      }
    : null;

  const rows = (v: unknown) =>
    arr(v)
      .map(dict)
      .map((r) => ({ what: str(r.what), reason: str(r.reason) }))
      .filter((r) => r.what || r.reason);

  return {
    cast,
    motion: str(a.motion) || str(env.motion) || str(env.motion_preview),
    method: str(a.method) || str(env.method),
    limits: str(a.limits) || str(env.limits),
    provenance,
    diversity,
    instrument_check: isDict(env.instrument_check) ? env.instrument_check : null,
    refusals: rows(a.refusals),
    dropped_replicates: rows(a.dropped_replicates),
    turns_failed: num(env.turns_failed),
    consensus: isDict(a.consensus)
      ? (Object.fromEntries(
          Object.entries(a.consensus).map(([k, v]) => [k, num(v)])
        ) as Record<string, number | null>)
      : null,
    findings,
    influence: adaptInfluence(a, names),
    centrality: adaptCentrality(a),
    equilibrium: adaptEquilibrium(a),
    counterfactuals: adaptCounterfactuals(a),
    falsification: adaptFalsification(a, names),
    conformity: adaptConformity(a),
    cascades: adaptCascades(a),
    coalitions: adaptCoalitions(a, names),
    order_effect: adaptOrderEffect(a),
    placebo: adaptPlacebo(a),
    turns,
    run_id: str(env.run_id) || undefined,
    rounds: Math.max(1, roundCount ?? maxTurnRound, maxTurnRound),
    replicates: num(a.n_replicates) ?? num(env.replicates) ?? undefined,
  };
}
