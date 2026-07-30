/** Typed client for the FastAPI backend (proxied under /api/cs). */

import { reportBackendResult } from "./health";

export interface FeaturePayload {
  event_count: number;
  micro_hesitation_count: number;
  rage_click_bursts: number;
  scroll_velocity_mean: number;
  backtrack_ratio: number;
  zone_dwell_ms: Record<string, number>;
  [k: string]: unknown;
}

export interface Trait {
  score: number;
  confidence: number;
}

export interface BehavioralSignal {
  deliberation: string;
  frustration_signal: string;
  exploration_style: string;
  price_sensitivity_signal: string;
  friction_hotspot: string | null;
  evidence: string;
  likely_mindset: string;
  confidence: number;
}

export interface PersonaSeed {
  persona_id: string;
  label: string;
  signal: BehavioralSignal;
  traits: {
    openness: Trait;
    conscientiousness: Trait;
    extraversion: Trait;
    agreeableness: Trait;
    neuroticism: Trait;
    disclaimer: string;
  };
  system_prompt: string;
}

export interface SessionRow {
  id: string;
  site_id: string;
  page_path: string;
  received_at: string;
  features: FeaturePayload;
  signal: BehavioralSignal | null;
  persona: PersonaSeed | null;
  cognition: CognitionProfile | null;
}

// ── computational cognition layer ─────────────────────────────────────────

export interface DDMResult {
  drift_rate: number;
  boundary_separation: number;
  non_decision_time_s: number;
  mean_decision_time_s: number;
  interpretation: string;
  model: string;
  n: number;
}

export interface HMMState {
  label: string;
  dominant_event: string;
  dominance: number;
}

export interface HMMResult {
  n_states: number;
  log_likelihood: number;
  ll_trace: number[];
  bic: number;
  bic_by_k: Record<string, number>;
  transition: number[][];
  emission: number[][];
  state_path: number[];
  states: HMMState[];
  em_iterations: number;
}

export interface InfoDynamics {
  transition?: number[][];
  stationary?: number[];
  entropy_rate_bits?: number;
  predictability?: number;
  lz76_complexity?: number | null;
  kinematic_sample_entropy?: number | null;
}

export interface SpectralProfileResult {
  /** null when the band's frequencies are unresolvable in this record's span —
   *  distinct from 0, which would mean "measured, and empty". */
  drift_band: number | null;
  rhythm_band: number | null;
  burst_band: number | null;
  peak_frequency_hz: number | null;
  n_samples: number;
  method: string;
}

export interface ForagingResult {
  alpha: number;
  loglik_ratio_per_sample: number;
  regime: string;
  interpretation: string;
  n: number;
}

export interface HabituationResult {
  decay_rate_per_s: number;
  half_life_s: number | null;
  initial_engagement: number;
  r2: number;
  habituating: boolean;
}

export interface CognitionProfile {
  models_run: string[];
  models_skipped: string[];
  ddm: DDMResult | null;
  hmm: HMMResult | null;
  information: InfoDynamics | null;
  spectral: SpectralProfileResult | null;
  foraging: ForagingResult | null;
  habituation: HabituationResult | null;
  summary: Record<string, unknown>;
}

/* ── findings ──────────────────────────────────────────────────────────
   Every study result now carries the interpretation layer alongside its
   statistics. The evidence rows are harvested from the study's own numbers by
   server-side code, FDR-controlled as a family, and the written findings may
   only cite rows that exist — so a `FindingRow` can always be resolved back to
   the measurement that licenses it. That resolution is what the UI renders,
   and it is why these types live in the API contract rather than in the
   component that happens to draw them. */

export interface EvidenceRow {
  id: string;
  label: string;
  value: number | null;
  display: string;
  ci: [number, number] | null;
  p_value: number | null;
  /** FDR-adjusted. Read this, not `p_value`, when deciding anything. */
  q_value: number | null;
  n: number | null;
  where: string;
  interpretation: string;
  supported: boolean;
  support_reason: string;
  magnitude: number;
}

export interface FindingRow {
  headline: string;
  detail: string;
  evidence_ids: string[];
  direction: "strength" | "risk" | "opportunity" | "neutral";
  confidence: "high" | "moderate" | "low";
  answers_question: boolean;
  recommended_action: string;
  /** Set when every cited row failed FDR control — the claim is not retracted,
   *  it is demoted, because "we looked and could not tell" is information. */
  downgraded?: string;
}

export interface FindingsBlock {
  question: string;
  evidence: EvidenceRow[];
  multiplicity: {
    n_tests: number;
    n_rejected: number;
    alpha: number;
    cutoff: number | null;
    method: string;
  };
  n_supported: number;
  method: string;
  /** Synthesis failed; the evidence table is still valid and still rendered. */
  error?: string;
  report: {
    answer: string;
    findings: FindingRow[];
    what_would_change_the_answer: string;
    limits: string;
    dropped_uncited: number;
  } | null;
}

/** Mixed into every study result. */
export interface WithFindings {
  findings?: FindingsBlock | null;
}

export interface SwarmAggregate {
  findings?: FindingsBlock | null;
  run_id: string;
  scenario_preview: string;
  /** Twins whose reaction was USABLE — a survivor count. Always read it with
   *  twins_requested/twins_failed: a run where most twins errored otherwise
   *  looks identical to a healthy run of the smaller size. */
  twin_count: number;
  /** Twin calls dispatched. Absent on runs recorded before this was tracked. */
  twins_requested?: number | null;
  /** Twin calls that errored or returned unusable output. */
  twins_failed?: number | null;
  cognitive_load: string;
  mean_engagement: number;
  mean_intent: number;
  action_distribution: Record<string, number>;
  top_dropoff_points: string[];
  top_frictions: string[];
  reactions: unknown[];
  // statistical layer
  intent_ci?: [number, number] | null;
  engagement_ci?: [number, number] | null;
  polarization?: number;
  bimodality?: number | null;
  consensus?: string;
  segments?: SegmentReport | null;
}

export interface SegmentGroup {
  label: string;
  size: number;
  share: number;
  mean_intent: number;
  mean_engagement: number;
  dominant_action: string;
}

/** k-means++ discovered audience segments (k chosen by silhouette). */
export interface SegmentReport {
  k: number;
  silhouette: number;
  groups: SegmentGroup[];
}

export interface BayesianArm {
  name: string;
  successes: number;
  trials: number;
  posterior_mean: number;
  p_best: number;
  ci_low: number;
  ci_high: number;
  expected_loss: number;
}

export interface BayesianResult {
  arms: BayesianArm[];
  winner: string;
  p_win: number;
  expected_loss: number;
  conclusive: boolean;
  rope: number;
  draws: number;
  verdict: string;
}

export interface SurvivalStep {
  step_index: number;
  entered: number;
  dropped: number;
  hazard: number;
  survival: number;
  survival_ci_low: number;
  survival_ci_high: number;
}

export interface SurvivalResult {
  steps: SurvivalStep[];
  overall_survival: number;
  worst_step_index: number | null;
  worst_step_hazard: number | null;
}

export interface Econometrics {
  ok: boolean;
  elasticity?: number | null;
  elasticity_r2?: number | null;
  elasticity_se?: number | null;
  elasticity_ci?: [number, number] | null;
  demand_fit_r2?: number | null;
  continuous_optimal_price?: number | null;
  classification?: string;
  reason?: string;
}

/** Every `kind` the server actually persists.
 *
 *  This union said `"swarm" | "compare" | "walk"` while `storage.insert_swarm_run`
 *  was being called with ten different values, so any code that switched on it
 *  was exhaustive over a third of reality and the compiler agreed. `result` is
 *  still typed as `SwarmAggregate`, which is only structurally true for `swarm`
 *  — the other nine are different shapes entirely. Narrow it with a check on
 *  `kind` before reaching into `result`, and see the type-contract note in
 *  docs/AUDIT-BACKLOG.md: the real fix is generating this file from the OpenAPI
 *  schema, which closes the whole class rather than this one instance. */
export type RunKind =
  | "swarm"
  | "compare"
  | "walk"
  | "content"
  | "page"
  | "price"
  | "objection"
  | "virality"
  | "optimize"
  | "sequence"
  | "room";

export interface SwarmRunRow {
  id: string;
  created_at: string;
  kind: RunKind;
  request: { scenario?: string; cognitive_load: string };
  result: SwarmAggregate;
  actuals: { engagement?: number; intent?: number } | null;
}

export interface VariantRank {
  name: string;
  mean_intent: number;
  mean_engagement: number;
  lift_vs_control_pct: number | null;
}

export interface CompareResult {
  findings?: FindingsBlock | null;
  run_id: string;
  cognitive_load: string;
  twin_count_per_variant: number;
  winner: string;
  ranking: VariantRank[];
  variants: Record<string, SwarmAggregate>;
  bayesian?: BayesianResult | null;
}

export interface WalkStepStat {
  step_index: number;
  step: string;
  entered: number;
  abandoned: number;
  converted: number;
  /** null when no twin reached this step. Distinct from 0, which is the worst
   *  possible score — an unreached step is not a hated one. */
  mean_engagement: number | null;
}

export interface WalkAggregate {
  findings?: FindingsBlock | null;
  run_id: string;
  twin_count: number;
  cognitive_load: string;
  context_window_steps: number;
  completion_rate: number;
  conversion_rate: number;
  steps: WalkStepStat[];
  walks: unknown[];
  survival?: SurvivalResult | null;
}

/** MAE and bias alone cannot separate a compressed predictor from a noisy one
 *  — slope and r are what tell them apart, so they travel together. */
export interface MetricCalibration {
  n: number;
  mae: number;
  bias: number;
  rmse?: number | null;
  r?: number | null;
  slope?: number | null;
  intercept?: number | null;
  slope_r2?: number | null;
  mae_ci?: [number, number] | null;
  bias_ci?: [number, number] | null;
  /** The verdict is GATED on this interval — "compressed" is asserted only when
   *  the whole of it sits below 1 — so rendering the adjective without it shows
   *  a conclusion while hiding what licenses it. Null below n = 8. */
  slope_ci?: [number, number] | null;
  /** The slopes consistent with the data under ANY assumption about which axis
   *  is noisier. The verdict gates on THIS, not on slope_ci — a point estimate
   *  needs an error-ratio assumption nobody can check, and picking one wrongly
   *  produced a false "exaggerated" verdict in 44.5% of honest runs at n = 40. */
  slope_bracket_ci?: [number, number] | null;
  r_ci?: [number, number] | null;
  /** Isotonic (PAV) reliability diagram: where along the range the predictor
   *  drifts, which one slope cannot show. Null below n = 8. */
  reliability_curve?: [number, number][] | null;
  brier_score?: number | null;
  /** CORP decomposition. Separates "wrong" from "uninformative": a predictor
   *  can be perfectly calibrated and still useless. Null below n = 8. */
  brier_decomposition?: {
    miscalibration: number;
    discrimination: number;
    uncertainty: number;
  } | null;
  verdict?: string | null;
}

export interface CalibrationReport {
  runs_with_actuals: number;
  engagement: MetricCalibration | null;
  intent: MetricCalibration | null;
}

export interface JobRow {
  id: string;
  created_at: string;
  updated_at: string;
  kind: string;
  status: "queued" | "running" | "done" | "error";
  run_id: string | null;
  error: string | null;
}

export interface PanelMember {
  id: string;
  label: string;
  created_at: string;
  consented_at: string | null;
  revoked_at: string | null;
  session_count: number;
  disclosure_url: string;
}

export interface PricePoint {
  price: number;
  demand: number;
  revenue_index: number;
}

export interface PriceSensitivityResult {
  findings?: FindingsBlock | null;
  run_id: string;
  twin_count: number;
  cognitive_load: string;
  product_preview: string;
  points: PricePoint[];
  recommended_price: number;
  econometrics?: Econometrics | null;
}

export interface ObjectionTheme {
  objection: string;
  count: number;
  dealbreaker_count: number;
  variants?: string[];
}

export interface ObjectionResult {
  findings?: FindingsBlock | null;
  run_id: string;
  twin_count: number;
  cognitive_load: string;
  pitch_preview: string;
  sentiment_distribution: Record<string, number>;
  dealbreaker_rate: number;
  mean_recommend: number;
  top_objections: ObjectionTheme[];
  clustering_method?: string;
  recommend_ci?: [number, number] | null;
  /** kernel-PCA projection of theme centroids — x/y in [-1,1], semantic layout */
  theme_map?: ThemeMapPoint[] | null;
}

export interface ThemeMapPoint {
  label: string;
  x: number;
  y: number;
  count: number;
  dealbreaker: boolean;
}

// ── virality forecaster (Galton-Watson branching process) ─────────────────

/** Per-generation quantiles across the 2,000 seeded cascade simulations. */
export interface GenerationBand {
  g: number;
  active_q10: number;
  active_q50: number;
  active_q90: number;
  cum_q10: number;
  cum_q50: number;
  cum_q90: number;
  alive_frac: number;
}

/** cascade_tail_test() over final cascade sizes — a DISCRETE power-law fit.
 *
 *  Cascade sizes are counts with a large atom at 1, and a continuous MLE on
 *  them is misspecified: it read α = 1.58 where Galton-Watson theory gives
 *  exactly 3/2, using less of the sample to do it. The discrete MLE (Hurwitz
 *  zeta normaliser) measures 1.4978 ± 0.0043 on the same process. */
export interface ViralityCriticality {
  alpha?: number;
  /** SE of the tail index from the observed Fisher information. */
  alpha_se?: number;
  /** estimated by KS minimisation, not assumed to be min(x) */
  xmin?: number;
  ks_distance?: number;
  loglik_ratio_per_sample?: number;
  /** Vuong statistic and p-value for power-law vs geometric. `vuong_p_value` is
   *  the MODEL COMPARISON — which of two families fits better — and answers a
   *  different question from `p_value` below. */
  vuong_z?: number;
  vuong_p_value?: number;
  /** Goodness-of-fit p from a semi-parametric bootstrap (Clauset §4): does the
   *  fitted power law actually describe this tail at all? A comparison can
   *  prefer a power law over a geometric while BOTH are wrong, so this is the
   *  headline p — the downstream claim is "power-law tail", and this is the
   *  test of that claim. Low p = the power law is rejected. */
  p_value?: number;
  /** Whether the fitted power law survived that test. When false, `alpha` is a
   *  fitted parameter of a REJECTED model and must not be presented as the
   *  measured tail index. */
  power_law_plausible?: boolean;
  /** Synthetic datasets behind `p_value`. */
  gof_sims?: number;
  discrete?: boolean;
  method?: string;
  /**
   * "levy_like" | "brownian_like" | "heavy_tailed" | "not_power_law" |
   * "explosive" | "unresolved".
   *
   * `heavy_tailed` and `not_power_law` both mean the pure power law was
   * rejected by the goodness-of-fit test; they differ in what Vuong then says.
   * `heavy_tailed` — rejected, but decisively heavier than geometric, which is
   * the measured verdict for real critical Galton-Watson data (its tail is a
   * power law with a cutoff, so neither pure model fits). `not_power_law` —
   * rejected and not detectably heavier than exponential.
   *
   * Neither `explosive` nor `unresolved` carries a tail fit, because in both
   * cases a material share of simulated cascades never terminated and their
   * sizes are lower bounds rather than observations. They differ in *which*
   * bound was hit, and the distinction is the whole finding:
   *
   * · `explosive`  — cascades outgrew the 50,000-node size cap. Unbounded
   *   growth; size is limited by the audience, not the content.
   * · `unresolved` — cascades were still spreading at the generation horizon.
   *   Near-critical slow burn, which is a completely different story and used
   *   to be reported as `explosive` because the two censoring causes were
   *   summed. Verified at R₀ = 1.2: 36.7% censored, of which *zero* had
   *   reached the cap.
   *
   * `alpha` is absent for both; render `interpretation` instead.
   */
  regime?: string;
  interpretation?: string;
  n?: number;
  n_total?: number;
  /** fraction censored for any reason — capped_frac + horizon_frac */
  censored_frac?: number;
  /** fraction that outgrew the size cap (the "unbounded growth" claim) */
  capped_frac?: number;
  /** fraction still spreading at the generation horizon */
  horizon_frac?: number;
  n_uncensored?: number;
}

/** Cascade-size quantiles straight from the simulation — the only size figures
 *  that survive the supercritical regime, where 1/(1−R₀) diverges. */
export interface SimulatedSize {
  median: number | null;
  p90: number | null;
  max: number | null;
  /** censored for any reason; do NOT render this as "hit the bound" */
  censored_frac: number;
  /** outgrew the size cap — this is the one that means "hit the bound" */
  capped_frac?: number;
  /** still spreading at the generation horizon */
  horizon_frac?: number;
  note?: string;
}

export interface ViralityResult {
  findings?: FindingsBlock | null;
  run_id: string;
  twin_count: number;
  cognitive_load: string;
  content_preview: string;
  fanout: number;
  share_mean: number;
  share_ci: [number, number] | null;
  r0: number;
  r0_ci: [number, number] | null;
  critical_fanout: number | null;
  supercritical: boolean;
  extinction_q_mixing: number;
  extinction_q_hetero: number;
  extinction_q_ci: [number, number] | null;
  takeoff_probability: number;
  expected_cascade_size: number | null;
  cascade_size_ci: [number, number] | null;
  generations: GenerationBand[];
  final_size_bins: [number, number, number][];
  criticality: ViralityCriticality | null;
  /** true when cascade_size_ci was bootstrapped conditional on subcriticality */
  cascade_size_ci_conditional_on_subcritical?: boolean;
  simulated_size?: SimulatedSize;
  share_intents: number[];
  n_sims: number;
  seed: number;
  method: string;
  disclaimer: string;
}

export interface ViralityRequestBody {
  content: string;
  fanout: number;
  session_ids: string[];
  twins_per_persona: number;
  cognitive_load: CognitiveLoad;
}

/* ── the white room (deliberation study) ───────────────────────────────────
   A cast of characters sits at a table and argues a motion over several
   rounds. Each member says something PUBLICLY with a position on the motion,
   and holds a PRIVATE position the others never see. The gap between those two
   numbers is preference falsification, and it is the measurement this study
   exists for.

   Every analysis field below is optional and nullable ON PURPOSE. The backend
   refuses to compute what the data cannot support and sends a reason instead —
   an influence matrix fitted from three observations per member is not a
   quieter version of a good one, it is a different claim — so the UI must be
   able to render the refusal as a finding rather than as an empty chart. */

export interface RoomMember {
  name: string;
  role: string;
  archetype: string;
  /** what they personally stand to gain or lose from the motion */
  stake: string;
  /** The second-person conditioning prompt the member is simulated from. The
   *  server requires it on every member of a supplied cast, so the UI carries
   *  it through untouched rather than editing or dropping it. */
  system_prompt: string;
  /** 0‥1 dispositions. DECLARED by the casting model, never measured — the
   *  fitted susceptibility in the results is free to contradict them, and the
   *  contradiction is one of the study's better findings. */
  openness: number;
  assertiveness: number;
  seniority: number;
  risk_appetite: number;
  expertise: string[];
}

export interface RoomTurn {
  replicate: number;
  /** true when this turn came from the control arm — the member was shown
   *  statements from a DIFFERENT room, so any convergence here is the language
   *  model being agreeable rather than the deliberation doing work. */
  placebo: boolean;
  round: number;
  member: string;
  speaking_index: number;
  public_statement: string;
  /** 0 = against the motion, 1 = for it */
  public_position: number;
  private_position: number;
  private_note: string;
  conceded_to: string[];
}

/** Fitted linear influence: row i's next position as a weighted average of the
 *  positions i heard. `refused` non-null means the fit is NOT identifiable and
 *  W must not be drawn — read the reason and say so instead. */
export interface RoomInfluence {
  W: number[][];
  susceptibility: number[];
  names: string[];
  r2_per_member: (number | null)[];
  n_obs_per_member: number;
  refused: string | null;
  method: string;
  /** Rows whose regressors were collinear: their susceptibility is identified
   *  but WHO moved them is not, at any round count. Drawn dashed. */
  unidentified_rows?: { name: string; index: number; reason: string }[] | null;
  /**
   * FALSE means the individual entries of W were measured to be worse than a
   * uniform-row guess at this observation count. The row ORDERING, the
   * susceptibilities, the equilibrium and the leave-one-out deltas still
   * recover — so the network may be drawn as a pattern, but no single weight
   * may be quoted as a number. The UI enforces that.
   */
  w_entries_supported?: boolean;
  w_entry_caveat?: string | null;
  /** per member: is their fitted susceptibility distinguishable from immovable */
  susceptibility_supported?: boolean[] | null;
}

/**
 * Perron weights of W — whose opinion the consensus actually weights.
 *
 * `centrality` is NULL when the influence graph is not strongly connected: W
 * then has more than one stationary distribution and no unique centrality
 * exists. That is a room which has SPLIT, not a failed computation, and `note`
 * says so — which is why the note is rendered rather than dropped.
 */
export interface RoomCentrality {
  centrality: number[] | null;
  spectral_gap: number | null;
  converges: boolean;
  consensus_forecast: number | null;
  note?: string | null;
}

export interface RoomCounterfactual {
  name: string;
  /** the room's fixed point with this member removed, or null when it has none */
  equilibrium: number[] | null;
  room_delta: number;
  /** where the remaining room lands without them, when the server solved it */
  landing?: number | null;
  /** the same room's mean WITH them — the anchor `room_delta` is measured from,
   *  taken over the surviving members only, so it is not the full-room mean */
  anchor?: number | null;
  note?: string | null;
}

/**
 * The gap, per member and for the room.
 *
 * SIGN: gap = public − private, the server's convention. Positive means they
 * said the motion was better than they privately think it is — public support
 * that does not exist. Negative means the reverse: quiet agreement nobody voiced.
 */
export interface RoomFalsification {
  per_member: { name: string; gap: number; public: number; private: number }[];
  room_gap: number;
  room_gap_ci: [number, number] | null;
  p_value: number | null;
  by_round: { round: number; gap: number }[];
  /** did the room gap clear its own evidential bar */
  supported?: boolean;
  direction?: string | null;
  /** why the interval is what it is — e.g. n too small for BCa */
  gap_ci_note?: string | null;
  method?: string | null;
  refused?: string | null;
}

/** Did the room converge on information, or on each other? `bayes_rate` is the
 *  variance collapse a rational updater would show given the same signals.
 *  Both are per-round log-contraction rates: HIGHER means faster collapse. */
export interface RoomConformity {
  observed_rate: number;
  bayes_rate: number;
  ratio: number;
  verdict: string;
  /** the measured cross-member spread per round, straight from the estimator */
  sd_by_round?: (number | null)[] | null;
  refused?: string | null;
}

export interface RoomCascades {
  per_member: { name: string; score: number }[];
}

export interface RoomCoalitions {
  /** camps in the FINAL PUBLIC positions */
  camps: { members: string[]; mean_position: number }[];
  /** camps in the final PRIVATE positions — the room as it actually is */
  private_camps?: { members: string[]; mean_position: number }[] | null;
  /** true when the public camps and the private camps are not the same room */
  hidden_split?: boolean | null;
  power?: { name: string; shapley_shubik: number; banzhaf: number }[] | null;
  power_note?: string | null;
  /** where the voting weights came from. When it says "equal weights", every
   *  power index is 1/n by construction and says nothing about this room. */
  weight_source?: string | null;
}

/** How much the seeded speaking order moved the outcome. A large effect here
 *  means the room's verdict is partly an artefact of who spoke first. */
export interface RoomOrderEffect {
  effect_size: number;
  p_value: number | null;
}

/**
 * THE CONTROL. If the placebo arm converged as hard as the real one, the
 * deliberation measured nothing and the screen has to say so.
 *
 * DIRECTION, because it is the opposite of the intuitive one: a convergence
 * figure here is `sd_final / sd_opening` — the spread REMAINING. Lower means the
 * room converged harder, and 1.0 means it never converged at all. The damning
 * case is therefore `placebo_convergence <= real_convergence`.
 */
export interface RoomPlacebo {
  real_convergence: number | null;
  placebo_convergence: number | null;
  real_gap: number | null;
  placebo_gap: number | null;
  verdict: string;
  /** the server's own categorical read: "none — …" / "marginal — …" / "clear — …" */
  separation?: string | null;
  /** placebo − real on the spread ratio; ≤ 0 means the control matched or beat it */
  convergence_delta?: number | null;
  /** real − placebo on the public/private gap */
  gap_delta?: number | null;
  gap_note?: string | null;
  ran?: boolean;
  refused?: string | null;
}

/**
 * What the ROOM SCREEN renders.
 *
 * This is a view model, not the wire format. The endpoint returns the
 * orchestration envelope with every estimator nested under `analysis`, plus a
 * handful of keys under different names (`absence` for the counterfactuals,
 * `cascade` for the cascade scores, `motion_preview` for the motion). One
 * adapter — components/room/adapt.tsx — maps that onto this, so the drift lives
 * in a single documented place instead of in nine components.
 */
export interface RoomResult {
  cast: RoomMember[];
  motion: string;
  method: string;
  limits: string;
  /** how the cast came to exist: approved / described / inferred, with a note */
  provenance?: { provenance: string; members: number; note?: string | null } | null;
  /** the pre-flight check on the cast itself: a room that agrees by
   *  construction will converge whatever the deliberation does */
  diversity?: {
    seats?: number;
    distinct_archetypes?: number;
    distinct_roles?: number;
    mean_archetype_overlap?: number;
    degenerate?: boolean;
    note?: string | null;
  } | null;
  /** did the instrument move at all — checked before any estimate is read */
  instrument_check?: Record<string, unknown> | null;
  /** every estimator that declined, with its reason */
  refusals?: { what: string; reason: string }[] | null;
  /** replicates discarded whole for a missing turn, and why */
  dropped_replicates?: { what: string; reason: string }[] | null;
  turns_failed?: number | null;
  /** observed final mean/spread and the forecast, straight from the estimator */
  consensus?: Record<string, number | null> | null;
  findings?: FindingsBlock | null;
  influence?: RoomInfluence | null;
  centrality?: RoomCentrality | null;
  equilibrium?: number[] | null;
  counterfactuals?: RoomCounterfactual[] | null;
  falsification?: RoomFalsification | null;
  conformity?: RoomConformity | null;
  cascades?: RoomCascades | null;
  coalitions?: RoomCoalitions | null;
  order_effect?: RoomOrderEffect | null;
  placebo?: RoomPlacebo | null;
  /** every turn of every replicate, INCLUDING the private positions — which
   *  are deliberately absent from the stream and arrive only here */
  turns: RoomTurn[];
  run_id?: string;
  rounds?: number;
  replicates?: number;
}

/** POST /room/cast — prose brief in, a cast the researcher can edit out. */
export interface RoomCastResponse {
  cast: RoomMember[];
  note?: string | null;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`/api/cs${path}`, {
      ...init,
      headers: { "Content-Type": "application/json", ...init?.headers },
    });
  } catch (e) {
    // The proxy itself is unreachable (dev server down, network gone). Report
    // it before rethrowing so the chrome can show a degraded state rather than
    // letting every caller's .catch() render this as "no data".
    reportBackendResult(null, e instanceof Error ? e.message : "Network error");
    throw e;
  }
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    const detail = body?.detail ?? `${res.status} ${res.statusText}`;
    reportBackendResult(res.status, detail);
    throw new Error(detail);
  }
  reportBackendResult(res.status);
  return res.json();
}

export type CognitiveLoad = "low" | "medium" | "high";

export const api = {
  sessions: () => request<SessionRow[]>("/sessions"),
  profile: (id: string) => request<unknown>(`/sessions/${id}/profile`, { method: "POST" }),
  runSwarm: (body: {
    scenario: string;
    twins_per_persona: number;
    cognitive_load: CognitiveLoad;
  }) => request<SwarmAggregate>("/swarm/run", { method: "POST", body: JSON.stringify(body) }),
  compare: (body: {
    variants: { name: string; scenario: string }[];
    twins_per_persona: number;
    cognitive_load: CognitiveLoad;
  }) => request<CompareResult>("/swarm/compare", { method: "POST", body: JSON.stringify(body) }),
  walk: (body: {
    steps: string[];
    twins_per_persona: number;
    cognitive_load: CognitiveLoad;
  }) => request<WalkAggregate>("/swarm/walk", { method: "POST", body: JSON.stringify(body) }),
  swarmRuns: (kind?: string) =>
    request<SwarmRunRow[]>(`/swarm/runs${kind ? `?kind=${kind}` : ""}`),
  recordActuals: (runId: string, body: { engagement?: number; intent?: number }) =>
    request<unknown>(`/runs/${runId}/actuals`, { method: "POST", body: JSON.stringify(body) }),
  validationReport: () => request<CalibrationReport>("/validation/report"),

  // Phase 3 — jobs
  createJob: (kind: string, payload: unknown) =>
    request<{ job_id: string; status: string }>("/jobs", {
      method: "POST",
      body: JSON.stringify({ kind, payload }),
    }),
  jobs: () => request<JobRow[]>("/jobs"),

  // Phase 4 — panel
  panelMembers: () => request<PanelMember[]>("/panel/members"),
  createPanelMember: (label: string) =>
    request<PanelMember & { token: string }>("/panel/members", {
      method: "POST",
      body: JSON.stringify({ label }),
    }),
  revokePanelMember: (token: string) =>
    request<{ deleted_sessions: number }>(`/panel/${token}/revoke`, { method: "POST" }),

  // Studies — advanced use cases
  priceStudy: (body: {
    product: string;
    prices: number[];
    twins_per_persona: number;
    cognitive_load: CognitiveLoad;
  }) => request<PriceSensitivityResult>("/studies/price", { method: "POST", body: JSON.stringify(body) }),
  objectionStudy: (body: {
    pitch: string;
    twins_per_persona: number;
    cognitive_load: CognitiveLoad;
  }) => request<ObjectionResult>("/studies/objection", { method: "POST", body: JSON.stringify(body) }),
};
