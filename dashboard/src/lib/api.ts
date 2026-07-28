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

export interface SwarmAggregate {
  run_id: string;
  scenario_preview: string;
  twin_count: number;
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

export interface SwarmRunRow {
  id: string;
  created_at: string;
  kind: "swarm" | "compare" | "walk";
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

/** Raw powerlaw_vs_exponential() dict over final cascade sizes. */
export interface ViralityCriticality {
  alpha?: number;
  /** closed-form SE of the tail index, (α−1)/√n */
  alpha_se?: number;
  /** estimated by KS minimisation, not assumed to be min(x) */
  xmin?: number;
  ks_distance?: number;
  loglik_ratio_per_sample?: number;
  /** Vuong statistic and p-value for power-law vs exponential */
  vuong_z?: number;
  p_value?: number;
  /**
   * "levy_like" | "brownian_like" | "indistinguishable" | "explosive" |
   * "unresolved".
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
