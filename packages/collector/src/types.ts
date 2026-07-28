export type RawEventType = "scroll" | "move" | "click" | "key" | "focus";

export interface RawEvent {
  /** performance.now() timestamp, ms */
  t: number;
  type: RawEventType;
  x: number;
  y: number;
  /** semantic zone from the client's data-cs whitelist, if any */
  zone?: string;
}

/** Layer-1 output: the dense Heuristic feature payload. Math only — no PII, no DOM text. */
export interface FeaturePayload {
  segment_start: number;
  segment_end: number;
  event_count: number;
  scroll_velocity_mean: number;
  scroll_velocity_variance: number;
  scroll_direction_changes: number;
  micro_hesitation_count: number;
  hesitation_total_ms: number;
  rage_click_bursts: number;
  click_count: number;
  backtrack_ratio: number;
  zone_dwell_ms: Record<string, number>;
  zone_click_counts: Record<string, number>;
  /** [[t_ms, symbol_id]] — symbolic event stream for cognitive modeling.
   *  Alphabet: 0 scroll, 1 dwell, 2 click, 3 hesitate, 4 rage, 5 backtrack, 6 zone. */
  event_stream?: number[][];
  /** [[t_ms, velocity]] — scroll kinematics for spectral / entropy analysis. */
  velocity_series?: number[][];
  /** Quiet-period → click intervals (ms) — decision-latency proxies for DDM. */
  decision_latencies_ms?: number[];
}

export interface CogniSwarmOptions {
  siteId: string;
  /** ingestion endpoint, e.g. https://api.example.com/v1/ingest */
  endpoint: string;
  /**
   * Consent gate. `true` only when the user has affirmatively consented to
   * behavioral analytics on this site. May be a function re-checked at flush time.
   */
  consent: boolean | (() => boolean);
  /** Whitelist of semantic zones (data-cs attribute values) to attribute events to. */
  zones?: string[];
  /** How often to flush a feature segment. Default 10_000 ms. */
  flushIntervalMs?: number;
  /**
   * Research-panel enrollment token. Only for disclosed, opted-in panel
   * members — the server rejects tokens without an active consent record.
   */
  panelToken?: string;
}

export interface CogniSwarmHandle {
  /** Force-flush the current segment. */
  flush(): void;
  /** Stop all collection and release listeners/worker. */
  shutdown(): void;
  /** Update consent; revoking stops collection immediately and drops the buffer. */
  updateConsent(granted: boolean): void;
}
