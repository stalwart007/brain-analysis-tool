/**
 * Backend reachability, tracked at the one place every read passes through.
 *
 * Why this exists: with the API stopped, `GET /api/cs/sessions` and
 * `/api/cs/swarm/runs` both return 502 and the proxy supplies a perfectly good
 * `{detail: "Backend unreachable — is the FastAPI server running?"}`. Every
 * caller then discarded it with `.catch(() => {})`, so the dashboard rendered
 * `0 / 0 / 0`, the copy "No telemetry yet — open the demo site, accept consent,
 * and interact", and a lit "online" chip. The interface confidently asserted an
 * empty database when what had actually happened was total backend failure —
 * and there was nothing in the console either.
 *
 * Instrumenting `request()` fixes all eight sites at once and keeps future
 * callers honest by default, which per-page error state would not.
 */

export type BackendHealth = {
  /** null until the first request resolves either way */
  ok: boolean | null;
  /** the proxy's `detail`, or the transport error, when ok === false */
  detail: string | null;
  /** ms epoch of the last successful response, for "last known state" copy */
  lastOkAt: number | null;
};

let state: BackendHealth = { ok: null, detail: null, lastOkAt: null };

const listeners = new Set<(h: BackendHealth) => void>();

function emit() {
  for (const fn of listeners) fn(state);
}

/** A 502/503 from the proxy means the backend is down; a 4xx is a real answer
 *  from a backend that is up, and must not flip the indicator. */
export function reportBackendResult(status: number | null, detail?: string) {
  const unreachable = status === null || status === 502 || status === 503;
  state = unreachable
    ? { ok: false, detail: detail ?? "Backend unreachable", lastOkAt: state.lastOkAt }
    : { ok: true, detail: null, lastOkAt: Date.now() };
  emit();
}

export function getBackendHealth(): BackendHealth {
  return state;
}

export function subscribeBackendHealth(fn: (h: BackendHealth) => void): () => void {
  listeners.add(fn);
  return () => {
    listeners.delete(fn);
  };
}
