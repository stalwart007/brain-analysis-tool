import { SwarmAggregate } from "./api";

/** One SSE frame from any simulation stream. Shapes by `type`:
 *  start        {total, variants?|prices?|steps?, adaptive?}
 *  agent        per-agent payload (varies per simulation)
 *  agent_failed {variant?}
 *  step         walk only: {twin, persona, step_index, action, engagement}
 *  twin_done    walk only: {twin, outcome}
 *  wave         adaptive compare: {index, allocation}
 *  posterior    compare: {bayesian}
 *  early_stop   adaptive compare: {spent, budget, saved, reason}
 *  clustering   objection: phase marker before the themed result
 *  done         {result}
 *  error        {detail}
 */
export interface StreamEvent {
  type: string;
  [key: string]: unknown;
}

/**
 * POST to an SSE endpoint and dispatch each `data:` frame to `onEvent` as it
 * arrives — this is what lets every visualization spawn agents live. Resolves
 * when the stream closes.
 */
export async function streamRun(
  path: string,
  body: unknown,
  onEvent: (evt: StreamEvent) => void
): Promise<void> {
  const res = await fetch(`/api/cs${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok || !res.body) {
    const detail = await res
      .json()
      .then((d) => d?.detail)
      .catch(() => null);
    onEvent({ type: "error", detail: detail ?? `Stream failed (${res.status})` });
    return;
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  // A stream is only successful if it SAYS so. Without this the loop exits on
  // a clean close and `streamRun` resolves normally, so a backend that died
  // mid-run — killed process, proxy read timeout, or an exception raised after
  // the headers were already sent — left every panel with result === null AND
  // error === null. `finally { setBusy(false) }` then closed the stage on a
  // run the user had just watched 40 agents land in, with nothing said.
  let terminal = false;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let idx: number;
    while ((idx = buffer.indexOf("\n\n")) >= 0) {
      const frame = buffer.slice(0, idx);
      buffer = buffer.slice(idx + 2);
      const line = frame.split("\n").find((l) => l.startsWith("data:"));
      if (!line) continue;
      let evt: StreamEvent;
      try {
        evt = JSON.parse(line.slice(5).trim()) as StreamEvent;
      } catch {
        continue; /* genuinely malformed wire frame */
      }
      // Dispatch OUTSIDE the parse guard. It used to sit inside, so any throw
      // in a panel's own handler was swallowed as "malformed frame" — e.g.
      // CognitionPipeline's `ll_trace[0].toFixed(1)` on an empty array, which
      // left the HMM stage pill stuck on "running" forever with a clean
      // console and no way to tell the difference from a slow model.
      if (evt.type === "done" || evt.type === "error") terminal = true;
      onEvent(evt);
    }
  }
  if (!terminal) {
    onEvent({
      type: "error",
      detail:
        "The stream ended before the run completed — the backend may have " +
        "restarted or timed out. Any partial results below are incomplete.",
    });
  }
}

// ── typed convenience wrappers ────────────────────────────────────────────

export type SwarmStreamEvent =
  | { type: "start"; total: number }
  | { type: "agent"; action: string; engagement: number; intent: number }
  | { type: "agent_failed" }
  | { type: "done"; result: SwarmAggregate }
  | { type: "error"; detail: string };

export async function streamSwarm(
  body: { scenario: string; twins_per_persona: number; cognitive_load: string },
  onEvent: (evt: SwarmStreamEvent) => void
): Promise<void> {
  await streamRun("/swarm/stream", body, (e) => onEvent(e as unknown as SwarmStreamEvent));
}
