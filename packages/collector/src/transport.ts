import type { FeaturePayload } from "./types";

export interface IngestEnvelope {
  site_id: string;
  consent: true;
  sdk_version: string;
  page_path: string;
  features: FeaturePayload;
  panel_token?: string;
}

export const SDK_VERSION = "0.1.0";

/**
 * Fire-and-forget delivery. sendBeacon survives pagehide and never blocks the
 * main thread; fetch(keepalive) is the fallback for older browsers / large payloads.
 */
export function send(endpoint: string, envelope: IngestEnvelope): void {
  const body = JSON.stringify(envelope);
  if (typeof navigator.sendBeacon === "function") {
    const ok = navigator.sendBeacon(
      endpoint,
      new Blob([body], { type: "application/json" })
    );
    if (ok) return;
  }
  void fetch(endpoint, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body,
    keepalive: true,
  }).catch(() => {
    /* telemetry is best-effort; never surface errors to the host page */
  });
}
