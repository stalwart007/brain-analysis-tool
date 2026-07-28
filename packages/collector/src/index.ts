import { startCapture, type CaptureController } from "./capture";
import { send, SDK_VERSION } from "./transport";
import type {
  CogniSwarmHandle,
  CogniSwarmOptions,
  FeaturePayload,
} from "./types";

const DEFAULT_FLUSH_MS = 10_000;
const DEFAULT_ZONES = ["hero", "pricing-table", "cta", "checkout", "content"];

/** Respect browser-level privacy signals regardless of what the page says. */
function privacySignalsOptOut(): boolean {
  const nav = navigator as Navigator & { globalPrivacyControl?: boolean };
  return nav.doNotTrack === "1" || nav.globalPrivacyControl === true;
}

/**
 * Initialize the CogniSwarm collector.
 *
 * Returns null (and collects nothing) when consent is absent or the browser
 * signals an opt-out (DNT / Global Privacy Control). Consent is re-checked at
 * every flush — a revocation mid-session stops the very next segment.
 */
export function init(opts: CogniSwarmOptions): CogniSwarmHandle | null {
  const hasConsent = () =>
    typeof opts.consent === "function" ? opts.consent() : opts.consent === true;

  if (!hasConsent() || privacySignalsOptOut()) return null;

  const zones = opts.zones ?? DEFAULT_ZONES;
  let capture: CaptureController | null = startCapture(zones);

  capture.worker.onFeatures((features: FeaturePayload) => {
    if (!hasConsent()) return; // consent revoked between capture and flush
    send(opts.endpoint, {
      site_id: opts.siteId,
      consent: true,
      sdk_version: SDK_VERSION,
      page_path: location.pathname,
      features,
      ...(opts.panelToken ? { panel_token: opts.panelToken } : {}),
    });
  });

  const flush = () => {
    if (!capture) return;
    capture.worker.post({ kind: hasConsent() ? "flush" : "drop" });
  };

  const interval = window.setInterval(
    flush,
    opts.flushIntervalMs ?? DEFAULT_FLUSH_MS
  );
  const onPageHide = () => flush();
  window.addEventListener("pagehide", onPageHide);
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "hidden") flush();
  });

  const shutdown = () => {
    window.clearInterval(interval);
    window.removeEventListener("pagehide", onPageHide);
    capture?.stop();
    capture = null;
  };

  return {
    flush,
    shutdown,
    updateConsent(granted: boolean) {
      if (!granted) {
        capture?.worker.post({ kind: "drop" });
        shutdown();
      }
    },
  };
}

export type { CogniSwarmHandle, CogniSwarmOptions, FeaturePayload };
