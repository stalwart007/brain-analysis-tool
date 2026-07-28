"use client";

import { useEffect, useState } from "react";

import {
  type BackendHealth,
  getBackendHealth,
  subscribeBackendHealth,
} from "@/lib/health";

/**
 * The vitals chip, reading the actual backend rather than asserting "online".
 *
 * The old markup was a lit dot and the literal string "online", rendered
 * unconditionally. With the API stopped it kept saying online while every read
 * 502'd — which is the opposite of what a vitals readout is for. An indicator
 * that cannot indicate anything is worse than no indicator, because it is
 * actively consulted.
 *
 * Three states, because "we haven't asked yet" is not "it's up": null while
 * the first request is in flight, then reachable / unreachable.
 */
function useBackendHealth(): BackendHealth {
  const [health, setHealth] = useState<BackendHealth>(getBackendHealth);
  useEffect(() => {
    setHealth(getBackendHealth());
    return subscribeBackendHealth(setHealth);
  }, []);
  return health;
}

/**
 * The banner that stops an outage reading as an empty database.
 *
 * Without it the zeros and the "No telemetry yet — open the demo site" copy
 * are indistinguishable from a fresh install, which is the specific failure
 * this fixes: every panel's empty state is written for "nothing has happened
 * yet", and none of them for "we couldn't ask".
 */
export function BackendBanner() {
  const health = useBackendHealth();
  if (health.ok !== false) return null;
  return (
    <div
      role="status"
      /* below the nav AND the depth rail — the rail's anatomy label is how you
         know where you are, and an error banner must not be the thing that
         hides it */
      className="pointer-events-none fixed left-1/2 top-28 z-30 -translate-x-1/2 border border-critical/40 bg-critical/[0.09] px-4 py-2 backdrop-blur-md"
    >
      <div className="hud-label text-critical">Backend unreachable</div>
      <p className="mt-0.5 max-w-md text-[11px] leading-relaxed text-ink-2">
        {health.detail}
        {health.lastOkAt
          ? " — showing the last data that loaded."
          : " — nothing has loaded, so any zeros below are unknown rather than empty."}
      </p>
    </div>
  );
}

export default function BackendPulse() {
  const health = useBackendHealth();

  const down = health.ok === false;
  const pending = health.ok === null;

  return (
    <span
      className="flex items-center gap-1.5"
      title={down ? (health.detail ?? "Backend unreachable") : undefined}
    >
      <span
        className={
          down
            ? "inline-block h-1.5 w-1.5 rounded-full bg-critical"
            : pending
              ? "inline-block h-1.5 w-1.5 rounded-full bg-muted"
              : "neon-dot pulse-dot inline-block h-1.5 w-1.5 rounded-full bg-bone"
        }
      />
      <span className={`hud-label ${down ? "text-critical" : "text-ink-2"}`}>
        {down ? "no backend" : pending ? "connecting" : "online"}
      </span>
    </span>
  );
}
