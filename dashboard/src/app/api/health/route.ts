/**
 * Liveness for the platform's health check.
 *
 * WHY THIS EXISTS. The dashboard had no `[[http_service.checks]]` block, so
 * Fly had no way to tell a serving process from a wedged one — `fly status`
 * showed an empty CHECKS column, and traffic would have kept arriving at a
 * hung Node process indefinitely. The backend has had a check on `/healthz`
 * since it was deployed; the public surface, which is the one users actually
 * hit, did not.
 *
 * It answers LIVENESS, not readiness, and the difference is deliberate. It
 * does not call the backend. A readiness check that fails when a dependency is
 * down takes the dashboard out of rotation for the very request that would
 * have rendered the error page explaining the outage — so a backend problem
 * would present as "site unreachable" rather than as a backend problem. The
 * only question here is whether this process can still serve HTTP.
 *
 * Public by necessity: the check arrives with no session. Added to
 * `PUBLIC_PATHS` in the middleware, and it discloses nothing — a fixed string
 * and the boot time, no version, no config, no dependency state.
 */

import { NextResponse } from "next/server";

/** Set once at module load, so the value is process start rather than now. */
const BOOTED_AT = Date.now();

// Never prerendered or cached: a cached 200 from build time would report a
// dead process as healthy, which is worse than having no check at all.
export const dynamic = "force-dynamic";
export const revalidate = 0;

export function GET() {
  return NextResponse.json(
    { status: "ok", uptime_s: Math.round((Date.now() - BOOTED_AT) / 1000) },
    { headers: { "Cache-Control": "no-store" } }
  );
}
