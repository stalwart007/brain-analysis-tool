/**
 * Is the backend reachable? One question, asked cheaply.
 *
 * WHY THIS EXISTS. The vitals chip has three states — connecting, online, no
 * backend — and it only ever LISTENED. `reportBackendResult` fires as a side
 * effect of real API traffic, so the chip resolved only on pages that happened
 * to fetch something. On `/room`, which loads no data until you cast a room, it
 * sat at "connecting" indefinitely. A permanently-in-flight indicator reads as
 * a broken connection, which is precisely the misreading the component's own
 * docstring warns about: an indicator that cannot indicate anything is worse
 * than no indicator, because it is actively consulted.
 *
 * NOT ROUTED THROUGH `/api/cs`. That forwarder maps `/api/cs/<p>` to
 * `<BACKEND>/v1/<p>`, and the health endpoint is at the ROOT — `/healthz`, not
 * `/v1/healthz` — so it is unreachable through the proxy without loosening the
 * mapping for every path. Probing a real data endpoint instead would work and
 * would mean pulling a list of sessions on every page load to answer a yes/no.
 *
 * AUTHENTICATED, deliberately. It is not in `PUBLIC_PATHS`: it reports the
 * state of private infrastructure, and only signed-in users ever see the chip.
 * Distinct from `/api/health`, which is the platform's liveness probe for THIS
 * process and says nothing about the backend — conflating the two would take
 * the dashboard out of rotation whenever the API had a bad minute.
 */

import { NextResponse } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const BACKEND = process.env.COGNISWARM_BACKEND ?? "http://127.0.0.1:8000";

export async function GET() {
  try {
    const response = await fetch(`${BACKEND}/healthz`, {
      cache: "no-store",
      // Bounded so a hung backend leaves the chip at "connecting" rather than
      // holding a Node request open until the platform kills it.
      signal: AbortSignal.timeout(4000),
    });
    if (!response.ok) {
      return NextResponse.json(
        { detail: `Backend answered ${response.status}` },
        { status: 502 }
      );
    }
    return NextResponse.json({ ok: true }, { headers: { "Cache-Control": "no-store" } });
  } catch {
    return NextResponse.json(
      { detail: "Backend unreachable — is the FastAPI server running?" },
      { status: 502 }
    );
  }
}
