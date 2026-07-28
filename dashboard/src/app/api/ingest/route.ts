/**
 * PUBLIC telemetry passthrough — the SDK's beacon target.
 *
 * Ingest and analysis have opposite exposure requirements, and this route is
 * what lets them coexist. End users' browsers must be able to POST telemetry,
 * so *something* has to be public; but the analysis surface behind it — every
 * session, run history carrying every twin's inner monologue, and every
 * endpoint that spends OpenAI credit — must not be. Forwarding only this one
 * path from the public app lets the backend sit entirely on a private network
 * with no public address at all.
 *
 * That containment is doing real work today: the analysis surface still has an
 * unauthenticated persona-eviction path (200 ingests evict every profiled
 * session from the load window) and no tenant isolation whatsoever. Neither is
 * fixed here. Keeping the backend private means neither is reachable from the
 * internet while they are being fixed properly.
 *
 * Three things this route deliberately does NOT do:
 *   · forward any other path — it is a single fixed URL, not a proxy
 *   · inject COGNISWARM_API_KEY — ingest is unauthenticated by design (consent
 *     is its gate), and attaching the analysis key to a public route would
 *     hand the whole API to anyone who can reach this endpoint
 *   · pass through arbitrary headers — only the content type is forwarded
 */

import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const BACKEND = process.env.COGNISWARM_BACKEND ?? "http://127.0.0.1:8000";

/** Beacon payloads are small feature vectors; anything larger is not ours.
 *  Bounding it here keeps an unauthenticated public route from being a memory
 *  amplifier, since the backend's own payload limits are still open. */
const MAX_BODY_BYTES = 64 * 1024;

export async function POST(request: NextRequest) {
  const body = await request.text();
  if (body.length > MAX_BODY_BYTES) {
    return NextResponse.json({ detail: "Payload too large." }, { status: 413 });
  }

  try {
    const response = await fetch(`${BACKEND}/v1/ingest`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body,
      cache: "no-store",
    });
    // Mirror the status so the SDK's fetch fallback can tell success from
    // rejection, but never forward the body: backend error details are for
    // operators, not for every browser on the internet.
    return new NextResponse(null, { status: response.status });
  } catch {
    return NextResponse.json({ detail: "Ingest unavailable." }, { status: 502 });
  }
}
