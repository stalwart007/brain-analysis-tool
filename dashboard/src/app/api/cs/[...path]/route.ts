/**
 * Backend forwarder. Replaces a blind rewrite so that:
 *  1. middleware's session gate applies (it already covers /api/*), and
 *  2. the headless API key (COGNISWARM_API_KEY) is injected server-side —
 *     the browser never sees it.
 */

import { NextRequest, NextResponse } from "next/server";

// SSE streaming (POST /api/cs/swarm/stream) must not be buffered or statically
// optimized — force the Node runtime and dynamic rendering.
export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const BACKEND = process.env.COGNISWARM_BACKEND ?? "http://127.0.0.1:8000";

async function forward(
  request: NextRequest,
  params: Promise<{ path: string[] }>
): Promise<NextResponse> {
  const { path } = await params;
  const url = `${BACKEND}/v1/${path.join("/")}${request.nextUrl.search}`;

  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (process.env.COGNISWARM_API_KEY) {
    headers["X-API-Key"] = process.env.COGNISWARM_API_KEY;
  }

  let response: Response;
  try {
    response = await fetch(url, {
      method: request.method,
      headers,
      body: request.method === "GET" || request.method === "HEAD" ? undefined : await request.text(),
      cache: "no-store",
    });
  } catch {
    return NextResponse.json(
      { detail: "Backend unreachable — is the FastAPI server running?" },
      { status: 502 }
    );
  }

  const contentType = response.headers.get("Content-Type") ?? "application/json";
  const outHeaders: Record<string, string> = { "Content-Type": contentType };
  if (contentType.includes("text/event-stream")) {
    outHeaders["Cache-Control"] = "no-cache, no-transform";
    outHeaders["Connection"] = "keep-alive";
  }
  return new NextResponse(response.body, { status: response.status, headers: outHeaders });
}

export async function GET(req: NextRequest, ctx: { params: Promise<{ path: string[] }> }) {
  return forward(req, ctx.params);
}
export async function POST(req: NextRequest, ctx: { params: Promise<{ path: string[] }> }) {
  return forward(req, ctx.params);
}
