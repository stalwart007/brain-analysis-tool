import { NextRequest, NextResponse } from "next/server";
import { jwtVerify } from "jose";

const SESSION_COOKIE = "cs_session";

/**
 * Routes reachable without a session.
 *
 * `/api/ingest` is here because the SDK posts from end users' browsers and
 * they are, by construction, not members — consent is that route's gate, not
 * a session. It forwards only to /v1/ingest and injects no API key.
 *
 * Matched EXACTLY, not by prefix. The previous `startsWith` check made every
 * path with a public prefix public too: `/loginfoo` was open, and adding the
 * ingest route under the old rule would have opened `/api/ingestwhatever`
 * along with it — an unauthenticated hole created by a route that was only
 * ever meant to be one URL.
 */
const PUBLIC_PATHS = new Set(["/login", "/api/auth/login", "/api/ingest"]);

function secret(): Uint8Array {
  const raw = process.env.AUTH_SECRET;
  if (!raw) {
    // auth.ts throws in production for exactly this reason; middleware had its
    // own silent copy of the dev secret, so a deployment missing AUTH_SECRET
    // would have every session forgeable by anyone who has read this repo,
    // while the app itself looked fine.
    if (process.env.NODE_ENV === "production") {
      throw new Error("AUTH_SECRET must be set in production");
    }
    return new TextEncoder().encode("cogniswarm-dev-secret-do-not-ship");
  }
  return new TextEncoder().encode(raw);
}

export async function middleware(request: NextRequest) {
  const { pathname } = request.nextUrl;
  if (PUBLIC_PATHS.has(pathname)) {
    return NextResponse.next();
  }

  const token = request.cookies.get(SESSION_COOKIE)?.value;
  if (token) {
    try {
      await jwtVerify(token, secret());
      return NextResponse.next();
    } catch {
      /* fall through to redirect */
    }
  }

  if (pathname.startsWith("/api/")) {
    return NextResponse.json({ detail: "Unauthorized" }, { status: 401 });
  }
  const login = new URL("/login", request.url);
  login.searchParams.set("from", pathname);
  return NextResponse.redirect(login);
}

export const config = {
  /**
   * `sdk/` is excluded because the collector bundle is fetched by <script> tags
   * on OTHER PEOPLE'S websites, where there is no session and never will be.
   * Without this the middleware answered every SDK request with a 307 to
   * /login — verified — so every integration would have loaded an HTML login
   * page instead of the collector and silently collected nothing.
   *
   * It is safe to exclude: the directory contains only published, immutable,
   * versioned build artifacts, and the SRI hash printed at release time is
   * what lets customers detect any substitution.
   */
  matcher: [
    "/((?!_next/static|_next/image|favicon.ico|sdk/|.*\\.(?:svg|png|jpg|woff2)$).*)",
  ],
};
