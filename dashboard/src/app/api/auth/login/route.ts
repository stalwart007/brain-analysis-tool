import { NextRequest, NextResponse } from "next/server";
import {
  SESSION_COOKIE,
  createSessionToken,
  sessionCookieOptions,
  verifyCredentials,
} from "@/lib/auth";

// Simple in-memory throttle: 5 attempts / 15 min per (ip, email).
const attempts = new Map<string, { count: number; resetAt: number }>();
const MAX_ATTEMPTS = 5;
const WINDOW_MS = 15 * 60 * 1000;

function throttled(key: string): boolean {
  const now = Date.now();
  const entry = attempts.get(key);
  if (!entry || now > entry.resetAt) {
    attempts.set(key, { count: 1, resetAt: now + WINDOW_MS });
    return false;
  }
  entry.count += 1;
  return entry.count > MAX_ATTEMPTS;
}

export async function POST(request: NextRequest) {
  const body = await request.json().catch(() => null);
  const email = typeof body?.email === "string" ? body.email.trim() : "";
  const password = typeof body?.password === "string" ? body.password : "";
  if (!email || !password) {
    return NextResponse.json({ detail: "Email and password required" }, { status: 400 });
  }

  const ip = request.headers.get("x-forwarded-for")?.split(",")[0] ?? "local";
  if (throttled(`${ip}:${email.toLowerCase()}`)) {
    return NextResponse.json(
      { detail: "Too many attempts. Try again in 15 minutes." },
      { status: 429 }
    );
  }

  const user = await verifyCredentials(email, password);
  if (!user) {
    return NextResponse.json({ detail: "Invalid credentials" }, { status: 401 });
  }

  const response = NextResponse.json({ ok: true, name: user.name ?? user.email });
  response.cookies.set(SESSION_COOKIE, await createSessionToken(user.email), sessionCookieOptions);
  return response;
}
