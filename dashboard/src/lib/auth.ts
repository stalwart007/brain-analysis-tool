/**
 * Closed-group authentication.
 *
 * There is deliberately NO signup flow anywhere in this app. Members are
 * provisioned by an admin via `npm run add-user -- <email> <password>`, which
 * writes a bcrypt hash into data/users.json. Sessions are stateless signed
 * JWTs in an httpOnly cookie, verified by middleware on every request.
 */

import { promises as fs } from "fs";
import path from "path";
import bcrypt from "bcryptjs";
import { SignJWT, jwtVerify } from "jose";

export const SESSION_COOKIE = "cs_session";
const SESSION_TTL_SECONDS = 60 * 60 * 24 * 7; // 7 days

function secret(): Uint8Array {
  const raw = process.env.AUTH_SECRET;
  if (!raw) {
    if (process.env.NODE_ENV === "production") {
      throw new Error("AUTH_SECRET must be set in production");
    }
    return new TextEncoder().encode("cogniswarm-dev-secret-do-not-ship");
  }
  return new TextEncoder().encode(raw);
}

/**
 * Where the member list lives.
 *
 * Overridable because in a container this file is STATE, not config: it is
 * written by `npm run add-user`, so an image with it baked in loses every
 * account on the next rebuild — and the symptom is "my password stopped
 * working", which reads as an auth bug rather than a deployment one. Point
 * COGNISWARM_USERS_FILE at a mounted secret or volume in production.
 */
export const USERS_PATH =
  process.env.COGNISWARM_USERS_FILE ?? path.join(process.cwd(), "data", "users.json");

interface UserRecord {
  email: string;
  passwordHash: string;
  name?: string;
}

export async function findUser(email: string): Promise<UserRecord | null> {
  let raw: string;
  try {
    raw = await fs.readFile(USERS_PATH, "utf8");
  } catch {
    // Distinguish "nobody provisioned yet" from "the mount is missing" in the
    // server log. Both must still return null to the caller — leaking which
    // one it is would tell an attacker whether the deployment is misconfigured
    // — but an operator staring at a login that rejects every correct password
    // needs this line to exist.
    console.error(
      `[auth] no readable user file at ${USERS_PATH} — ` +
        "provision with `npm run add-user`, or set COGNISWARM_USERS_FILE to a mounted file"
    );
    return null;
  }
  let users: UserRecord[];
  try {
    users = JSON.parse(raw);
  } catch {
    console.error(`[auth] user file at ${USERS_PATH} is not valid JSON`);
    return null;
  }
  return users.find((u) => u.email.toLowerCase() === email.toLowerCase()) ?? null;
}

export async function verifyCredentials(
  email: string,
  password: string
): Promise<UserRecord | null> {
  const user = await findUser(email);
  // Always run a compare so response timing doesn't reveal whether the email exists.
  const hash =
    user?.passwordHash ??
    "$2a$12$C6UzMDM.H6dfI/f/IKcEeO7ZBVdY0V1O3P8mRj6P8mRj6P8mRj6P8"; // dummy
  const ok = await bcrypt.compare(password, hash);
  return ok && user ? user : null;
}

export async function createSessionToken(email: string): Promise<string> {
  return new SignJWT({ sub: email })
    .setProtectedHeader({ alg: "HS256" })
    .setIssuedAt()
    .setExpirationTime(`${SESSION_TTL_SECONDS}s`)
    .sign(secret());
}

export async function verifySessionToken(token: string): Promise<string | null> {
  try {
    const { payload } = await jwtVerify(token, secret());
    return typeof payload.sub === "string" ? payload.sub : null;
  } catch {
    return null;
  }
}

export const sessionCookieOptions = {
  httpOnly: true,
  sameSite: "lax" as const,
  secure: process.env.NODE_ENV === "production",
  path: "/",
  maxAge: SESSION_TTL_SECONDS,
};
