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

const USERS_PATH = path.join(process.cwd(), "data", "users.json");

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
    return null; // no users provisioned yet
  }
  const users: UserRecord[] = JSON.parse(raw);
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
