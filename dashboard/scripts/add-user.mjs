#!/usr/bin/env node
/**
 * Provision a member of the closed group:
 *   npm run add-user -- alice@team.com  s3cret-password  "Alice"
 * Upserts into data/users.json with a bcrypt hash. No signup UI exists — this
 * script (run by an admin) is the only way in.
 */
import bcrypt from "bcryptjs";
import { promises as fs } from "fs";
import path from "path";

const [email, password, name] = process.argv.slice(2);
if (!email || !password) {
  console.error('Usage: npm run add-user -- <email> <password> ["Display Name"]');
  process.exit(1);
}
if (password.length < 8) {
  console.error("Password must be at least 8 characters.");
  process.exit(1);
}

// Must resolve identically to USERS_PATH in src/lib/auth.ts, or you provision
// accounts into a file the app never reads.
const usersPath =
  process.env.COGNISWARM_USERS_FILE ?? path.join(process.cwd(), "data", "users.json");
await fs.mkdir(path.dirname(usersPath), { recursive: true });

let users = [];
try {
  users = JSON.parse(await fs.readFile(usersPath, "utf8"));
} catch {
  /* first user */
}

const passwordHash = await bcrypt.hash(password, 12);
const existing = users.findIndex(
  (u) => u.email.toLowerCase() === email.toLowerCase()
);
if (existing >= 0) {
  users[existing] = { ...users[existing], passwordHash, name };
  console.log(`Updated ${email}`);
} else {
  users.push({ email, passwordHash, name });
  console.log(`Added ${email}`);
}

await fs.writeFile(usersPath, JSON.stringify(users, null, 2) + "\n");
console.log(`-> ${usersPath} (${users.length} member${users.length === 1 ? "" : "s"})`);
