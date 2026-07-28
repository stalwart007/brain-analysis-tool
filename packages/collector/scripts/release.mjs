#!/usr/bin/env node
/**
 * Publish the collector to the dashboard's static tree at an IMMUTABLE,
 * VERSIONED url, and print the SRI hash customers pin against.
 *
 *   npm run release --prefix packages/collector
 *
 * Why versioned and immutable. Customers embed this with a <script src> on
 * their own production sites. A single mutable URL means every change you ship
 * reaches every customer's live site within a cache TTL, with no staging and
 * no rollback — one bad build breaks all of them at once. Immutable
 * `/sdk/v<version>/cogniswarm.min.js` makes upgrading an explicit act by the
 * customer, which is the only version of this that is safe to operate.
 *
 * Why the SRI hash. With an immutable URL a customer can pin
 * `integrity="sha384-…"`, so a compromise of the host cannot silently swap
 * arbitrary JavaScript into their page. This script prints the exact tag to
 * hand them; it is meaningless without immutability, which is the other half
 * of why the URL carries the version.
 *
 * The built bundle is gitignored, so before this existed the shipped artifact
 * lived only on whichever machine last ran `npm run build`.
 */

import { createHash } from "crypto";
import { promises as fs } from "fs";
import path from "path";
import { fileURLToPath } from "url";

const here = path.dirname(fileURLToPath(import.meta.url));
const pkgRoot = path.resolve(here, "..");
const repoRoot = path.resolve(pkgRoot, "..", "..");

const pkg = JSON.parse(await fs.readFile(path.join(pkgRoot, "package.json"), "utf8"));
const version = pkg.version;

const source = path.join(pkgRoot, "dist", "cogniswarm.min.js");
let bundle;
try {
  bundle = await fs.readFile(source);
} catch {
  console.error(`✗ No build at ${source}\n  Run: npm run build --prefix packages/collector`);
  process.exit(1);
}

// SDK_VERSION in transport.ts is stamped into every ingest envelope, so a
// mismatch here means the telemetry corpus is labelled with a version that was
// never published — and version-conditional parsing server-side would silently
// mis-handle it.
const transport = await fs.readFile(path.join(pkgRoot, "src", "transport.ts"), "utf8");
const declared = transport.match(/SDK_VERSION\s*=\s*"([^"]+)"/)?.[1];
if (declared !== version) {
  console.error(
    `✗ Version mismatch: package.json says ${version}, transport.ts SDK_VERSION says ${declared}.\n` +
      "  These are stamped into every ingest envelope; make them agree before releasing."
  );
  process.exit(1);
}

const outDir = path.join(repoRoot, "dashboard", "public", "sdk", `v${version}`);
const outFile = path.join(outDir, "cogniswarm.min.js");

// Refuse to overwrite a published version. Mutating an already-embedded URL is
// the exact failure immutability exists to prevent, so it must be a hard stop
// rather than a warning — bump the version instead.
try {
  const existing = await fs.readFile(outFile);
  if (!existing.equals(bundle)) {
    console.error(
      `✗ v${version} is already published with DIFFERENT contents at\n  ${outFile}\n` +
        "  Customers may already have this URL pinned. Bump the version in\n" +
        "  package.json and transport.ts rather than changing what it serves."
    );
    process.exit(1);
  }
} catch {
  /* not yet published — normal path */
}

await fs.mkdir(outDir, { recursive: true });
await fs.writeFile(outFile, bundle);

const sri = `sha384-${createHash("sha384").update(bundle).digest("base64")}`;
await fs.writeFile(
  path.join(outDir, "integrity.txt"),
  `${sri}\n`,
  "utf8"
);

const kb = (bundle.length / 1024).toFixed(1);
console.log(`✓ Published collector v${version} (${kb} kB)`);
console.log(`  ${path.relative(repoRoot, outFile)}`);
console.log("\n  Embed snippet for customers:\n");
console.log(`    <script src="https://YOUR-HOST/sdk/v${version}/cogniswarm.min.js"`);
console.log(`            integrity="${sri}"`);
console.log(`            crossorigin="anonymous"></script>\n`);
console.log("  Their origin must be listed in COGNISWARM_ALLOWED_ORIGINS on the");
console.log("  server, or the beacon's CORS preflight is refused and every");
console.log("  segment is dropped silently.\n");
