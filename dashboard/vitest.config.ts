/**
 * Frontend tests. There were none before this file.
 *
 * The gap mattered more than it looked. Typecheck proves the shapes line up and
 * the production build proves it compiles; neither says anything about whether
 * a function is CORRECT. Every frontend defect fixed in this codebase so far —
 * the picker inheriting a stale kind, the seek that hung forever, the object
 * URL leaked on the error path, the stream that could not be cancelled — passed
 * `tsc` and `next build` cleanly on the way in.
 *
 * `node` rather than `jsdom` by default: most of what is worth testing here is
 * pure (cue parsing, clock formatting, the SSE framer, the quadrant verdict),
 * and a DOM is opted into per-file with a `@vitest-environment jsdom` docblock
 * where a hook or component genuinely needs one. Keeping the default cheap is
 * what makes the suite fast enough to actually run.
 */

import { defineConfig } from "vitest/config";
import path from "node:path";

export default defineConfig({
  // JSX in tests, without a plugin. Next compiles the app itself; vitest runs
  // the files directly and needs its own transform, or every `.tsx` test fails
  // on `React is not defined`. esbuild does this natively, so this avoids
  // adding `@vitejs/plugin-react` — which in this tree hits a peer conflict
  // and would be a build dependency earning nothing the runtime already has.
  esbuild: { jsx: "automatic" },
  test: {
    environment: "node",
    include: ["src/**/*.test.{ts,tsx}"],
    // Fail on an unhandled rejection rather than printing it and passing. Two
    // of the bugs this suite now covers were unhandled rejections in practice.
    dangerouslyIgnoreUnhandledErrors: false,
    coverage: {
      provider: "v8",
      include: ["src/lib/**", "src/components/**"],
      reporter: ["text-summary"],
    },
  },
  resolve: {
    alias: { "@": path.resolve(__dirname, "src") },
  },
});
