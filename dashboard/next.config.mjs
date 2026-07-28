/** @type {import('next').NextConfig} */
const nextConfig = {
  // Backend access goes through the session-gated forwarder at
  // src/app/api/cs/[...path]/route.ts (configure with COGNISWARM_BACKEND).

  // Emit a self-contained server bundle with only the traced dependencies, so
  // the runtime image copies ~120 MB instead of the whole node_modules tree
  // (three.js, hls.js and the Next toolchain alone push that past a gigabyte).
  // Harmless outside Docker: it writes an extra .next/standalone directory and
  // changes nothing about `next dev` or `next start`.
  output: "standalone",

  // Let a verification build target its own directory. Without this, running
  // `next build` while `next dev` is running writes a production build straight
  // into `.next` and leaves the dev server serving a broken tree (the symptom
  // is a page that renders as completely unstyled HTML).
  //   BUILD_DIR=.next-verify npx next build
  ...(process.env.BUILD_DIR ? { distDir: process.env.BUILD_DIR } : {}),
};

export default nextConfig;
