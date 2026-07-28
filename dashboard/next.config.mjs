/** @type {import('next').NextConfig} */
const nextConfig = {
  // Backend access goes through the session-gated forwarder at
  // src/app/api/cs/[...path]/route.ts (configure with COGNISWARM_BACKEND).

  // Let a verification build target its own directory. Without this, running
  // `next build` while `next dev` is running writes a production build straight
  // into `.next` and leaves the dev server serving a broken tree (the symptom
  // is a page that renders as completely unstyled HTML).
  //   BUILD_DIR=.next-verify npx next build
  ...(process.env.BUILD_DIR ? { distDir: process.env.BUILD_DIR } : {}),
};

export default nextConfig;
