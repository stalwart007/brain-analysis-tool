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

  async headers() {
    return [
      {
        source: "/:path*",
        headers: [
          // This app renders LLM-authored strings — inner_monologue,
          // likely_mindset, friction_notes — that originate from a model
          // reading attacker-influenceable input. React escapes them, but a
          // CSP is the layer that holds when something is rendered unescaped
          // later. `unsafe-inline`/`unsafe-eval` for scripts are what Next's
          // dev overlay and R3F shader compilation currently need; tightening
          // those to a nonce is the obvious next step and needs its own
          // testing pass, so it is deliberately not being done blind tonight.
          {
            key: "Content-Security-Policy",
            value: [
              "default-src 'self'",
              "script-src 'self' 'unsafe-inline' 'unsafe-eval'",
              "style-src 'self' 'unsafe-inline'",
              "img-src 'self' data: blob:",
              // Keyframe extraction decodes video in a `<video>` element fed
              // from `URL.createObjectURL(file)`, so media has to be allowed
              // from `blob:` — and it was not. There is no `media-src` in a
              // default CSP, so it inherited `default-src 'self'`, and every
              // video the browser tried to decode failed with
              // `MEDIA_ELEMENT_ERROR: Media load rejected by URL safety check`.
              //
              // That silently broke BOTH video paths — uploading a file and
              // pasting a hosted link — because both end at the same decoder.
              // The failure surfaced as "Could not decode that video in this
              // browser", which reads as a codec problem and sends the reader
              // to re-encode a file that was fine.
              //
              // `blob:` only, deliberately: no remote origin is added, so this
              // permits decoding bytes this page already holds and nothing
              // else. Video and audio never come from a third party directly —
              // they come through /api/cs/content/media, same-origin.
              "media-src 'self' blob:",
              "font-src 'self' data:",
              // The browser only ever talks to this origin: the backend is
              // private and reached server-side through /api/cs.
              "connect-src 'self'",
              "worker-src 'self' blob:",
              "object-src 'none'",
              "base-uri 'self'",
              "form-action 'self'",
              // Clickjacking: this app has one-click destructive controls
              // (panel revoke erases a member's telemetry permanently).
              "frame-ancestors 'none'",
            ].join("; "),
          },
          { key: "X-Content-Type-Options", value: "nosniff" },
          { key: "X-Frame-Options", value: "DENY" },
          // Page paths are the PII channel this codebase already templates
          // server-side; do not hand them to third parties in a Referer.
          { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
          {
            key: "Permissions-Policy",
            value: "camera=(), microphone=(), geolocation=(), interest-cohort=()",
          },
          // Only meaningful over HTTPS; harmless on localhost, which browsers
          // exempt. Fly terminates TLS, so this is live in production.
          {
            key: "Strict-Transport-Security",
            value: "max-age=31536000; includeSubDomains",
          },
        ],
      },
      {
        // The SDK is embedded by third-party sites by design, so it needs the
        // opposite of the default: cross-origin readable, and immutable
        // because the URL carries the version (see the release script).
        source: "/sdk/:version/:file*",
        headers: [
          { key: "Access-Control-Allow-Origin", value: "*" },
          { key: "Cache-Control", value: "public, max-age=31536000, immutable" },
        ],
      },
    ];
  },
};

export default nextConfig;
