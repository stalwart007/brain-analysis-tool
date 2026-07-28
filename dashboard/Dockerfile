# CogniSwarm dashboard.
#
# Multi-stage against Next's standalone output: the runtime image carries the
# traced server bundle rather than node_modules, which matters here because
# three.js, hls.js and the Next toolchain would otherwise push it past a
# gigabyte.

FROM node:22-slim AS deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci


FROM node:22-slim AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .

# A THROWAWAY secret, used only so `next build` can prerender pages whose module
# graph reaches auth.ts — which throws when AUTH_SECRET is unset in production.
#
# It must never survive into the runtime stage. Session cookies are signed with
# whatever AUTH_SECRET the *running* container has; if this build-time value
# leaked forward, every session would be signed with a constant that is sitting
# in this Dockerfile in plain text, and anyone who has read the repo could forge
# one. It is scoped to this stage, and the runtime stage below never sets it —
# so a missing AUTH_SECRET at run time is a startup crash rather than a silent
# downgrade to a public secret.
ENV AUTH_SECRET=build-time-placeholder-never-used-at-runtime
ENV NEXT_TELEMETRY_DISABLED=1
RUN npm run build


FROM node:22-slim AS runtime
WORKDIR /app

ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    PORT=3000 \
    HOSTNAME=0.0.0.0

# The member list is STATE, not config — `npm run add-user` writes it. Baking
# it into the image loses every account on the next rebuild, and the symptom is
# "my password stopped working", which reads as an auth bug rather than a
# deployment one. Mount a file here instead; auth.ts logs plainly when it is
# missing.
ENV COGNISWARM_USERS_FILE=/secrets/users.json

COPY --from=builder /app/public ./public
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static

RUN useradd --system --uid 10002 --create-home nextjs \
    && mkdir -p /secrets \
    && chown -R nextjs:nextjs /app /secrets
USER nextjs

EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=3s --start-period=15s --retries=3 \
    CMD node -e "require('http').get('http://127.0.0.1:3000/login',r=>process.exit(r.statusCode===200?0:1)).on('error',()=>process.exit(1))"

CMD ["node", "server.js"]
