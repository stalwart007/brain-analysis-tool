#!/usr/bin/env bash
# Start the CogniSwarm backend with secrets from .env — one command.
#   ./run.sh
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -f .env ]; then
  echo "✗ No .env file. Copy the template:  cp .env.example .env  then edit it."
  exit 1
fi

# load .env into the environment
set -a
# shellcheck disable=SC1091
source .env
set +a

if [ -z "${OPENAI_API_KEY:-}" ] || [ "${OPENAI_API_KEY}" = "REPLACE_WITH_YOUR_KEY" ]; then
  echo "✗ Edit .env and set OPENAI_API_KEY to your real key, then re-run ./run.sh"
  exit 1
fi

# The analysis surface fails closed, so a .env predating that change starts a
# server that 503s on every endpoint — with the dashboard showing an empty
# database rather than an error. Say so here instead of letting it look broken.
if [ -z "${COGNISWARM_API_KEYS:-}" ] && [ -z "${COGNISWARM_ALLOW_ANONYMOUS:-}" ]; then
  echo "✗ Neither COGNISWARM_API_KEYS nor COGNISWARM_ALLOW_ANONYMOUS is set."
  echo "  Every /v1/* endpoint would return 503 (auth fails closed by design)."
  echo "  For local dev, add this to .env:  COGNISWARM_ALLOW_ANONYMOUS=1"
  echo "  For anything else, set:           COGNISWARM_API_KEYS=key1,key2"
  exit 1
fi

echo "✓ Key loaded (…${OPENAI_API_KEY: -4}).  Starting backend on http://localhost:8000"
exec .venv/bin/uvicorn app.main:app --app-dir . --port 8000
