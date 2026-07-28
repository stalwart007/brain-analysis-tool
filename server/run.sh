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

echo "✓ Key loaded (…${OPENAI_API_KEY: -4}).  Starting backend on http://localhost:8000"
exec .venv/bin/uvicorn app.main:app --app-dir . --port 8000
