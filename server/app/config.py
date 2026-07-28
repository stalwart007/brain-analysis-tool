import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Layer-4 profiling + swarm aggregation: highest-quality reasoning.
PROFILER_MODEL = os.environ.get("COGNISWARM_PROFILER_MODEL", "gpt-4o")

# Per-twin model: cheap and fast at swarm scale. Overridable per deployment.
TWIN_MODEL = os.environ.get("COGNISWARM_TWIN_MODEL", "gpt-4o-mini")

# Embeddings power semantic clustering of free-text objections (so that
# "too pricey" and "the price is too high" collapse into one theme).
EMBEDDING_MODEL = os.environ.get("COGNISWARM_EMBEDDING_MODEL", "text-embedding-3-small")

# Cognitive-load → sampling temperature. Higher load = more erratic, impulsive,
# "System 1" behavior — this is the Challenge-3 "parameter shifting" lever,
# now literally available because OpenAI models accept `temperature`.
LOAD_TO_TEMPERATURE = {"low": 0.45, "medium": 0.85, "high": 1.15}

DB_PATH = Path(os.environ.get("COGNISWARM_DB", REPO_ROOT / "cogniswarm.db"))

# Max concurrent twin requests against the OpenAI API.
SWARM_CONCURRENCY = int(os.environ.get("COGNISWARM_CONCURRENCY", "8"))
