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

def _default_db_path() -> Path:
    """Where the database lives when COGNISWARM_DB is unset.

    In a container the repo tree is an image layer, so a database written there
    is destroyed by every redeploy — silently, because SQLite happily creates a
    fresh empty file and the app starts perfectly. Preferring a mounted /data
    volume when one exists makes the container case correct by default while
    leaving local development exactly as it was.

    Set COGNISWARM_DB explicitly to override either way.
    """
    data_dir = Path("/data")
    if data_dir.is_dir() and os.access(data_dir, os.W_OK):
        return data_dir / "cogniswarm.db"
    return REPO_ROOT / "cogniswarm.db"


DB_PATH = Path(os.environ["COGNISWARM_DB"]) if os.environ.get("COGNISWARM_DB") else _default_db_path()

# Max concurrent twin requests against the OpenAI API.
SWARM_CONCURRENCY = int(os.environ.get("COGNISWARM_CONCURRENCY", "8"))

# Hard ceiling on the twin calls ONE request may dispatch.
#
# The per-field caps (twins_per_persona <= 20, variants <= 8) look like they
# bound this, and they do not: the persona count is the multiplier and it comes
# from the load window, not from the request. Worst case before this existed —
# compare with 200 personas x 20 twins x 8 variants = 32,000 twin calls from a
# single unpriced button press, and a multi-step walk was 24,000.
#
# Enforced as a REFUSAL rather than a truncation. Silently running a smaller
# swarm than asked for would produce a result whose twin_count nobody reads,
# which is the same class of error as the survivor-count bug: a number that
# looks complete and is not.
MAX_TWINS_PER_RUN = int(os.environ.get("COGNISWARM_MAX_TWINS_PER_RUN", "2000"))
