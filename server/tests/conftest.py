import os
import tempfile

# Must run before any `app.*` import so config picks up the throwaway DB.
os.environ.setdefault("COGNISWARM_DB", os.path.join(tempfile.mkdtemp(), "test.db"))

# The analysis surface fails CLOSED when no API keys are configured, so the
# suite has to declare its posture explicitly rather than relying on the absence
# of configuration to mean "open". Tests that exercise key enforcement set
# COGNISWARM_API_KEYS themselves, and configured keys always take precedence
# over this flag.
os.environ.setdefault("COGNISWARM_ALLOW_ANONYMOUS", "1")
