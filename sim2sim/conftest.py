"""Make ``quadruped_sim`` importable however pytest is invoked."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
