"""Put the reference package and the generated module on sys.path for the tests."""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "reference", REPO / "generated" / "py", REPO / "tools"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
