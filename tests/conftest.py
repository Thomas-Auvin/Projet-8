# tests/conftest.py
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import warnings

def pytest_configure():
    warnings.filterwarnings(
        "ignore",
        message="X does not have valid feature names, but .* was fitted with feature names",
        category=UserWarning,
    )
