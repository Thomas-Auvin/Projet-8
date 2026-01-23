# tests/conftest.py
import sys
from pathlib import Path
import warnings

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pytest_configure():
    warnings.filterwarnings(
        "ignore",
        message=r"X does not have valid feature names, but .* was fitted with feature names",
        category=UserWarning,
    )


@pytest.fixture()
def p8_env(monkeypatch, tmp_path):
    """
    Configure un environnement de test stable :
    - DB SQLite temporaire
    - input relax
    - artifacts de test légers (commités)
    """
    monkeypatch.setenv("P8_DB_PATH", str(tmp_path / "preds.sqlite"))
    monkeypatch.setenv("P8_STRICT_INPUT", "0")

    artifacts = (ROOT / "tests" / "assets" / "artifacts").resolve()
    monkeypatch.setenv("P8_ARTIFACTS_DIR", str(artifacts))

    return tmp_path
