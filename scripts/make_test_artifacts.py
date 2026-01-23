from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline


def main() -> None:
    out_dir = Path("tests/assets/artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)

    pipe = Pipeline([("clf", DummyClassifier(strategy="prior"))])

    # Fit minimal pour activer predict_proba
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([0, 1, 0])
    pipe.fit(X, y)

    joblib.dump(pipe, out_dir / "model.joblib")

    meta = {
        "model_version": "test-model-v1",
        "threshold": 0.5,
        "feature_names": ["SK_ID_CURR"],
    }
    (out_dir / "model_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"✅ Test artifacts written to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
