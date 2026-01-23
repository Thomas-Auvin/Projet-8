import json
import pandas as pd
from fastapi.testclient import TestClient


def test_predict_batch(monkeypatch, tmp_path):
    monkeypatch.setenv("P8_DB_PATH", str(tmp_path / "preds.sqlite"))
    monkeypatch.setenv("P8_STRICT_INPUT", "0")

    from app.main import app

    with TestClient(app) as client:  # ✅ indispensable avec lifespan
        df = pd.read_csv("data/reference/reference_sample.csv").head(3)

        # ✅ NaN -> null + types JSON-safe
        rows = json.loads(df.to_json(orient="records"))
        payload = {"rows": rows}

        r = client.post("/predict_batch", json=payload)

        # Debug utile si ça re-plante :
        assert r.status_code == 200, r.text
