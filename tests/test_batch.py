import json
import pandas as pd
from fastapi.testclient import TestClient


def test_predict_batch(p8_env):
    from app.main import app

    with TestClient(app) as client:
        df = pd.read_csv("data/reference/reference_sample.csv").head(3)

        # ✅ NaN -> null + types JSON-safe
        rows = json.loads(df.to_json(orient="records"))
        payload = {"rows": rows}

        r = client.post("/predict_batch", json=payload)
        assert r.status_code == 200, r.text
