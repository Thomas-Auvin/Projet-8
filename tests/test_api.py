from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

EXAMPLE_CSV = Path("app/examples/example_input_compact.csv")
EXAMPLE_JSON = Path("app/examples/example_input_compact.json")


@pytest.fixture
def client(p8_env, monkeypatch):
    # On coupe l'async logging pour des tests plus déterministes
    monkeypatch.setenv("P8_ASYNC_LOG", "0")
    from app.main import app

    with TestClient(app) as c:
        yield c


def _get_any_allowed_key(client: TestClient) -> str:
    r = client.get("/features")
    assert r.status_code == 200, r.text
    data = r.json()
    allowed = data["allowed_input_keys"]
    assert isinstance(allowed, list) and allowed
    return "SK_ID_CURR" if "SK_ID_CURR" in allowed else allowed[0]


def test_root_redirects_to_docs(client):
    r = client.get("/", follow_redirects=False)
    assert r.status_code in (302, 307)
    loc = r.headers.get("location") or r.headers.get("Location")
    assert loc is not None
    assert loc.endswith("/docs")


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert isinstance(data["n_features"], int)
    assert data["n_features"] > 0


def test_model_info(client):
    r = client.get("/model")
    assert r.status_code == 200
    data = r.json()
    assert "model_version" in data
    assert "threshold" in data
    assert "n_features" in data
    assert data["n_features"] > 0


def test_features_schema(client):
    r = client.get("/features")
    assert r.status_code == 200
    data = r.json()
    assert "feature_names" in data
    assert "allowed_input_keys" in data
    assert "onehot_groups" in data
    assert data["n_features"] > 0
    assert isinstance(data["allowed_input_keys"], list)
    assert len(data["allowed_input_keys"]) > 0


def test_examples_listing_and_files(client):
    r = client.get("/examples")
    assert r.status_code == 200
    files = r.json()["files"]
    assert "example_input_compact.csv" in files
    assert "example_input_compact.json" in files

    r = client.get("/examples/example_input_compact.csv")
    assert r.status_code == 200
    assert "text/csv" in r.headers.get("content-type", "")

    r = client.get("/examples/example_input_compact.json")
    assert r.status_code == 200
    assert isinstance(r.json(), dict)


def test_predict_ok_creates_db_file(p8_env, client):
    db_path = p8_env / "preds.sqlite"
    key = _get_any_allowed_key(client)

    r = client.post("/predict", json={"features": {key: 100001}})
    assert r.status_code == 200, r.text

    data = r.json()
    assert 0.0 <= float(data["proba_default"]) <= 1.0
    assert data["decision"] in (0, 1)
    assert db_path.exists()


def test_predict_unknown_key_strict_422(monkeypatch, client):
    monkeypatch.setenv("P8_STRICT_INPUT", "1")
    key = _get_any_allowed_key(client)

    r = client.post(
        "/predict", json={"features": {key: 100001, "UNKNOWN_KEY_FOR_TEST": 1}}
    )
    assert r.status_code == 422
    detail = r.json().get("detail", {})
    assert detail.get("error") == "invalid_input"


def test_predict_unknown_key_non_strict_ok(monkeypatch, client):
    monkeypatch.setenv("P8_STRICT_INPUT", "0")
    key = _get_any_allowed_key(client)

    r = client.post(
        "/predict", json={"features": {key: 100001, "UNKNOWN_KEY_FOR_TEST": 1}}
    )
    assert r.status_code == 200, r.text


def test_predict_insufficient_filled_requested_422(monkeypatch, client):
    monkeypatch.setenv("P8_STRICT_INPUT", "1")
    key1 = _get_any_allowed_key(client)

    loaded = client.app.state.loaded
    orig_meta = getattr(loaded, "meta", None)
    # On force 2 groupes demandés dont un "fantôme" => impossible d'atteindre 100% remplissage
    loaded.meta = {
        "requested_groups": [key1, "__MISSING_GROUP_FOR_TEST__"],
        "min_filled_rate_requested": 1.0,
    }
    try:
        r = client.post("/predict", json={"features": {key1: 100001}})
        assert r.status_code == 422
        detail = r.json().get("detail", {})
        assert detail.get("error") == "insufficient_filled_requested"
        assert detail.get("missing_first_20")
    finally:
        loaded.meta = orig_meta


def test_predict_decision_function_path(client):
    class DummyPipe:
        def decision_function(self, X):
            return np.zeros(X.shape[0], dtype=float)

    loaded = client.app.state.loaded
    orig_pipe = loaded.pipeline
    loaded.pipeline = DummyPipe()

    try:
        key = _get_any_allowed_key(client)
        r = client.post("/predict", json={"features": {key: 100001}})
        assert r.status_code == 200, r.text
        data = r.json()
        assert abs(float(data["proba_default"]) - 0.5) < 1e-9
    finally:
        loaded.pipeline = orig_pipe


def test_predict_batch_ok(monkeypatch, client):
    monkeypatch.setenv("P8_STRICT_INPUT", "0")
    key = _get_any_allowed_key(client)

    payload = {"rows": [{key: 100001}, {key: 100002}]}
    r = client.post("/predict_batch", json=payload)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["n_rows"] == 2
    assert len(data["items"]) == 2


def test_predict_csv_json_and_csv(monkeypatch, client):
    monkeypatch.setenv("P8_STRICT_INPUT", "0")

    assert EXAMPLE_CSV.exists(), f"Missing example CSV: {EXAMPLE_CSV}"
    files = {"file": (EXAMPLE_CSV.name, EXAMPLE_CSV.read_bytes(), "text/csv")}

    r = client.post("/predict_csv?output=json", files=files)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["n_rows"] >= 1
    assert "items" in data

    r = client.post("/predict_csv?output=csv", files=files)
    assert r.status_code == 200, r.text
    assert "text/csv" in r.headers.get("content-type", "")
    assert "Content-Disposition" in r.headers


def test_predict_csv_invalid_extension_415(client):
    files = {"file": ("bad.txt", b"hello", "text/plain")}
    r = client.post("/predict_csv?output=json", files=files)
    assert r.status_code == 415


def test_predict_csv_empty_csv_400(client):
    files = {"file": ("empty.csv", b"SK_ID_CURR\n", "text/csv")}
    r = client.post("/predict_csv?output=json", files=files)
    assert r.status_code == 400


def test_predict_missing_model(monkeypatch, tmp_path):
    monkeypatch.setenv("P8_ARTIFACTS_DIR", str(tmp_path / "empty_artifacts"))
    monkeypatch.setenv("P8_DB_PATH", str(tmp_path / "preds.sqlite"))
    monkeypatch.setenv("P8_STRICT_INPUT", "0")
    monkeypatch.setenv("P8_ASYNC_LOG", "0")

    from app.main import app

    with pytest.raises(Exception):
        with TestClient(app):
            pass


def test_helpers_json_safe_and_sigmoid():
    from app.main import _json_safe_features, _sigmoid, _sigmoid_vec

    d = {
        "b": np.bool_(True),
        "i": np.int64(3),
        "f": np.float64(1.25),
        "nan": np.float64(np.nan),
        "inf": np.float64(np.inf),
    }
    out = _json_safe_features(d)
    assert out["b"] is True
    assert out["i"] == 3
    assert out["f"] == 1.25
    assert out["nan"] is None
    assert out["inf"] is None

    assert abs(_sigmoid(0.0) - 0.5) < 1e-12
    v = _sigmoid_vec(np.array([0.0, 0.0], dtype=float))
    assert v.shape == (2,)
    assert float(v[0]) == 0.5
