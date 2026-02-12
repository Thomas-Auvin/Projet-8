from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient


class DummyPipe:
    def predict_proba(self, X):
        # shape (n, 2) attendu ; proba = colonne 1
        n = len(X)
        return np.tile(np.array([[0.2, 0.8]]), (n, 1))


def _dummy_load_model(_artifacts_dir):
    # Modèle minimal pour démarrer l'app
    return SimpleNamespace(
        feature_names=["DAYS_BIRTH", "AMT_INCOME_TOTAL"],
        pipeline=DummyPipe(),
        threshold=0.5,
        model_version="dummy-test",
        meta={},  # pas de requested_groups/min_rate pour ne pas gêner les tests
    )


@pytest.fixture()
def client(monkeypatch, tmp_path):
    # Evite la DB en tests (même si store.init() est appelé)
    os.environ["P8_DISABLE_DB_LOG"] = "1"
    os.environ["P8_DB_ASYNC_WRITER"] = "0"
    os.environ["P8_DB_PATH"] = str(tmp_path / "predictions.sqlite")
    os.environ["P8_STRICT_INPUT"] = "1"

    # Patch du loader AVANT le démarrage FastAPI (lifespan)
    import app.main as main

    monkeypatch.setattr(main, "load_model", _dummy_load_model)

    with TestClient(main.app) as c:
        yield c


def test_predict_rejects_days_birth_positive(client: TestClient) -> None:
    r = client.post("/predict", json={"features": {"DAYS_BIRTH": 10}})
    assert r.status_code == 422
    body = r.json()
    assert body["detail"]["error"] == "invalid_input"
    assert "DAYS_BIRTH" in body["detail"]["message"]


def test_predict_rejects_days_birth_too_young(client: TestClient) -> None:
    # ~10 ans
    r = client.post("/predict", json={"features": {"DAYS_BIRTH": -3650}})
    assert r.status_code == 422
    body = r.json()
    assert body["detail"]["error"] == "invalid_input"
    assert "Âge incohérent" in body["detail"]["message"]


def test_predict_rejects_income_zero(client: TestClient) -> None:
    r = client.post("/predict", json={"features": {"AMT_INCOME_TOTAL": 0}})
    assert r.status_code == 422
    body = r.json()
    assert body["detail"]["error"] == "invalid_input"
    assert "AMT_INCOME_TOTAL" in body["detail"]["message"]


def test_predict_rejects_income_negative(client: TestClient) -> None:
    r = client.post("/predict", json={"features": {"AMT_INCOME_TOTAL": -1}})
    assert r.status_code == 422
    body = r.json()
    assert body["detail"]["error"] == "invalid_input"
    assert "AMT_INCOME_TOTAL" in body["detail"]["message"]


def test_predict_rejects_invalid_numeric_string(client: TestClient) -> None:
    r = client.post("/predict", json={"features": {"AMT_INCOME_TOTAL": "2020-01-01"}})
    assert r.status_code == 422
    body = r.json()
    assert body["detail"]["error"] == "invalid_input"
    assert "Valeur numérique invalide" in body["detail"]["message"]
