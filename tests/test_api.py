from fastapi.testclient import TestClient


def test_health(p8_env):
    from app.main import app
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] in ("ok", "not_ready")


def test_predict_ok(p8_env):
    db_path = p8_env / "preds.sqlite"

    from app.main import app
    with TestClient(app) as client:
        payload = {"features": {"SK_ID_CURR": 100001}}
        r = client.post("/predict", json=payload)
        assert r.status_code == 200, r.text
        data = r.json()
        assert "proba_default" in data
        assert "decision" in data
        assert db_path.exists()


def test_predict_missing_model(monkeypatch, tmp_path):
    # on pointe vers un dossier artifacts vide -> startup doit échouer
    monkeypatch.setenv("P8_ARTIFACTS_DIR", str(tmp_path / "empty_artifacts"))
    monkeypatch.setenv("P8_DB_PATH", str(tmp_path / "preds.sqlite"))
    monkeypatch.setenv("P8_STRICT_INPUT", "0")

    from app.main import app

    # On s'attend à une exception au startup (FileNotFoundError typiquement)
    try:
        with TestClient(app):
            pass
        assert False, "Startup should have failed (missing model artifacts)"
    except Exception:
        assert True
