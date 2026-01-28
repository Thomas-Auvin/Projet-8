from __future__ import annotations

import json
import math
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse

from app.model_loader import LoadedModel, load_model
from app.schemas import (
    PredictRequest,
    PredictResponse,
    PredictBatchRequest,
    PredictBatchResponse,
    PredictBatchItem,
)
from app.storage import SqliteStore


def get_artifacts_dir() -> Path:
    """Permet de surcharger le dossier d'artifacts via env var (utile pour tests/CI)."""
    p = os.getenv("P8_ARTIFACTS_DIR")
    if p:
        return Path(p).expanduser().resolve()
    return (Path(__file__).parent / "artifacts").resolve()


def get_db_path() -> Path:
    """Permet de surcharger le chemin DB via env var (utile pour tests/CI)."""
    p = os.getenv("P8_DB_PATH")
    if p:
        return Path(p).expanduser().resolve()

    try:
        from project_paths import DATA_DIR  # type: ignore
        return (DATA_DIR / "prod" / "predictions.sqlite").resolve()
    except Exception:
        return (Path("data") / "prod" / "predictions.sqlite").resolve()


def _sigmoid(x: float) -> float:
    # stable enough for our use (scores not extreme usually)
    return 1.0 / (1.0 + math.exp(-x))


def _sigmoid_vec(x: np.ndarray) -> np.ndarray:
    # vectorized sigmoid
    return 1.0 / (1.0 + np.exp(-x))


def _strict_input_enabled() -> bool:
    return os.getenv("P8_STRICT_INPUT", "1") == "1"


@asynccontextmanager
async def lifespan(app: FastAPI):
    artifacts_dir = get_artifacts_dir()
    db_path = get_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    loaded = load_model(artifacts_dir)

    store = SqliteStore(db_path)
    store.init()

    # stockage dans app.state (pas de globals)
    app.state.loaded = loaded
    app.state.store = store

    # Logs utiles (tu peux les retirer plus tard)
    print("✅ Loaded model. n_features =", len(loaded.feature_names))
    print("✅ First 10 features:", loaded.feature_names[:10])
    print("✅ ARTIFACTS_DIR =", str(artifacts_dir))
    print("✅ DB_PATH       =", str(db_path))

    yield

    # pas de connexion persistante à fermer (sqlite3.connect dans log_prediction),
    # mais on peut nettoyer l'état
    app.state.loaded = None
    app.state.store = None


app = FastAPI(title="Credit Scoring API", version="0.1.0", lifespan=lifespan)


def _get_loaded() -> LoadedModel:
    _ensure_state()
    loaded = getattr(app.state, "loaded", None)
    if loaded is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return loaded


def _get_store() -> SqliteStore:
    _ensure_state()
    store = getattr(app.state, "store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="Store not initialized")
    return store


@app.get("/health")
def health() -> dict[str, Any]:
    loaded = getattr(app.state, "loaded", None)
    ok = loaded is not None
    return {
        "status": "ok" if ok else "not_ready",
        "model_version": getattr(loaded, "model_version", None) if loaded else None,
        "n_features": len(loaded.feature_names) if loaded else None,
    }



@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    loaded = _get_loaded()
    store = _get_store()

    t0 = time.perf_counter()
    features = req.features

    expected = loaded.feature_names
    strict_input = _strict_input_enabled()

    missing = [c for c in expected if c not in features]
    if strict_input and missing:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "missing_features",
                "missing_count": len(missing),
                "missing_first_20": missing[:20],
            },
        )

    # alignement strict sur colonnes attendues
    row = {c: features.get(c, None) for c in expected}
    X = pd.DataFrame([row], columns=expected)

    pipe = loaded.pipeline
    if hasattr(pipe, "predict_proba"):
        proba = float(pipe.predict_proba(X)[:, 1][0])
    else:
        score = float(pipe.decision_function(X)[0])
        proba = float(_sigmoid(score))

    thr = float(loaded.threshold)
    decision = int(proba >= thr)

    latency_ms = (time.perf_counter() - t0) * 1000.0
    request_id = str(uuid4())

    store.log_prediction(
        {
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "request_id": request_id,
            "model_version": loaded.model_version,
            "proba_default": proba,
            "threshold": thr,
            "decision": decision,
            "latency_ms": latency_ms,
            "features": features,
        }
    )

    return PredictResponse(
        request_id=request_id,
        proba_default=proba,
        threshold=thr,
        decision=decision,
        model_version=loaded.model_version,
        latency_ms=latency_ms,
    )


@app.post("/predict_batch", response_model=PredictBatchResponse)
def predict_batch(req: PredictBatchRequest) -> PredictBatchResponse:
    loaded = _get_loaded()
    store = _get_store()

    expected = loaded.feature_names
    pipe = loaded.pipeline
    thr = float(loaded.threshold)

    t0 = time.perf_counter()
    strict_input = _strict_input_enabled()

    # validation (optionnelle) : toutes les features doivent être présentes
    if strict_input:
        for idx, r in enumerate(req.rows):
            missing = [c for c in expected if c not in r]
            if missing:
                raise HTTPException(
                    status_code=422,
                    detail={
                        "error": "missing_features",
                        "row_index": idx,
                        "missing_count": len(missing),
                        "missing_first_20": missing[:20],
                    },
                )

    # DataFrame avec colonnes attendues (ignore extras)
    aligned_rows = [{c: r.get(c, None) for c in expected} for r in req.rows]
    X = pd.DataFrame(aligned_rows, columns=expected)

    # prédiction proba
    if hasattr(pipe, "predict_proba"):
        probas = pipe.predict_proba(X)[:, 1].astype(float)
    else:
        scores = pipe.decision_function(X)
        probas = _sigmoid_vec(np.asarray(scores, dtype=float))

    decisions = (probas >= thr).astype(int)

    latency_ms_total = (time.perf_counter() - t0) * 1000.0
    n = len(req.rows)
    latency_ms_per_row = latency_ms_total / max(n, 1)

    items: list[PredictBatchItem] = []
    now_iso = datetime.now(timezone.utc).isoformat()

    log_rows: list[dict[str, Any]] = []

    for i in range(n):
        request_id = str(uuid4())
        proba = float(probas[i])
        decision = int(decisions[i])

        items.append(
            PredictBatchItem(
                request_id=request_id,
                proba_default=proba,
                threshold=thr,
                decision=decision,
                model_version=loaded.model_version,
                latency_ms=latency_ms_per_row,
            )
        )

        log_rows.append(
            {
                "ts_utc": now_iso,
                "request_id": request_id,
                "model_version": loaded.model_version,
                "proba_default": proba,
                "threshold": thr,
                "decision": decision,
                "latency_ms": latency_ms_per_row,
                "features": req.rows[i],  # on log l'input brut reçu
            }
        )

    store.log_predictions_many(log_rows)

    return PredictBatchResponse(n_rows=n, items=items)


def _ensure_state() -> None:
    if getattr(app.state, "loaded", None) is not None and getattr(app.state, "store", None) is not None:
        return

    artifacts_dir = get_artifacts_dir()
    db_path = get_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    loaded = load_model(artifacts_dir)
    store = SqliteStore(db_path)
    store.init()

    app.state.loaded = loaded
    app.state.store = store
