from __future__ import annotations

import math
import os
import time
import io
import numpy as np
import pandas as pd
import json

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from typing import Any

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import RedirectResponse,FileResponse
from fastapi.staticfiles import StaticFiles

from app.model_loader import LoadedModel, load_model
from app.schemas import (
    PredictRequest,
    PredictResponse,
    PredictBatchRequest,
    PredictBatchResponse,
    PredictBatchItem,
)
from app.storage import SqliteStore
from app.input_adapter import InputAdapter, InputError


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


def _json_safe_features(d: dict[str, Any]) -> dict[str, Any]:
    """Convertit NaN/Inf en None pour pouvoir sérialiser en JSON strict."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, (float, np.floating)):
            fv = float(v)
            if math.isnan(fv) or math.isinf(fv):
                out[k] = None
            else:
                out[k] = fv
        else:
            out[k] = v
    return out


def _get_requested_config(loaded: LoadedModel) -> tuple[list[str], float]:
    """
    Récupère la config côté meta.
    - requested_groups: liste des clés "demandées" (groupes OHE ou features simples)
    - min_filled_rate: taux minimal à remplir parmi cette liste (0.0 => pas de règle)
    """
    meta = loaded.meta or {}

    requested = meta.get("requested_groups")
    if not requested:
        # compat si tu as déjà écrit "required_groups" auparavant
        requested = meta.get("required_groups", [])

    if not isinstance(requested, list):
        requested = []

    min_rate = meta.get("min_filled_rate_requested")
    if min_rate is None:
        min_rate = meta.get("min_filled_rate_groups", 0.0)

    try:
        min_rate_f = float(min_rate)
    except Exception:
        min_rate_f = 0.0

    # clamp
    min_rate_f = max(0.0, min(1.0, min_rate_f))
    return [str(x) for x in requested], min_rate_f


def _is_group_filled(group: str, aligned: dict[str, Any], adapter: InputAdapter) -> bool:
    """
    Une "clé demandée" est remplie si :
    - c'est un groupe OHE (ex NAME_INCOME_TYPE) : au moins 1 dummy non-NaN
    - sinon, c'est une feature simple : valeur non-NaN
    """
    if group in adapter.groups:
        cols = adapter.groups[group].columns
        for c in cols:
            v = aligned.get(c, float("nan"))
            if isinstance(v, (float, np.floating)) and not math.isnan(float(v)):
                return True
            if v is not None and not (isinstance(v, str) and v.strip() == ""):
                return True
        return False

    # feature simple
    v = aligned.get(group, float("nan"))
    if v is None:
        return False
    if isinstance(v, (float, np.floating)):
        return not math.isnan(float(v))
    if isinstance(v, str):
        return v.strip() != ""
    return True


@asynccontextmanager
async def lifespan(app: FastAPI):
    artifacts_dir = get_artifacts_dir()
    db_path = get_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    loaded = load_model(artifacts_dir)

    store = SqliteStore(db_path)
    store.init()

    adapter = InputAdapter.from_feature_names(loaded.feature_names)
    app.state.adapter = adapter
    print("✅ Compact input groups:", list(adapter.groups.keys())[:10])

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
    app.state.adapter = None


app = FastAPI(title="Credit Scoring API", version="0.1.0", lifespan=lifespan)

EXAMPLES_DIR = (Path(__file__).resolve().parent / "examples").resolve()  # app/examples
if not EXAMPLES_DIR.is_dir():
    EXAMPLES_DIR = (Path(__file__).resolve().parent.parent / "examples").resolve()  # repo/examples

print(f"✅ EXAMPLES_DIR = {EXAMPLES_DIR} (exists={EXAMPLES_DIR.is_dir()})")


@app.get("/examples")
def list_examples() -> dict[str, Any]:
    if not EXAMPLES_DIR.is_dir():
        return {"files": [], "error": f"Examples dir not found: {EXAMPLES_DIR}"}
    files = sorted([p.name for p in EXAMPLES_DIR.glob("*") if p.is_file()])
    return {"files": files}


@app.get("/examples/example_input_compact.csv")
def example_csv():
    path = EXAMPLES_DIR / "example_input_compact.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Example CSV not found: {path}")
    return FileResponse(path, media_type="text/csv", filename=path.name)


@app.get("/examples/example_input_compact.json")
def example_json() -> dict[str, Any]:
    path = EXAMPLES_DIR / "example_input_compact.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Example JSON not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _get_adapter() -> InputAdapter:
    _ensure_state()
    adapter = getattr(app.state, "adapter", None)
    if adapter is None:
        raise HTTPException(status_code=503, detail="Adapter not initialized")
    return adapter


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


@app.get("/model")
def model_info() -> dict[str, Any]:
    loaded = _get_loaded()
    # renvoie les meta + infos utiles
    return {
        "model_version": loaded.model_version,
        "threshold": loaded.threshold,
        "n_features": len(loaded.feature_names),
        "meta": loaded.meta,
    }


@app.get("/features")
def features() -> dict[str, Any]:
    loaded = _get_loaded()
    adapter = _get_adapter()
    return {
        "n_features": len(loaded.feature_names),
        "feature_names": loaded.feature_names,  # oui c'est long, mais c'est la source de vérité
        "allowed_input_keys": sorted(list(adapter.allowed_input_keys())),
        "onehot_groups": {
            gname: {
                "n_dummies": len(g.columns),
                "examples": sorted({k for k in g.value_to_column.keys() if "_" not in k})[:15],
            }
            for gname, g in adapter.groups.items()
        },
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    loaded = _get_loaded()
    store = _get_store()
    adapter = _get_adapter()

    t0 = time.perf_counter()
    strict_input = _strict_input_enabled()

    user_features = req.features

    # 1) Convertit l'input "compact" (ex: NAME_INCOME_TYPE="Working")
    #    en features attendues par le modèle (dummies + numériques), le reste = NaN
    try:
        aligned, _stats = adapter.to_aligned_features(
            user_features,
            forbid_unknown_keys=strict_input,  # strict => interdit clés inconnues
        )
    except InputError as e:
        raise HTTPException(status_code=422, detail={"error": "invalid_input", "message": str(e)})
    
    if strict_input:
        requested, min_rate = _get_requested_config(loaded)
        if requested and min_rate > 0.0:
            filled = [g for g in requested if _is_group_filled(g, aligned, adapter)]
            min_count = math.ceil(min_rate * len(requested))

            if len(filled) < min_count:
                missing = [g for g in requested if g not in filled]
                raise HTTPException(
                    status_code=422,
                    detail={
                        "error": "insufficient_filled_requested",
                        "requested_count": len(requested),
                        "filled_count": len(filled),
                        "min_filled_rate": min_rate,
                        "min_filled_count": min_count,
                        "missing_first_20": missing[:20],
                        "hint": "Utilise GET /features pour voir les clés et groupes possibles.",
                    },
                )
    
    expected = loaded.feature_names
    X = pd.DataFrame([aligned], columns=expected)

    # 2) Prédiction
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

    # 3) Log en DB : on log l'aligné (stable pour drift)
    store.log_prediction(
        {
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "request_id": request_id,
            "model_version": loaded.model_version,
            "proba_default": proba,
            "threshold": thr,
            "decision": decision,
            "latency_ms": latency_ms,
            "features": _json_safe_features(aligned),
            # Optionnel si tu veux garder aussi le brut (uniquement si ta DB le supporte)
            # "features_raw": user_features,
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
    adapter = _get_adapter()

    expected = loaded.feature_names
    pipe = loaded.pipeline
    thr = float(loaded.threshold)

    t0 = time.perf_counter()
    strict_input = _strict_input_enabled()

    # 1) Adapter chaque ligne vers le format attendu
    requested, min_rate = _get_requested_config(loaded)
    aligned_rows: list[dict[str, Any]] = []
    for idx, r in enumerate(req.rows):
        try:
            aligned, _stats = adapter.to_aligned_features(
                r,
                forbid_unknown_keys=strict_input,
            )
        except InputError as e:
            raise HTTPException(
                status_code=422,
                detail={"error": "invalid_input", "row_index": idx, "message": str(e)},
            )

        # Validation strict partiel (par ligne)
        if strict_input and requested and min_rate > 0.0:
            filled = [g for g in requested if _is_group_filled(g, aligned, adapter)]
            min_count = math.ceil(min_rate * len(requested))
            if len(filled) < min_count:
                missing = [g for g in requested if g not in filled]
                raise HTTPException(
                    status_code=422,
                    detail={
                        "error": "insufficient_filled_requested",
                        "row_index": idx,
                        "requested_count": len(requested),
                        "filled_count": len(filled),
                        "min_filled_rate": min_rate,
                        "min_filled_count": min_count,
                        "missing_first_20": missing[:20],
                    },
                )

        aligned_rows.append(aligned)

    X = pd.DataFrame(aligned_rows, columns=expected)

    # 2) Prédiction
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
                "features": _json_safe_features(aligned_rows[i]),  # aligné (stable drift)
                # Optionnel si DB supporte:
                # "features_raw": req.rows[i],
            }
        )

    store.log_predictions_many(log_rows)

    return PredictBatchResponse(n_rows=n, items=items)


@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...)) -> dict[str, Any]:
    loaded = _get_loaded()
    store = _get_store()
    adapter = _get_adapter()

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=415, detail="Merci d'envoyer un fichier .csv")

    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"CSV illisible: {e}")

    if df.shape[0] == 0:
        raise HTTPException(status_code=400, detail="CSV vide")

    rows = df.to_dict(orient="records")

    expected = loaded.feature_names
    pipe = loaded.pipeline
    thr = float(loaded.threshold)

    t0 = time.perf_counter()
    strict_input = _strict_input_enabled()

    requested, min_rate = _get_requested_config(loaded)  # calculé 1 fois

    aligned_rows: list[dict[str, Any]] = []
    for idx, r in enumerate(rows):
        try:
            aligned, _stats = adapter.to_aligned_features(
                r,
                forbid_unknown_keys=strict_input,
            )
        except InputError as e:
            raise HTTPException(
                status_code=422,
                detail={"error": "invalid_input", "row_index": idx, "message": str(e)},
            )

        # Validation strict partiel (par ligne)
        if strict_input and requested and min_rate > 0.0:
            filled = [g for g in requested if _is_group_filled(g, aligned, adapter)]
            min_count = math.ceil(min_rate * len(requested))
            if len(filled) < min_count:
                missing = [g for g in requested if g not in filled]
                raise HTTPException(
                    status_code=422,
                    detail={
                        "error": "insufficient_filled_requested",
                        "row_index": idx,
                        "requested_count": len(requested),
                        "filled_count": len(filled),
                        "min_filled_rate": min_rate,
                        "min_filled_count": min_count,
                        "missing_first_20": missing[:20],
                    },
                )

        aligned_rows.append(aligned)

    X = pd.DataFrame(aligned_rows, columns=expected)

    if hasattr(pipe, "predict_proba"):
        probas = pipe.predict_proba(X)[:, 1].astype(float)
    else:
        scores = pipe.decision_function(X)
        probas = _sigmoid_vec(np.asarray(scores, dtype=float))

    decisions = (probas >= thr).astype(int)

    latency_ms_total = (time.perf_counter() - t0) * 1000.0
    n = len(rows)
    latency_ms_per_row = latency_ms_total / max(n, 1)

    now_iso = datetime.now(timezone.utc).isoformat()
    items = []
    log_rows = []

    for i in range(n):
        request_id = str(uuid4())
        proba = float(probas[i])
        decision = int(decisions[i])

        items.append(
            {
                "request_id": request_id,
                "proba_default": proba,
                "threshold": thr,
                "decision": decision,
                "model_version": loaded.model_version,
                "latency_ms": float(latency_ms_per_row),
            }
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
                "features": _json_safe_features(aligned_rows[i]),  # aligné
            }
        )

    store.log_predictions_many(log_rows)

    return {"n_rows": n, "items": items}


def _ensure_state() -> None:
    if (
        getattr(app.state, "loaded", None) is not None
        and getattr(app.state, "store", None) is not None
        and getattr(app.state, "adapter", None) is not None
    ):
        return

    artifacts_dir = get_artifacts_dir()
    db_path = get_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    loaded = load_model(artifacts_dir)
    adapter = InputAdapter.from_feature_names(loaded.feature_names)

    store = SqliteStore(db_path)
    store.init()

    app.state.loaded = loaded
    app.state.store = store
    app.state.adapter = adapter
