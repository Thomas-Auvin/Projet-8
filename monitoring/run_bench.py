from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
import numpy as np
import pandas as pd


# -------------------------
# Utils
# -------------------------
def _json_safe(v: Any) -> Any:
    """Convertit NaN/Inf + numpy scalars en objets JSON safe."""
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        fv = float(v)
        if np.isnan(fv) or np.isinf(fv):
            return None
        return fv
    return v


def stats_ms(samples: List[float]) -> Dict[str, float]:
    arr = np.asarray(samples, dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "p95": 0.0}
    return {
        "mean": float(arr.mean()),
        "p95": float(np.percentile(arr, 95)),
    }


def _request_json(client: httpx.Client, method: str, url: str, **kwargs: Any) -> Tuple[int, Any]:
    r = client.request(method, url, **kwargs)
    try:
        data = r.json()
    except Exception:
        data = r.text
    return r.status_code, data


def _predict_ok(client: httpx.Client, base_url: str, features: Dict[str, Any]) -> Tuple[bool, Optional[Dict[str, Any]]]:
    url = base_url.rstrip("/") + "/predict"
    r = client.post(url, json={"features": features}, timeout=60.0)
    if r.status_code == 200:
        return True, None
    try:
        return False, r.json()
    except Exception:
        return False, {"status_code": r.status_code, "text": r.text}


def _filter_allowed_keys(features: Dict[str, Any], allowed_keys: Optional[set[str]]) -> Dict[str, Any]:
    if not allowed_keys:
        return features
    return {k: v for k, v in features.items() if k in allowed_keys}


def _load_candidates_from_csv(sample_csv: Path, max_rows: int = 200) -> List[Dict[str, Any]]:
    df = pd.read_csv(sample_csv)
    if df.shape[0] == 0:
        raise ValueError(f"CSV vide: {sample_csv}")

    if "TARGET" in df.columns:
        df = df.drop(columns=["TARGET"])

    candidates: List[Dict[str, Any]] = []
    take = min(max_rows, df.shape[0])
    for i in range(take):
        row = df.iloc[i].to_dict()
        candidates.append({k: _json_safe(v) for k, v in row.items()})
    return candidates


def _try_payload_from_api_examples(client: httpx.Client, base_url: str) -> Optional[Dict[str, Any]]:
    """
    Essaie de récupérer un exemple "qui marche" depuis l'API.
    Ton API expose /examples/example_input_compact.json.
    """
    url = base_url.rstrip("/") + "/examples/example_input_compact.json"
    status, data = _request_json(client, "GET", url, timeout=30.0)
    if status != 200 or not isinstance(data, dict):
        return None

    # accepte 2 formats :
    # - {"features": {...}}
    # - {...} directement
    if "features" in data and isinstance(data["features"], dict):
        feats = data["features"]
    else:
        feats = data

    feats = {k: _json_safe(v) for k, v in feats.items()}
    ok, _err = _predict_ok(client, base_url, feats)
    return feats if ok else None


def _get_allowed_input_keys(client: httpx.Client, base_url: str) -> Optional[set[str]]:
    url = base_url.rstrip("/") + "/features"
    status, data = _request_json(client, "GET", url, timeout=30.0)
    if status != 200 or not isinstance(data, dict):
        return None
    keys = data.get("allowed_input_keys")
    if isinstance(keys, list):
        return {str(x) for x in keys}
    return None


def _find_working_features(
    client: httpx.Client,
    base_url: str,
    candidates: List[Dict[str, Any]],
    allowed_keys: Optional[set[str]],
    max_tries: int = 80,
) -> Dict[str, Any]:
    last_err: Optional[Dict[str, Any]] = None
    tries = min(max_tries, len(candidates))

    for i in range(tries):
        feats = _filter_allowed_keys(candidates[i], allowed_keys)
        ok, err = _predict_ok(client, base_url, feats)
        if ok:
            return feats
        last_err = err

    raise RuntimeError(
        "Impossible de trouver une ligne qui passe /predict (422).\n"
        f"Dernière erreur: {last_err}"
    )


# -------------------------
# Bench endpoints
# -------------------------
def bench_predict(
    client: httpx.Client,
    base_url: str,
    features: Dict[str, Any],
    n: int,
    warmup: int = 5,
) -> List[float]:
    url = base_url.rstrip("/") + "/predict"
    payload = {"features": features}

    # warmup (on vérifie status)
    for _ in range(warmup):
        r = client.post(url, json=payload, timeout=60.0)
        r.raise_for_status()

    times: List[float] = []
    for _ in range(n):
        t0 = time.perf_counter()
        r = client.post(url, json=payload, timeout=60.0)
        r.raise_for_status()
        times.append((time.perf_counter() - t0) * 1000.0)
    return times


def bench_predict_batch(
    client: httpx.Client,
    base_url: str,
    features: Dict[str, Any],
    n: int,
    batch_size: int,
    warmup: int = 2,
) -> List[float]:
    url = base_url.rstrip("/") + "/predict_batch"
    payload = {"rows": [features] * batch_size}

    for _ in range(warmup):
        r = client.post(url, json=payload, timeout=120.0)
        r.raise_for_status()

    times: List[float] = []
    for _ in range(n):
        t0 = time.perf_counter()
        r = client.post(url, json=payload, timeout=120.0)
        r.raise_for_status()
        times.append((time.perf_counter() - t0) * 1000.0)
    return times


# -------------------------
# Main
# -------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", type=str, default=os.getenv("P8_API_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--sample-csv", type=str, default="data/reference/reference_sample.csv")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--out", type=str, default="outputs/perf/bench_current.json")
    parser.add_argument("--max-csv-rows", type=int, default=200)
    args = parser.parse_args()

    base_url = args.base_url
    sample_csv = Path(args.sample_csv).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with httpx.Client() as client:
        # (optionnel) infos modèle
        model_version = None
        threshold = None
        status, model_data = _request_json(client, "GET", base_url.rstrip("/") + "/model", timeout=30.0)
        if status == 200 and isinstance(model_data, dict):
            model_version = model_data.get("model_version")
            threshold = model_data.get("threshold")

        allowed_keys = _get_allowed_input_keys(client, base_url)

        # 1) meilleur cas: on récupère un exemple via l'API
        features = _try_payload_from_api_examples(client, base_url)
        payload_source = "api_examples" if features is not None else None

        # 2) fallback: CSV + filtrage + recherche d'une ligne qui passe
        if features is None:
            candidates = _load_candidates_from_csv(sample_csv, max_rows=args.max_csv_rows)
            features = _find_working_features(client, base_url, candidates, allowed_keys)
            payload_source = f"csv:{sample_csv.name}"

        if features is None:
            raise ValueError("features is None: impossible de lancer le bench (entrée/features manquantes).")

        predict_times = bench_predict(client, base_url, features, n=args.n)
        batch_times = bench_predict_batch(client, base_url, features, n=args.n, batch_size=args.batch_size)

    pred = stats_ms(predict_times)
    batch_total = stats_ms(batch_times)
    batch_per_row = {
        "mean": batch_total["mean"] / max(int(args.batch_size), 1),
        "p95": batch_total["p95"] / max(int(args.batch_size), 1),
    }

    report: Dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_url": base_url,
        "n": int(args.n),
        "batch_size": int(args.batch_size),
        "payload_source": payload_source,
        "model_version": model_version,
        "threshold": threshold,
        "predict_ms": pred,
        "batch_total_ms": batch_total,
        "batch_per_row_ms": batch_per_row,
    }

    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ Bench written to: {out_path}")


if __name__ == "__main__":
    main()
