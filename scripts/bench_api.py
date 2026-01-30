from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, Dict

import httpx
import pandas as pd


def p95(xs: list[float]) -> float:
    xs = sorted(xs)
    if not xs:
        return 0.0
    k = int(0.95 * (len(xs) - 1))
    return float(xs[k])


def json_safe_features(d: Dict[str, Any]) -> Dict[str, Any]:
    """
    Rend un dict JSON-safe pour httpx/json (allow_nan=False) :
    - NaN -> None
    - +inf / -inf -> None
    """
    out: Dict[str, Any] = {}
    for k, v in d.items():
        # pandas/numpy NaN
        try:
            if pd.isna(v):
                out[k] = None
                continue
        except Exception:
            pass

        # float NaN/Inf
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            out[k] = None
            continue

        out[k] = v
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=200)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    df = pd.read_csv("data/reference/reference_sample.csv").drop(columns=["TARGET"], errors="ignore")

    # 1 row (JSON-safe)
    row_raw = df.iloc[0].to_dict()
    row = json_safe_features(row_raw)

    # Batch (copie du dict pour éviter un aliasing surprise)
    rows = [dict(row) for _ in range(args.batch_size)]

    with httpx.Client(timeout=60.0) as client:
        # /predict
        t: list[float] = []
        for _ in range(args.n):
            t0 = time.perf_counter()
            r = client.post(f"{args.base_url}/predict", json={"features": row})
            r.raise_for_status()
            t.append((time.perf_counter() - t0) * 1000.0)

        # /predict_batch
        tb: list[float] = []
        for _ in range(args.n):
            t0 = time.perf_counter()
            r = client.post(f"{args.base_url}/predict_batch", json={"rows": rows})
            r.raise_for_status()
            tb.append((time.perf_counter() - t0) * 1000.0)

    report = {
        "base_url": args.base_url,
        "n": int(args.n),
        "batch_size": int(args.batch_size),
        "predict_ms": {"mean": float(statistics.mean(t)), "p95": p95(t)},
        "batch_total_ms": {"mean": float(statistics.mean(tb)), "p95": p95(tb)},
        "batch_per_row_ms": {
            "mean": float(statistics.mean(tb) / max(args.batch_size, 1)),
            "p95": float(p95(tb) / max(args.batch_size, 1)),
        },
    }

    out = Path(args.out) if args.out else Path("outputs/perf/bench_latest.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
