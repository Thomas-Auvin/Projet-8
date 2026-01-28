from __future__ import annotations
import argparse
import json
import time
import statistics
from pathlib import Path
import pandas as pd
import httpx


def p95(xs):
    xs = sorted(xs)
    if not xs:
        return 0.0
    k = int(0.95 * (len(xs)-1))
    return xs[k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=200)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    df = pd.read_csv("data/reference/reference_sample.csv").drop(columns=["TARGET"], errors="ignore")
    row = df.iloc[0].to_dict()
    rows = [row for _ in range(args.batch_size)]

    client = httpx.Client(timeout=60.0)

    # /predict
    t = []
    for _ in range(args.n):
        t0 = time.perf_counter()
        r = client.post(f"{args.base_url}/predict", json={"features": row})
        r.raise_for_status()
        t.append((time.perf_counter()-t0)*1000)

    # /predict_batch
    tb = []
    for _ in range(args.n):
        t0 = time.perf_counter()
        r = client.post(f"{args.base_url}/predict_batch", json={"rows": rows})
        r.raise_for_status()
        tb.append((time.perf_counter()-t0)*1000)

    report = {
        "base_url": args.base_url,
        "n": args.n,
        "batch_size": args.batch_size,
        "predict_ms": {"mean": statistics.mean(t), "p95": p95(t)},
        "batch_total_ms": {"mean": statistics.mean(tb), "p95": p95(tb)},
        "batch_per_row_ms": {"mean": statistics.mean(tb)/args.batch_size, "p95": p95(tb)/args.batch_size},
    }

    out = Path(args.out) if args.out else Path("outputs/perf/bench_latest.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
