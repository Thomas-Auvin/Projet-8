from __future__ import annotations

import argparse
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from .drift_utils import compute_drift


def get_db_path() -> Path:
    p = os.getenv("P8_DB_PATH")
    if p:
        return Path(p).expanduser().resolve()

    try:
        from project_paths import DATA_DIR  # type: ignore
        return (DATA_DIR / "prod" / "predictions.sqlite").resolve()
    except Exception:
        return (Path("data") / "prod" / "predictions.sqlite").resolve()


def get_output_dir() -> Path:
    try:
        from project_paths import OUT_DIR  # type: ignore
        return (OUT_DIR / "monitoring").resolve()
    except Exception:
        return (Path("outputs") / "monitoring").resolve()


def load_prod_features(db_path: Path, limit: int | None = 2000) -> pd.DataFrame:
    """
    Lit la table predictions et reconstruit un DataFrame depuis input_json.
    Par défaut on prend les 'limit' dernières lignes (suffisant pour monitoring).
    """
    if not db_path.exists():
        raise FileNotFoundError(f"DB not found: {db_path}")

    query = "SELECT input_json FROM predictions ORDER BY id DESC"
    params: tuple[Any, ...] = ()
    if limit is not None:
        query += " LIMIT ?"
        params = (int(limit),)

    rows: List[Dict[str, Any]] = []
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute(query, params)
        for (input_json,) in cur.fetchall():
            try:
                d = json.loads(input_json)
                if isinstance(d, dict):
                    rows.append(d)
            except Exception:
                continue

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--db-path",
        type=str,
        default=str(get_db_path()),
        help="Path vers predictions.sqlite (default: env P8_DB_PATH ou data/prod/predictions.sqlite)",
    )
    parser.add_argument(
        "--ref-csv",
        type=str,
        default="data/reference/reference_sample.csv",
        help="CSV de référence",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=2000,
        help="Nombre de lignes prod à analyser (dernières N).",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=10,
        help="Nombre de bins quantiles (numériques) pour PSI.",
    )
    parser.add_argument(
        "--cat-top-k",
        type=int,
        default=30,
        help="Top-K catégories gardées (catégorielles), le reste -> OTHER.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="Chemin du JSON de sortie. Par défaut: outputs/monitoring/drift_report.json",
    )

    args = parser.parse_args()

    db_path = Path(args.db_path).expanduser().resolve()
    ref_csv = Path(args.ref_csv).expanduser().resolve()

    df_ref = pd.read_csv(ref_csv)
    df_prod = load_prod_features(db_path, limit=args.limit)

    out_dir = get_output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out).expanduser().resolve() if args.out else (out_dir / "drift_report.json")

    report = compute_drift(df_ref=df_ref, df_prod=df_prod, bins=args.bins, cat_top_k=args.cat_top_k)
    report["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    report["db_path"] = str(db_path)
    report["ref_csv"] = str(ref_csv)
    report["prod_limit"] = int(args.limit)

    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ Drift report written to: {out_path}")


if __name__ == "__main__":
    main()
