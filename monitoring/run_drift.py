# monitoring/run_drift.py
from __future__ import annotations

import argparse
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

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


def load_prod_features(
    db_path: Path, limit: int | None = 2000
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Lit la table predictions et reconstruit un DataFrame depuis input_json.
    Par défaut on prend les 'limit' dernières lignes.
    Retourne (df, meta).
    """
    if not db_path.exists():
        raise FileNotFoundError(f"DB not found: {db_path}")

    query = "SELECT id, ts_utc, input_json FROM predictions ORDER BY id DESC"
    params: tuple[Any, ...] = ()
    if limit is not None:
        query += " LIMIT ?"
        params = (int(limit),)

    rows: List[Dict[str, Any]] = []
    meta: Dict[str, Any] = {
        "parse_errors": 0,
        "read_rows": 0,
        "ts_min": None,
        "ts_max": None,
    }

    with sqlite3.connect(db_path) as conn:
        cur = conn.execute(query, params)
        fetched = cur.fetchall()
        meta["read_rows"] = len(fetched)

        for _id, ts_utc, input_json in fetched:
            if isinstance(ts_utc, str):
                meta["ts_min"] = (
                    ts_utc if meta["ts_min"] is None else min(meta["ts_min"], ts_utc)
                )
                meta["ts_max"] = (
                    ts_utc if meta["ts_max"] is None else max(meta["ts_max"], ts_utc)
                )

            try:
                d = json.loads(input_json)
                if isinstance(d, dict):
                    rows.append(d)
                else:
                    meta["parse_errors"] += 1
            except (json.JSONDecodeError, TypeError):
                meta["parse_errors"] += 1

    if not rows:
        return pd.DataFrame(), meta

    return pd.DataFrame(rows), meta


def _features_df_from_report(rep: dict[str, Any]) -> pd.DataFrame:
    obj = (
        rep.get("all_features")
        or rep.get("features")
        or rep.get("per_feature")
        or rep.get("feature_stats")
    )

    if isinstance(obj, dict):
        rows: list[dict[str, Any]] = []
        for feat, stats in obj.items():
            if isinstance(stats, dict):
                rows.append({"feature": feat, **stats})
        df = pd.DataFrame(rows)
    elif isinstance(obj, list):
        df = pd.DataFrame(obj)
    else:
        df = pd.DataFrame()

    rename_map = {
        "missing_delta": "missing_rate_delta",
        "delta_missing": "missing_rate_delta",
        "ref_missing": "ref_missing_rate",
        "prod_missing": "prod_missing_rate",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    return df


def build_alerts(
    rep: dict[str, Any],
    *,
    psi_warn: float,
    psi_crit: float,
    missing_delta_warn: float,
    min_prod_rows: int,
) -> dict[str, Any]:
    """
    Construit une liste d'alertes à partir du report de compute_drift.
    On reste robuste aux variations de schéma (table per-feature en dict/list).
    """
    items: list[dict[str, Any]] = []
    status = "ok"

    n_prod = int(rep.get("n_prod_rows", rep.get("prod_rows", 0)) or 0)
    if n_prod < int(min_prod_rows):
        items.append(
            {
                "severity": "warn",
                "kind": "insufficient_prod_rows",
                "feature": None,
                "value": n_prod,
                "threshold": int(min_prod_rows),
                "message": f"Peu de lignes prod analysées ({n_prod}) → drift moins fiable.",
            }
        )

    df = _features_df_from_report(rep)
    if not df.empty:
        # PSI
        if "psi" in df.columns:
            df_psi = df.dropna(subset=["psi"]).copy()
            for _, r in df_psi[df_psi["psi"] >= psi_warn].iterrows():
                sev = "critical" if float(r["psi"]) >= psi_crit else "warn"
                items.append(
                    {
                        "severity": sev,
                        "kind": "psi",
                        "feature": r.get("feature"),
                        "value": float(r["psi"]),
                        "threshold": psi_crit if sev == "critical" else psi_warn,
                        "message": f"PSI élevé sur {r.get('feature')}: {float(r['psi']):.3f}",
                    }
                )

        # Missing rate delta
        if "missing_rate_delta" in df.columns:
            df_m = df.dropna(subset=["missing_rate_delta"]).copy()
            df_m["abs_delta"] = df_m["missing_rate_delta"].abs()
            for _, r in df_m[df_m["abs_delta"] >= missing_delta_warn].iterrows():
                items.append(
                    {
                        "severity": "warn",
                        "kind": "missing_rate_delta",
                        "feature": r.get("feature"),
                        "value": float(r["missing_rate_delta"]),
                        "threshold": float(missing_delta_warn),
                        "message": (
                            f"Delta missing élevé sur {r.get('feature')}: "
                            f"{float(r['missing_rate_delta']):+.3f}"
                        ),
                    }
                )

    n_crit = sum(1 for it in items if it["severity"] == "critical")
    n_warn = sum(1 for it in items if it["severity"] == "warn")

    if n_crit > 0:
        status = "critical"
    elif n_warn > 0:
        status = "warn"

    return {"status": status, "n_warn": n_warn, "n_critical": n_crit, "items": items}


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
        "--limit", type=int, default=2000, help="Dernières N lignes prod."
    )
    parser.add_argument(
        "--bins", type=int, default=10, help="Nombre de bins quantiles (PSI num)."
    )
    parser.add_argument(
        "--cat-top-k", type=int, default=30, help="Top-K catégories, reste -> OTHER."
    )
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="Chemin JSON sortie (défaut outputs/monitoring/drift_report.json).",
    )

    # Alerting
    parser.add_argument(
        "--psi-warn", type=float, default=0.10, help="Seuil PSI warning."
    )
    parser.add_argument(
        "--psi-crit", type=float, default=0.25, help="Seuil PSI critique."
    )
    parser.add_argument(
        "--missing-delta-warn",
        type=float,
        default=0.05,
        help="Seuil abs(delta missing) warning.",
    )
    parser.add_argument(
        "--min-prod-rows", type=int, default=200, help="Alerte si prod rows < ce seuil."
    )
    parser.add_argument(
        "--fail-on-alert",
        action="store_true",
        help="Retourne un code non-zéro si status warn/critical (utile en CI).",
    )

    args = parser.parse_args()

    db_path = Path(args.db_path).expanduser().resolve()
    ref_csv = Path(args.ref_csv).expanduser().resolve()

    df_ref = pd.read_csv(ref_csv)
    df_prod, meta = load_prod_features(db_path, limit=args.limit)

    out_dir = get_output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = (
        Path(args.out).expanduser().resolve()
        if args.out
        else (out_dir / "drift_report.json")
    )

    report = compute_drift(
        df_ref=df_ref, df_prod=df_prod, bins=args.bins, cat_top_k=args.cat_top_k
    )
    report["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    report["db_path"] = str(db_path)
    report["ref_csv"] = str(ref_csv)
    report["prod_limit"] = int(args.limit)
    report["prod_meta"] = meta

    report["alert_thresholds"] = {
        "psi_warn": float(args.psi_warn),
        "psi_crit": float(args.psi_crit),
        "missing_delta_warn": float(args.missing_delta_warn),
        "min_prod_rows": int(args.min_prod_rows),
    }
    report["alerts"] = build_alerts(
        report,
        psi_warn=float(args.psi_warn),
        psi_crit=float(args.psi_crit),
        missing_delta_warn=float(args.missing_delta_warn),
        min_prod_rows=int(args.min_prod_rows),
    )

    out_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"✅ Drift report written to: {out_path}")

    if args.fail_on_alert and report["alerts"]["status"] != "ok":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
