import json
import sqlite3
from datetime import datetime, timezone

import pandas as pd


def _create_predictions_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_utc TEXT NOT NULL,
            request_id TEXT NOT NULL,
            model_version TEXT NOT NULL,
            proba_default REAL NOT NULL,
            threshold REAL NOT NULL,
            decision INTEGER NOT NULL,
            latency_ms REAL NOT NULL,
            input_json TEXT NOT NULL
        )
        """
    )
    conn.commit()


def _insert_pred(conn: sqlite3.Connection, request_id: str, features: dict) -> None:
    conn.execute(
        """
        INSERT INTO predictions (
            ts_utc, request_id, model_version, proba_default, threshold,
            decision, latency_ms, input_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now(timezone.utc).isoformat(),
            request_id,
            "test_model_v1",
            0.42,
            0.5,
            0,
            12.3,
            json.dumps(features, ensure_ascii=False),
        ),
    )
    conn.commit()


def test_monitoring_compute_drift_from_sqlite(tmp_path):
    # --- Arrange: DB temp + quelques lignes "prod"
    db_path = tmp_path / "preds.sqlite"
    with sqlite3.connect(db_path) as conn:
        _create_predictions_table(conn)
        _insert_pred(conn, "r1", {"A": 1.0, "B": "x", "C": None})
        _insert_pred(conn, "r2", {"A": 2.0, "B": "y", "C": 10.0})
        _insert_pred(conn, "r3", {"A": 3.0, "B": "x", "C": 11.0})

    # --- Arrange: ref dataframe
    df_ref = pd.DataFrame(
        [
            {"A": 1.0, "B": "x", "C": 10.0},
            {"A": 1.5, "B": "x", "C": 10.5},
            {"A": 2.0, "B": "y", "C": 11.0},
            {"A": 2.5, "B": "y", "C": 11.5},
        ]
    )

    # --- Act
    from monitoring.run_drift import load_prod_features
    from monitoring.drift_utils import compute_drift

    df_prod = load_prod_features(db_path, limit=100)
    report = compute_drift(df_ref=df_ref, df_prod=df_prod, bins=5, cat_top_k=10)

    # --- Assert: structure minimale
    assert report["n_ref_rows"] == len(df_ref)
    assert report["n_prod_rows"] == 3
    assert report["features_total"] >= 3

    assert "top_psi" in report and isinstance(report["top_psi"], list)
    assert "top_missing_delta" in report and isinstance(report["top_missing_delta"], list)
    assert "all_features" in report and isinstance(report["all_features"], list)

    # au moins une feature devrait exister
    feats = {x["feature"] for x in report["all_features"]}
    assert {"A", "B", "C"}.issubset(feats)
