from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


EPS = 1e-6


def _json_safe(obj: Any) -> Any:
    """Convertit NaN/Inf en None pour garantir un JSON sérialisable."""
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


@dataclass
class SqliteStore:
    db_path: Path

    def init(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
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
            conn.execute("CREATE INDEX IF NOT EXISTS idx_predictions_ts ON predictions(ts_utc)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_predictions_model ON predictions(model_version)")
            conn.commit()

    def log_prediction(self, row: Dict[str, Any]) -> None:
        # schéma latency_ms NOT NULL -> on force un float
        latency_val = float(row.get("latency_ms") or 0.0)
        features = _json_safe(row["features"])

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO predictions (
                    ts_utc, request_id, model_version, proba_default, threshold,
                    decision, latency_ms, input_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["ts_utc"],
                    row["request_id"],
                    row["model_version"],
                    float(row["proba_default"]),
                    float(row["threshold"]),
                    int(row["decision"]),
                    latency_val,
                    json.dumps(features, ensure_ascii=False),
                ),
            )
            conn.commit()

    def log_predictions_many(self, rows: List[Dict[str, Any]]) -> None:
        """Insertion batch (perf) : 1 transaction + executemany."""
        values = []
        for row in rows:
            latency_val = float(row.get("latency_ms") or 0.0)
            features = _json_safe(row["features"])
            values.append(
                (
                    row["ts_utc"],
                    row["request_id"],
                    row["model_version"],
                    float(row["proba_default"]),
                    float(row["threshold"]),
                    int(row["decision"]),
                    latency_val,
                    json.dumps(features, ensure_ascii=False),
                )
            )

        with sqlite3.connect(self.db_path) as conn:
            conn.executemany(
                """
                INSERT INTO predictions (
                    ts_utc, request_id, model_version, proba_default, threshold,
                    decision, latency_ms, input_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                values,
            )
            conn.commit()
