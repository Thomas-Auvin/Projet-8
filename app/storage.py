from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


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
            conn.commit()

    # app/storage.py
    def log_prediction(self, row: Dict[str, Any]) -> None:
        latency = row.get("latency_ms")
        latency_val = None if latency is None else float(latency)

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
                    latency_val,  # ✅ None -> NULL en SQLite
                    json.dumps(row["features"], ensure_ascii=False),
                ),
            )
