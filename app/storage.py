from __future__ import annotations

import json
import math
import sqlite3
import queue
import threading
import time

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List


def _json_safe(obj: Any) -> Any:
    """Rend l'objet JSON-sérialisable :
    - scalaires numpy -> types Python (via .item())
    - NaN/Inf -> None
    - dict/list -> récursion
    """
    # numpy scalar (np.bool_, np.int64, np.float64, etc.) -> python scalar
    if hasattr(obj, "item") and type(obj).__module__.startswith("numpy"):
        try:
            return _json_safe(obj.item())
        except Exception:
            # si jamais .item() plante, on retombe sur les règles suivantes
            pass

    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_safe(v) for v in obj]
    return obj


@dataclass
class SqliteStore:
    db_path: Path
    timeout_s: float = 5.0

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=self.timeout_s)
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def init(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
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
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_predictions_ts ON predictions(ts_utc)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_predictions_model ON predictions(model_version)"
            )
            conn.commit()

    def log_prediction(self, row: Dict[str, Any]) -> None:
        latency_val = float(row.get("latency_ms") or 0.0)
        features = _json_safe(row["features"])

        with self._connect() as conn:
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
                    json.dumps(features, ensure_ascii=False, separators=(",", ":")),
                ),
            )
            conn.commit()

    def log_predictions_many(self, rows: List[Dict[str, Any]]) -> None:
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

        with self._connect() as conn:
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

    _writer: "AsyncSqliteWriter | None" = field(init=False, repr=False, default=None)

    def start_async_writer(self) -> None:
        if self._writer is not None:
            return
        self._writer = AsyncSqliteWriter(self.db_path, timeout_s=self.timeout_s)
        self._writer.start()

    def stop_async_writer(self) -> None:
        if self._writer is None:
            return
        self._writer.stop()
        self._writer = None

    def enqueue_prediction(self, row: Dict[str, Any]) -> None:
        if self._writer is None:
            self.log_prediction(row)
            return
        self._writer.submit(row)

    def enqueue_predictions_many(self, rows: List[Dict[str, Any]]) -> None:
        if self._writer is None:
            self.log_predictions_many(rows)
            return
        for r in rows:
            self._writer.submit(r)


class AsyncSqliteWriter:
    def __init__(
        self,
        db_path: Path,
        timeout_s: float = 5.0,
        batch_size: int = 50,
        flush_interval_s: float = 0.2,
        queue_max: int = 10000,
    ) -> None:
        self.db_path = db_path
        self.timeout_s = timeout_s
        self.batch_size = batch_size
        self.flush_interval_s = flush_interval_s
        self.q: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=queue_max)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.dropped = 0

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5.0)

    def submit(self, row: Dict[str, Any]) -> None:
        try:
            self.q.put_nowait(row)
        except queue.Full:
            self.dropped += 1

    def _flush(self, conn: sqlite3.Connection, buf: list[Dict[str, Any]]) -> None:
        if not buf:
            return
        values = []
        for row in buf:
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
                    json.dumps(features, ensure_ascii=False, separators=(",", ":")),
                )
            )

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
        buf.clear()

    def _run(self) -> None:
        conn = sqlite3.connect(self.db_path, timeout=self.timeout_s)
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")

            buf: list[Dict[str, Any]] = []
            last_flush = time.monotonic()

            while not self._stop.is_set() or not self.q.empty():
                timeout = max(
                    0.0, self.flush_interval_s - (time.monotonic() - last_flush)
                )
                try:
                    row = self.q.get(timeout=timeout)
                    buf.append(row)
                except queue.Empty:
                    pass

                now = time.monotonic()
                if buf and (
                    len(buf) >= self.batch_size
                    or (now - last_flush) >= self.flush_interval_s
                ):
                    self._flush(conn, buf)
                    last_flush = now

            # flush final
            if buf:
                self._flush(conn, buf)

        finally:
            conn.close()
