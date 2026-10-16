"""Tiny SQLite store for NON-sensitive metadata (model run history, schema version).

Ticket text and customer data are never written to disk. If the database path is not
writable (e.g. read-only container) every function degrades to a no-op."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .config import DB_PATH

MIGRATIONS = [
    """CREATE TABLE IF NOT EXISTS model_runs (
           id INTEGER PRIMARY KEY AUTOINCREMENT, created_utc TEXT NOT NULL,
           dataset_name TEXT, n_rows INTEGER, task TEXT, model TEXT, metrics_json TEXT)""",
    """CREATE TABLE IF NOT EXISTS app_events (
           id INTEGER PRIMARY KEY AUTOINCREMENT, created_utc TEXT NOT NULL, event TEXT, detail TEXT)""",
]


@contextmanager
def _conn(path):
    con = sqlite3.connect(str(path), timeout=5)
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init_db(path: Path | str | None = None) -> bool:
    """Create the DB and apply pending migrations. Returns False if unavailable."""
    try:
        p = Path(path or DB_PATH)
        p.parent.mkdir(parents=True, exist_ok=True)
        with _conn(p) as con:
            con.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)")
            current = con.execute("SELECT MAX(version) FROM schema_version").fetchone()[0] or 0
            for i, sql in enumerate(MIGRATIONS, start=1):
                if i > current:
                    con.execute(sql)
                    con.execute("INSERT INTO schema_version (version) VALUES (?)", (i,))
        return True
    except (sqlite3.Error, OSError):
        return False


def schema_version(path: Path | str | None = None) -> int:
    try:
        with _conn(Path(path or DB_PATH)) as con:
            return con.execute("SELECT MAX(version) FROM schema_version").fetchone()[0] or 0
    except (sqlite3.Error, OSError):
        return 0


def log_model_run(dataset_name: str, n_rows: int, task: str, model: str, metrics: dict, path=None) -> None:
    try:
        if not init_db(path):
            return
        with _conn(Path(path or DB_PATH)) as con:
            con.execute("INSERT INTO model_runs (created_utc, dataset_name, n_rows, task, model, metrics_json) VALUES (?,?,?,?,?,?)",
                        (datetime.now(timezone.utc).isoformat(timespec="seconds"), dataset_name, int(n_rows), task, model,
                         json.dumps(metrics, default=float)))
    except (sqlite3.Error, OSError):
        pass


def recent_model_runs(limit: int = 20, path=None) -> list[dict]:
    try:
        if not init_db(path):
            return []
        with _conn(Path(path or DB_PATH)) as con:
            rows = con.execute("SELECT created_utc, dataset_name, n_rows, task, model, metrics_json FROM model_runs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [{"created_utc": r[0], "dataset": r[1], "n_rows": r[2], "task": r[3], "model": r[4], "metrics": json.loads(r[5] or "{}")} for r in rows]
    except (sqlite3.Error, OSError):
        return []
