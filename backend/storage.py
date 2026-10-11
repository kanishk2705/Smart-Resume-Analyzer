import json
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS analyses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    resume_filename TEXT NOT NULL,
    job_title TEXT NOT NULL,
    match_score REAL NOT NULL,
    ats_score REAL NOT NULL,
    result TEXT NOT NULL
)
"""
_SUMMARY_COLUMNS = "id, created_at, resume_filename, job_title, match_score, ats_score"


class HistoryStore:
    """Past analyses in a SQLite file.

    Only the analysis result is stored, never the uploaded file. A connection is opened
    per call, because requests are served from several threads.
    """

    def __init__(self, path: str):
        self._path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as connection, connection:
            connection.execute(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._path)
        connection.row_factory = sqlite3.Row
        return connection

    def save(self, resume_filename: str, job_title: str, result: dict) -> dict:
        """Stores a result and returns it with its id and metadata added."""
        created_at = datetime.now(UTC).isoformat(timespec="seconds")
        with closing(self._connect()) as connection, connection:
            cursor = connection.execute(
                "INSERT INTO analyses (created_at, resume_filename, job_title, match_score, "
                "ats_score, result) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    created_at,
                    resume_filename,
                    job_title,
                    result["match_score"],
                    result["ats"]["score"],
                    json.dumps(result),
                ),
            )
        return {
            "id": cursor.lastrowid,
            "created_at": created_at,
            "resume_filename": resume_filename,
            "job_title": job_title,
            **result,
        }

    def list(self, limit: int = 50) -> list[dict]:
        """Summaries of the most recent analyses, newest first."""
        with closing(self._connect()) as connection:
            rows = connection.execute(
                f"SELECT {_SUMMARY_COLUMNS} FROM analyses ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(row) for row in rows]

    def get(self, analysis_id: int) -> dict | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                f"SELECT {_SUMMARY_COLUMNS}, result FROM analyses WHERE id = ?",
                (analysis_id,),
            ).fetchone()
        if row is None:
            return None
        record = dict(row)
        result = json.loads(record.pop("result"))
        del record["ats_score"]  # already inside the result
        return {**record, **result}

    def delete(self, analysis_id: int) -> bool:
        with closing(self._connect()) as connection, connection:
            cursor = connection.execute("DELETE FROM analyses WHERE id = ?", (analysis_id,))
        return cursor.rowcount > 0
