import json
import secrets
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
    result TEXT NOT NULL,
    owner TEXT,
    share_token TEXT
)
"""
# Columns added after the first release, for databases created before them
_ADDED_COLUMNS = {"owner": "TEXT", "share_token": "TEXT"}
_SUMMARY_COLUMNS = "id, created_at, resume_filename, job_title, match_score, ats_score"


class HistoryStore:
    """Past analyses in a SQLite file.

    Only the analysis result is stored, never the uploaded file. A connection is opened
    per call, because requests are served from several threads.

    Each analysis records its `owner` (the browser session that created it), so that a
    public deployment can show visitors only their own history, and a random
    `share_token` that lets anyone holding the link open that one analysis.
    """

    def __init__(self, path: str):
        self._path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as connection, connection:
            connection.execute(_SCHEMA)
            existing = {row["name"] for row in connection.execute("PRAGMA table_info(analyses)")}
            for column, kind in _ADDED_COLUMNS.items():
                if column not in existing:
                    connection.execute(f"ALTER TABLE analyses ADD COLUMN {column} {kind}")
            # Analyses saved before share links existed get a token now
            connection.execute(
                "UPDATE analyses SET share_token = lower(hex(randomblob(16))) "
                "WHERE share_token IS NULL"
            )
            connection.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS analyses_share_token ON analyses (share_token)"
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._path)
        connection.row_factory = sqlite3.Row
        return connection

    def save(self, resume_filename: str, job_title: str, result: dict, owner: str | None) -> dict:
        """Stores a result and returns it with its id, share token and metadata added."""
        created_at = datetime.now(UTC).isoformat(timespec="seconds")
        share_token = secrets.token_urlsafe(16)
        with closing(self._connect()) as connection, connection:
            cursor = connection.execute(
                "INSERT INTO analyses (created_at, resume_filename, job_title, match_score, "
                "ats_score, result, owner, share_token) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    created_at,
                    resume_filename,
                    job_title,
                    result["match_score"],
                    result["ats"]["score"],
                    json.dumps(result),
                    owner,
                    share_token,
                ),
            )
        return {
            "id": cursor.lastrowid,
            "created_at": created_at,
            "resume_filename": resume_filename,
            "job_title": job_title,
            "share_token": share_token,
            **result,
        }

    @staticmethod
    def _owned(owner: str | None) -> tuple[str, tuple]:
        """A WHERE clause limiting rows to one owner; None means every row."""
        return ("owner = ?", (owner,)) if owner is not None else ("1 = 1", ())

    def list(self, limit: int = 50, owner: str | None = None) -> list[dict]:
        """Summaries of the most recent analyses, newest first."""
        condition, values = self._owned(owner)
        with closing(self._connect()) as connection:
            rows = connection.execute(
                f"SELECT {_SUMMARY_COLUMNS} FROM analyses WHERE {condition} "
                "ORDER BY id DESC LIMIT ?",
                (*values, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def _fetch(self, condition: str, values: tuple) -> dict | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                f"SELECT {_SUMMARY_COLUMNS}, share_token, result FROM analyses WHERE {condition}",
                values,
            ).fetchone()
        if row is None:
            return None
        record = dict(row)
        result = json.loads(record.pop("result"))
        del record["ats_score"]  # already inside the result
        return {**record, **result}

    def get(self, analysis_id: int, owner: str | None = None) -> dict | None:
        condition, values = self._owned(owner)
        return self._fetch(f"id = ? AND {condition}", (analysis_id, *values))

    def get_shared(self, share_token: str) -> dict | None:
        """The analysis a share link points to, whoever created it."""
        return self._fetch("share_token = ?", (share_token,))

    def delete(self, analysis_id: int, owner: str | None = None) -> bool:
        condition, values = self._owned(owner)
        with closing(self._connect()) as connection, connection:
            cursor = connection.execute(
                f"DELETE FROM analyses WHERE id = ? AND {condition}", (analysis_id, *values)
            )
        return cursor.rowcount > 0
