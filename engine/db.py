"""SQLite meeting registry for MeetingMind."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "meetingmind.db"


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS meetings (
                meeting_id TEXT PRIMARY KEY,
                path TEXT NOT NULL,
                turn_count INTEGER NOT NULL,
                source TEXT NOT NULL CHECK(source IN ('demo', 'upload')),
                created_at TEXT NOT NULL
            )
            """
        )
        conn.commit()


def upsert_meeting(
    meeting_id: str,
    path: str,
    turn_count: int,
    source: str,
) -> None:
    init_db()
    created_at = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO meetings (meeting_id, path, turn_count, source, created_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(meeting_id) DO UPDATE SET
                path=excluded.path,
                turn_count=excluded.turn_count,
                source=excluded.source
            """,
            (meeting_id, path, turn_count, source, created_at),
        )
        conn.commit()


def list_meetings(source: str | None = None) -> list[dict]:
    init_db()
    with _connect() as conn:
        if source is None:
            rows = conn.execute(
                "SELECT meeting_id, path, turn_count, source, created_at "
                "FROM meetings ORDER BY source, meeting_id"
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT meeting_id, path, turn_count, source, created_at "
                "FROM meetings WHERE source = ? ORDER BY meeting_id",
                (source,),
            ).fetchall()
    return [dict(row) for row in rows]


def delete_upload_meetings() -> None:
    init_db()
    with _connect() as conn:
        conn.execute("DELETE FROM meetings WHERE source = 'upload'")
        conn.commit()


def seed_demo_meetings(demo_dir: Path) -> None:
    """Register demo transcript files present on disk."""
    init_db()
    from engine.corpus import load_transcript

    for path in sorted(demo_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in {".txt", ".vtt"}:
            continue
        turns = load_transcript(path)
        upsert_meeting(
            meeting_id=path.stem,
            path=str(path),
            turn_count=len(turns),
            source="demo",
        )
