"""Ingest uploaded transcripts into the shared retrieval index.

No LLM calls: ingesting a transcript is parse plus embed, both local.
"""

from __future__ import annotations

import re
from pathlib import Path

from engine.corpus import load_full_corpus, load_transcript
from engine.db import delete_upload_meetings, seed_demo_meetings, upsert_meeting
from engine.retrieval import build_index

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
PERSIST_DIR = DATA_DIR / "chroma_db"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

_SAFE_STEM = re.compile(r"[^A-Za-z0-9_-]+")


def _safe_stem(name: str) -> str:
    stem = Path(name).stem
    cleaned = _SAFE_STEM.sub("_", stem).strip("_")
    return cleaned or "upload"


def save_upload(filename: str, content: bytes | str) -> Path:
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(filename).suffix.lower()
    if suffix not in {".txt", ".vtt"}:
        raise ValueError("Only .txt and .vtt uploads are supported")
    dest = UPLOADS_DIR / f"{_safe_stem(filename)}{suffix}"
    if isinstance(content, bytes):
        dest.write_bytes(content)
    else:
        dest.write_text(content, encoding="utf-8")
    return dest


def rebuild_index() -> int:
    """Re-embed the whole corpus (demo transcripts plus uploads)."""
    turns = load_full_corpus(DATA_DIR, UPLOADS_DIR)
    if not turns:
        raise RuntimeError("No transcripts to index")
    index = build_index(turns, EMBED_MODEL, PERSIST_DIR)
    return len(index.turn_ids)


def ingest_upload(path: Path, status_callback=None) -> dict:
    """Parse an upload, register it, and re-embed the corpus."""

    def _status(msg: str) -> None:
        if status_callback is not None:
            status_callback(msg)

    path = Path(path)
    seed_demo_meetings(DATA_DIR)

    _status("parsing")
    new_turns = load_transcript(path)
    if not new_turns:
        raise RuntimeError(f"No turns parsed from {path}")

    upsert_meeting(
        meeting_id=path.stem,
        path=str(path),
        turn_count=len(new_turns),
        source="upload",
    )

    _status("indexing")
    turn_count = rebuild_index()

    from engine.agent import reload_stores

    reload_stores()

    _status("done")
    return {
        "meeting_id": path.stem,
        "new_turns": len(new_turns),
        "indexed_turns": turn_count,
    }


def reset_uploads(status_callback=None) -> dict:
    """Drop every upload and re-embed the demo corpus alone. No API calls."""

    def _status(msg: str) -> None:
        if status_callback is not None:
            status_callback(msg)

    _status("clearing uploads")
    if UPLOADS_DIR.is_dir():
        for path in UPLOADS_DIR.iterdir():
            if path.is_file():
                path.unlink()
    delete_upload_meetings()

    _status("indexing")
    seed_demo_meetings(DATA_DIR)
    turn_count = rebuild_index()

    from engine.agent import reload_stores

    reload_stores()

    _status("done")
    return {"indexed_turns": turn_count}
