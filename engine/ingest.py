"""Ingest uploaded transcripts into the shared index and graph."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

from engine.corpus import load_full_corpus, load_transcript
from engine.db import (
    delete_upload_meetings,
    seed_demo_meetings,
    upsert_meeting,
)
from engine.extract import extract_from_turns
from engine.graph_store import load_graph, merge_extractions, save_graph
from engine.retrieval import build_index

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
PERSIST_DIR = DATA_DIR / "chroma_db"
GRAPH_PATH = DATA_DIR / "graph.json"
BASELINE_GRAPH_PATH = DATA_DIR / "graph_baseline.json"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

_SAFE_STEM = re.compile(r"[^A-Za-z0-9_-]+")


def ensure_baseline_graph() -> None:
    """Copy current graph.json to graph_baseline.json once (demo restore point)."""
    if BASELINE_GRAPH_PATH.exists():
        return
    if not GRAPH_PATH.exists():
        raise RuntimeError(
            f"Missing {GRAPH_PATH}; run scripts/build_graph.py before ingesting."
        )
    shutil.copy2(GRAPH_PATH, BASELINE_GRAPH_PATH)


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
    turns = load_full_corpus(DATA_DIR, UPLOADS_DIR)
    if not turns:
        raise RuntimeError("No transcripts to index")
    index = build_index(turns, EMBED_MODEL, PERSIST_DIR)
    return len(index.turn_ids)


def ingest_upload(path: Path, status_callback=None) -> dict:
    """Parse upload, rebuild index, extract+merge graph, update DB."""

    def _status(msg: str) -> None:
        if status_callback is not None:
            status_callback(msg)

    path = Path(path)
    ensure_baseline_graph()
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

    _status("extracting")
    extractions = extract_from_turns(new_turns)
    if GRAPH_PATH.exists():
        g = load_graph(GRAPH_PATH)
        g = merge_extractions(g, extractions)
    else:
        from engine.graph_store import build_graph

        g = build_graph(extractions)
    save_graph(g, GRAPH_PATH)

    from engine.agent import reload_stores

    reload_stores()

    _status("done")
    return {
        "meeting_id": path.stem,
        "new_turns": len(new_turns),
        "indexed_turns": turn_count,
        "nodes": g.number_of_nodes(),
        "edges": g.number_of_edges(),
        "entities_extracted": len(extractions.get("entities") or []),
        "relations_extracted": len(extractions.get("relations") or []),
    }


def reset_uploads(status_callback=None) -> dict:
    """Remove uploads, restore baseline graph, rebuild demo index (no API)."""

    def _status(msg: str) -> None:
        if status_callback is not None:
            status_callback(msg)

    _status("clearing uploads")
    if UPLOADS_DIR.is_dir():
        for path in UPLOADS_DIR.iterdir():
            if path.is_file():
                path.unlink()
    delete_upload_meetings()

    _status("restoring graph baseline")
    if not BASELINE_GRAPH_PATH.exists() and GRAPH_PATH.exists():
        shutil.copy2(GRAPH_PATH, BASELINE_GRAPH_PATH)
    if BASELINE_GRAPH_PATH.exists():
        shutil.copy2(BASELINE_GRAPH_PATH, GRAPH_PATH)
    elif not GRAPH_PATH.exists():
        raise RuntimeError(
            "No graph baseline or graph.json; run scripts/build_graph.py"
        )

    _status("indexing")
    seed_demo_meetings(DATA_DIR)
    turn_count = rebuild_index()

    from engine.agent import reload_stores

    reload_stores()

    g = load_graph(GRAPH_PATH)
    _status("done")
    return {
        "indexed_turns": turn_count,
        "nodes": g.number_of_nodes(),
        "edges": g.number_of_edges(),
    }
