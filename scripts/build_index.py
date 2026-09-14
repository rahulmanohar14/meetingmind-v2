"""Build the persistent hybrid retrieval index from data/ transcripts."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.corpus import load_corpus
from engine.retrieval import build_index

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
PERSIST_DIR = ROOT / "data" / "chroma_db"


def main() -> None:
    turns = load_corpus(ROOT / "data")
    if not turns:
        raise RuntimeError(f"No transcripts found in {ROOT / 'data'}")
    index = build_index(turns, EMBED_MODEL, PERSIST_DIR)
    print(f"Indexed {len(index.turn_ids)} turns into {PERSIST_DIR}")


if __name__ == "__main__":
    main()
