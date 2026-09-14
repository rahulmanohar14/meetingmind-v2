"""Extract entities/relations from the corpus and save data/graph.json."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.corpus import load_corpus
from engine.extract import extract_from_turns
from engine.graph_store import build_graph, save_graph
from engine.llm import get_call_count

OUT_PATH = ROOT / "data" / "graph.json"


def main() -> None:
    turns = load_corpus(ROOT / "data")
    if not turns:
        raise RuntimeError(f"No transcripts found in {ROOT / 'data'}")

    print(f"Extracting from {len(turns)} turns ...")
    extractions = extract_from_turns(turns)
    print(
        f"Raw extraction: {len(extractions['entities'])} entities, "
        f"{len(extractions['relations'])} relations"
    )

    g = build_graph(extractions)
    save_graph(g, OUT_PATH)
    print(f"Saved graph to {OUT_PATH}")
    print(f"Nodes: {g.number_of_nodes()}")
    print(f"Edges: {g.number_of_edges()}")
    print(f"API calls used: {get_call_count()}")


if __name__ == "__main__":
    main()
