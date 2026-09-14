"""Smoke test: load data/graph.json and run a blocker multi-hop local search."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.graph_store import find_entities, load_graph, local_search

GRAPH_PATH = ROOT / "data" / "graph.json"
QUERY = "DataCorp"


def main() -> None:
    if not GRAPH_PATH.exists():
        raise RuntimeError(f"Missing {GRAPH_PATH}; run scripts/build_graph.py first")

    g = load_graph(GRAPH_PATH)
    print(f"Loaded graph: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")

    entities = find_entities(g, QUERY)
    print(f"find_entities({QUERY!r}): {entities}")

    _, facts = local_search(g, entities, hops=2)
    print(f"local_search facts ({len(facts)}):")
    for fact in facts:
        print(f"  - {fact}")

    if not facts:
        raise RuntimeError("Expected multi-hop relation facts; graph may be too sparse")


if __name__ == "__main__":
    main()
