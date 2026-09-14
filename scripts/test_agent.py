"""Smoke test the LangGraph routing agent on three hardcoded questions."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.agent import run

QUESTIONS = [
    # clearly factual -> expect vector
    "What is the agreed-upon launch date for the beta version?",
    # clearly multi-hop / relational -> expect graph
    "What is blocking the launch and what other work does that delay?",
    # ambiguous
    "What did Alice decide about beta seats?",
]


def main() -> None:
    results = []
    for i, question in enumerate(QUESTIONS, start=1):
        print(f"\n=== Q{i}: {question} ===")
        result = run(question)
        results.append(result)
        print(f"route: {result['route']}")
        print("decision_log:")
        for entry in result["decision_log"]:
            print(f"  - {entry}")
        print(f"documents ({len(result['documents'])}):")
        for doc in result["documents"][:8]:
            print(f"  - {doc}")
        print(f"answer:\n{result['answer']}")

    # Centrepiece check: multi-hop question must route to graph.
    if results[1]["route"] != "graph":
        raise RuntimeError(
            f"Expected graph route for multi-hop question, got {results[1]['route']!r}"
        )
    print("\nRouting check passed: multi-hop question -> graph")


if __name__ == "__main__":
    main()
