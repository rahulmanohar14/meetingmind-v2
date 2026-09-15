"""Smoke test the RAG agent: a factual question, a follow-up, and an off-topic one."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Model answers contain characters cp1252 cannot encode (narrow no-break
# space, typographic quotes), which crashed this script on the Windows console.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from engine.agent import ABSTAIN_ANSWER, run

CITATION = re.compile(r"\([^()]*\bline\s+\d+\)")

# A grounded question, a follow-up that only resolves against the history, and
# a question the corpus cannot answer.
FACTUAL = "What is the agreed-upon launch date for the beta version?"
FOLLOW_UP = "Who agreed to that?"
OFF_TOPIC = "What is the capital of Mongolia?"


def _show(label: str, result: dict) -> None:
    print(f"\n=== {label} ===")
    print(f"standalone_question: {result['standalone_question']}")
    print("decision_log:")
    for entry in result["decision_log"]:
        print(f"  - {entry}")
    print(f"documents ({len(result['documents'])}):")
    for doc in result["documents"]:
        print(f"  - {doc}")
    print(f"answer:\n{result['answer']}")


def main() -> None:
    factual = run(FACTUAL)
    _show(FACTUAL, factual)
    if not factual["documents"]:
        raise RuntimeError("Expected the factual question to retrieve documents")
    if not CITATION.search(factual["answer"]):
        raise RuntimeError(f"Expected an inline citation, got: {factual['answer']!r}")

    history = [
        {"role": "user", "content": FACTUAL},
        {"role": "assistant", "content": factual["answer"]},
    ]
    follow_up = run(FOLLOW_UP, history=history)
    _show(FOLLOW_UP, follow_up)
    if follow_up["standalone_question"] == FOLLOW_UP:
        raise RuntimeError(
            "Expected the follow-up to be rewritten against the history, "
            f"got it unchanged: {follow_up['standalone_question']!r}"
        )

    off_topic = run(OFF_TOPIC)
    _show(OFF_TOPIC, off_topic)
    if off_topic["answer"] != ABSTAIN_ANSWER:
        raise RuntimeError(
            f"Expected the off-topic question to abstain, got: {off_topic['answer']!r}"
        )

    print("\nChecks passed: cited answer, follow-up rewritten, off-topic abstained")


if __name__ == "__main__":
    main()
