"""Agent-level evals: retrieval coverage and abstention.

Makes zero LLM calls. Both measurements live entirely on the retrieval path,
which is local, so this script is free to rerun and the call count at the end
is an assertion, not a note.

Rebuilds the Chroma collection from data/ only, exactly like scripts/benchmark.py.
Uploads in data/uploads/ are excluded from the index this writes; re-ingest or
use "Reset uploads" in the UI afterwards if you had any.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Transcript and question text is not cp1252-safe on the Windows console.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from engine.agent import CANDIDATE_K, TOP_N
from engine.corpus import load_corpus
from engine.llm import get_call_count
from engine.retrieval import (
    RELEVANCE_FLOOR,
    build_index,
    hybrid_search,
    rerank_with_threshold,
)

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
PERSIST_DIR = ROOT / "data" / "chroma_db"
GOLDEN_PATH = ROOT / "eval" / "golden_set.json"
OFF_TOPIC_PATH = ROOT / "eval" / "off_topic.json"
PARAPHRASE_PATH = ROOT / "eval" / "paraphrases.json"
RESULTS_PATH = ROOT / "benchmarks" / "agent_eval.md"


def _agent_turn_ids(index, question: str) -> list[str]:
    """Turn ids the agent's retrieve node would hand the generator."""
    candidates = hybrid_search(index, question, k=CANDIDATE_K)
    hits = rerank_with_threshold(
        question,
        [tid for tid, _ in candidates],
        index.turns,
        top_n=TOP_N,
    )
    return [tid for tid, _ in hits]


def _coverage(retrieved, required: list[str]) -> tuple[float, float]:
    """Return (full coverage 1/0, fraction of required turns found)."""
    req = set(required)
    if not req:
        return 0.0, 0.0
    found = req & set(retrieved)
    full = 1.0 if found == req else 0.0
    return full, len(found) / len(req)


def _count_answered(index, questions: list[dict], label: str) -> int:
    """How many of these questions retrieve at least one document above the floor."""
    answered = 0
    for item in questions:
        if _agent_turn_ids(index, item["question"]):
            answered += 1
        else:
            print(f"  [abstained] {label}: {item['question']}")
    return answered


def _eval_coverage(index, questions: list[dict]) -> str:
    n = len(questions)
    rows = []
    for item in questions:
        retrieved = _agent_turn_ids(index, item["question"])
        cov = _coverage(retrieved, item["answer_turn_ids"])
        rows.append(cov)
        flag = "ok " if cov[0] else "MISS"
        print(f"  [{flag}] {item['question']}")
        if not cov[0]:
            print(f"         required={item['answer_turn_ids']} got={retrieved}")
    full = int(sum(r[0] for r in rows))
    partial = sum(r[1] for r in rows) / n if n else 0.0
    return "\n".join(
        [
            "| arm | full coverage | mean partial coverage |",
            "|---|---:|---:|",
            f"| hybrid + rerank, top-{TOP_N} (agent path) | {full}/{n} | "
            f"{partial:.2f} |",
        ]
    )


def _eval_abstention(
    index,
    golden: list[dict],
    paraphrases: list[dict],
    off_topic: list[dict],
) -> str:
    """Does the relevance floor separate corpus questions from off-topic ones?

    Golden-set questions are generated from the transcripts and score far
    higher than real phrasings, so they cannot calibrate this on their own.
    eval/paraphrases.json holds the kind of question a demo audience actually
    asks, and it is the set that catches a floor set too high.
    """
    answered_golden = _count_answered(index, golden, "golden")
    answered_para = _count_answered(index, paraphrases, "paraphrase")

    abstained_off = 0
    for item in off_topic:
        if _agent_turn_ids(index, item["question"]):
            print(f"  [MISS] answered an off-topic question: {item['question']}")
        else:
            abstained_off += 1

    n_golden = len(golden)
    n_para = len(paraphrases)
    n_off = len(off_topic)
    print(f"  answered {answered_golden}/{n_golden} golden questions")
    print(f"  answered {answered_para}/{n_para} natural paraphrases")
    print(f"  abstained on {abstained_off}/{n_off} off-topic questions")
    return "\n".join(
        [
            "| question set | n | desired behaviour | correct |",
            "|---|---:|---|---:|",
            f"| golden set (in corpus) | {n_golden} | answer | "
            f"{answered_golden}/{n_golden} |",
            f"| natural paraphrases (eval/paraphrases.json) | {n_para} | answer | "
            f"{answered_para}/{n_para} |",
            f"| off-topic (eval/off_topic.json) | {n_off} | abstain | "
            f"{abstained_off}/{n_off} |",
        ]
    )


def _load_questions(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError(f"Expected a non-empty JSON array in {path}")
    return data


def main() -> None:
    started = time.perf_counter()

    turns = load_corpus(ROOT / "data")
    if not turns:
        raise RuntimeError(f"No transcripts found in {ROOT / 'data'}")

    golden = _load_questions(GOLDEN_PATH)
    off_topic = _load_questions(OFF_TOPIC_PATH)
    paraphrases = _load_questions(PARAPHRASE_PATH)

    print(f"Building index with {EMBED_MODEL} ...")
    index = build_index(turns, EMBED_MODEL, PERSIST_DIR)
    print(f"Indexed turns: {len(index.turn_ids)}")

    print(f"\n--- retrieval coverage ({len(golden)} golden questions) ---")
    coverage_table = _eval_coverage(index, golden)

    print(f"\n--- abstention (relevance floor {RELEVANCE_FLOOR}) ---")
    abstention_table = _eval_abstention(index, golden, paraphrases, off_topic)

    body = "\n\n".join(
        [
            "# Agent evals",
            "The agent has one retrieval path: hybrid fusion to "
            f"{CANDIDATE_K} candidates, cross-encoder rerank to {TOP_N}, then "
            "the relevance floor. Full coverage means every turn id the "
            "question needs was retrieved; partial coverage is the mean "
            "fraction retrieved.",
            f"## Retrieval coverage ({len(golden)} questions)",
            coverage_table,
            "## Abstention",
            "The agent passes the generator only documents scoring at least "
            f"{RELEVANCE_FLOOR} on the cross-encoder, and abstains without an "
            "API call when none clear it.",
            abstention_table,
        ]
    )

    print()
    print(body)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(body + "\n", encoding="utf-8")
    print(f"\nWrote {RESULTS_PATH}")
    print(f"Total runtime: {time.perf_counter() - started:.1f}s")

    calls = get_call_count()
    print(f"LLM API calls made: {calls} (expected 0)")
    if calls:
        raise RuntimeError(f"Expected 0 API calls on the retrieval path, made {calls}")


if __name__ == "__main__":
    main()
