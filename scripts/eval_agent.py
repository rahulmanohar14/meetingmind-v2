"""Agent-level evals: multi-hop coverage (vector vs graph) and router accuracy.

The retrieval arms make no LLM calls. Router accuracy makes one call per golden
question through engine.llm.complete, cached on disk after the first run.

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

from engine.agent import route_question
from engine.corpus import load_corpus
from engine.graph_store import find_entities, load_graph, local_search
from engine.llm import get_call_count
from engine.retrieval import (
    RELEVANCE_FLOOR,
    build_index,
    hybrid_search,
    rerank,
    rerank_with_threshold,
)

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
PERSIST_DIR = ROOT / "data" / "chroma_db"
GRAPH_PATH = ROOT / "data" / "graph.json"
GOLDEN_PATH = ROOT / "eval" / "golden_set.json"
OFF_TOPIC_PATH = ROOT / "eval" / "off_topic.json"
PARAPHRASE_PATH = ROOT / "eval" / "paraphrases.json"
RESULTS_PATH = ROOT / "benchmarks" / "agent_eval.md"

# Mirror of the agent's vector path: hybrid k=20, cross-encoder rerank to 5.
AGENT_CANDIDATES = 20
AGENT_TOP_N = 5
GENEROUS_TOP_N = 10

# Expected route per question type. single_hop is a direct factual lookup;
# multi_hop spans meetings and needs relation traversal.
EXPECTED_ROUTE = {"single_hop": "vector", "multi_hop": "graph"}


def _vector_turn_ids(index, question: str, top_n: int, thresholded: bool = True) -> list[str]:
    """Turn ids from the vector path.

    thresholded=True mirrors the agent exactly (relevance floor applied).
    thresholded=False is the generous arm: every reranked hit, no floor, so
    nobody can claim the vector comparison was rigged by truncation.
    """
    candidates = hybrid_search(index, question, k=AGENT_CANDIDATES)
    ranker = rerank_with_threshold if thresholded else rerank
    hits = ranker(
        question,
        [tid for tid, _ in candidates],
        index.turns,
        top_n=top_n,
    )
    return [tid for tid, _ in hits]


def _count_answered(index, questions: list[dict], label: str) -> int:
    """How many of these questions retrieve at least one document above the floor."""
    answered = 0
    for item in questions:
        if _vector_turn_ids(index, item["question"], AGENT_TOP_N):
            answered += 1
        else:
            print(f"  [abstained] {label}: {item['question']}")
    return answered


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
        if _vector_turn_ids(index, item["question"], AGENT_TOP_N):
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


def _graph_covered_turn_ids(graph, question: str) -> tuple[list[str], list[str], set[str]]:
    """Entities seeded, relation facts returned, and the turn ids those facts cite.

    Only edge source_turns count: graph_search passes relation facts to the
    generator, so those citations are the evidence the model actually sees.
    """
    entities = find_entities(graph, question)
    sub, facts = local_search(graph, entities, hops=2)
    covered: set[str] = set()
    for _src, _tgt, attrs in sub.edges(data=True):
        covered.update(attrs.get("source_turns") or [])
    return entities, facts, covered


def _coverage(retrieved, required: list[str]) -> tuple[float, float]:
    """Return (full coverage 1/0, fraction of required turns found)."""
    req = set(required)
    if not req:
        return 0.0, 0.0
    found = req & set(retrieved)
    full = 1.0 if found == req else 0.0
    return full, len(found) / len(req)


def _summarise(rows: list[tuple[float, float]], n: int) -> tuple[str, float]:
    full = int(sum(r[0] for r in rows))
    partial = sum(r[1] for r in rows) / n if n else 0.0
    return f"{full}/{n}", partial


def _eval_multi_hop(index, graph, questions: list[dict]) -> tuple[str, list[dict]]:
    n = len(questions)
    agent_rows: list[tuple[float, float]] = []
    generous_rows: list[tuple[float, float]] = []
    graph_rows: list[tuple[float, float]] = []
    detail: list[dict] = []

    for item in questions:
        question = item["question"]
        required = item["answer_turn_ids"]

        agent_ids = _vector_turn_ids(index, question, AGENT_TOP_N)
        generous_ids = _vector_turn_ids(
            index, question, GENEROUS_TOP_N, thresholded=False
        )
        entities, facts, graph_ids = _graph_covered_turn_ids(graph, question)

        agent_cov = _coverage(agent_ids, required)
        generous_cov = _coverage(generous_ids, required)
        graph_cov = _coverage(graph_ids, required)

        agent_rows.append(agent_cov)
        generous_rows.append(generous_cov)
        graph_rows.append(graph_cov)

        print(f"\n=== {question}")
        print(f"  required turns : {required}")
        print(f"  vector top-{AGENT_TOP_N}   : {agent_ids}")
        print(f"    full={agent_cov[0]:.0f} partial={agent_cov[1]:.2f}")
        print(f"  vector top-{GENEROUS_TOP_N}  : {generous_ids}")
        print(f"    full={generous_cov[0]:.0f} partial={generous_cov[1]:.2f}")
        print(f"  graph seeds     : {len(entities)} entities, {len(facts)} facts")
        print(f"    cited turns={sorted(graph_ids)}")
        print(f"    full={graph_cov[0]:.0f} partial={graph_cov[1]:.2f}")

        detail.append(
            {
                "question": question,
                "required": required,
                "vector_partial": agent_cov[1],
                "graph_partial": graph_cov[1],
                "graph_entities": len(entities),
                "graph_facts": len(facts),
            }
        )

    agent_full, agent_partial = _summarise(agent_rows, n)
    generous_full, generous_partial = _summarise(generous_rows, n)
    graph_full, graph_partial = _summarise(graph_rows, n)

    table = "\n".join(
        [
            "| arm | full coverage | mean partial coverage |",
            "|---|---:|---:|",
            f"| vector, hybrid+rerank top-{AGENT_TOP_N} (agent path) | "
            f"{agent_full} | {agent_partial:.2f} |",
            f"| vector, hybrid+rerank top-{GENEROUS_TOP_N} (generous) | "
            f"{generous_full} | {generous_partial:.2f} |",
            f"| graph, 2-hop relation facts | {graph_full} | {graph_partial:.2f} |",
        ]
    )
    return table, detail


def _eval_single_hop(index, questions: list[dict]) -> str:
    n = len(questions)
    rows = [
        _coverage(_vector_turn_ids(index, q["question"], AGENT_TOP_N), q["answer_turn_ids"])
        for q in questions
    ]
    full, partial = _summarise(rows, n)
    return "\n".join(
        [
            "| arm | full coverage | mean partial coverage |",
            "|---|---:|---:|",
            f"| vector, hybrid+rerank top-{AGENT_TOP_N} (agent path) | "
            f"{full} | {partial:.2f} |",
        ]
    )


def _eval_router(questions: list[dict]) -> str:
    by_expected: dict[str, list[bool]] = {"vector": [], "graph": []}

    for item in questions:
        expected = EXPECTED_ROUTE[item["type"]]
        state = route_question(
            {
                "question": item["question"],
                "search_question": "",
                "history": [],
                "route": "",
                "documents": [],
                "answer": "",
                "decision_log": [],
            }
        )
        actual = state.get("route", "")
        by_expected[expected].append(actual == expected)
        flag = "ok " if actual == expected else "MISS"
        print(f"  [{flag}] expected={expected} actual={actual or '(none)'} | {item['question']}")

    lines = [
        "| expected route | questions | correct | accuracy |",
        "|---|---:|---:|---:|",
    ]
    total = 0
    correct = 0
    for expected in ("vector", "graph"):
        results = by_expected[expected]
        n = len(results)
        hits = sum(results)
        total += n
        correct += hits
        acc = hits / n if n else 0.0
        lines.append(f"| {expected} | {n} | {hits} | {acc:.2f} |")
    overall = correct / total if total else 0.0
    lines.append(f"| **overall** | {total} | {correct} | **{overall:.2f}** |")
    return "\n".join(lines)


def main() -> None:
    started = time.perf_counter()

    turns = load_corpus(ROOT / "data")
    if not turns:
        raise RuntimeError(f"No transcripts found in {ROOT / 'data'}")
    if not GRAPH_PATH.exists():
        raise RuntimeError(f"Missing {GRAPH_PATH}; run scripts/build_graph.py first")

    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    if not isinstance(golden, list):
        raise ValueError(f"Expected a JSON array in {GOLDEN_PATH}")

    single_hop = [q for q in golden if q.get("type") == "single_hop"]
    multi_hop = [q for q in golden if q.get("type") == "multi_hop"]
    if not multi_hop:
        raise RuntimeError("No multi_hop questions found in the golden set")

    print(f"Building index with {EMBED_MODEL} ...")
    index = build_index(turns, EMBED_MODEL, PERSIST_DIR)
    graph = load_graph(GRAPH_PATH)
    print(
        f"Indexed turns: {len(index.turn_ids)} | "
        f"graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
    )

    print(f"\n--- multi-hop coverage ({len(multi_hop)} questions) ---")
    multi_table, _detail = _eval_multi_hop(index, graph, multi_hop)

    print(f"\n--- single-hop sanity check ({len(single_hop)} questions) ---")
    single_table = _eval_single_hop(index, single_hop)

    off_topic = json.loads(OFF_TOPIC_PATH.read_text(encoding="utf-8"))
    if not isinstance(off_topic, list) or not off_topic:
        raise ValueError(f"Expected a non-empty JSON array in {OFF_TOPIC_PATH}")
    paraphrases = json.loads(PARAPHRASE_PATH.read_text(encoding="utf-8"))
    if not isinstance(paraphrases, list) or not paraphrases:
        raise ValueError(f"Expected a non-empty JSON array in {PARAPHRASE_PATH}")

    print(f"\n--- abstention (relevance floor {RELEVANCE_FLOOR}) ---")
    abstention_table = _eval_abstention(index, golden, paraphrases, off_topic)

    print(f"\n--- router accuracy ({len(golden)} questions) ---")
    router_table = _eval_router(golden)

    body = "\n\n".join(
        [
            "# Agent evals",
            "Full coverage means every turn id the question needs was retrieved; "
            "partial coverage is the mean fraction retrieved. The vector arms "
            "mirror the agent's vector path (hybrid k=20 then cross-encoder "
            "rerank). The graph arm counts the turn ids cited by the relation "
            "facts that `graph_search` hands the generator.",
            f"## Multi-hop coverage ({len(multi_hop)} questions)",
            multi_table,
            f"## Single-hop coverage ({len(single_hop)} questions)",
            single_table,
            "## Abstention",
            "The vector route passes the generator only documents scoring at "
            f"least {RELEVANCE_FLOOR} on the cross-encoder, and abstains when "
            "none clear it. The value sits in the measured gap between natural "
            "in-corpus phrasings (down to -5.0 at top-1) and off-topic "
            "questions (up to -9.7).",
            abstention_table,
            f"## Router accuracy ({len(golden)} questions)",
            "Expected route is derived from the golden set question type: "
            "`single_hop` -> vector, `multi_hop` -> graph.",
            router_table,
            "Sample sizes are small (12 single-hop, 5 multi-hop): one question "
            "moves single-hop metrics by 0.08 and multi-hop metrics by 0.20, so "
            "differences of a single question are not meaningful.",
        ]
    )

    print()
    print(body)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(body + "\n", encoding="utf-8")
    print(f"\nWrote {RESULTS_PATH}")
    print(f"Total runtime: {time.perf_counter() - started:.1f}s")
    print(f"LLM API calls made: {get_call_count()}")


if __name__ == "__main__":
    main()
