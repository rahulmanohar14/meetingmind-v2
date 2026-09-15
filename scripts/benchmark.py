"""Benchmark dense / BM25 / hybrid / hybrid+rerank on the golden question set."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.corpus import load_corpus
from engine.llm import get_call_count
from engine.retrieval import (
    bm25_search,
    build_index,
    dense_search,
    hybrid_search,
    rerank,
)

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
PERSIST_DIR = ROOT / "data" / "chroma_db"
GOLDEN_PATH = ROOT / "eval" / "golden_set.json"
RESULTS_PATH = ROOT / "benchmarks" / "results.md"

# rrf_k controls how much weight the fusion gives to top ranks. 60 is the value
# from the original RRF paper; the sweep shows what it is worth on this corpus.
RRF_K_SWEEP = (10, 20, 60, 120)


def _recall_at_k(ranked_ids: list[str], relevant: set[str], k: int) -> float:
    return 1.0 if any(tid in relevant for tid in ranked_ids[:k]) else 0.0


def _mrr_at_k(ranked_ids: list[str], relevant: set[str], k: int) -> float:
    for rank, tid in enumerate(ranked_ids[:k], start=1):
        if tid in relevant:
            return 1.0 / rank
    return 0.0


def _run_config(name: str, index, questions: list[dict]) -> dict:
    recalls: list[float] = []
    mrrs: list[float] = []
    latencies_ms: list[float] = []

    for item in tqdm(questions, desc=name):
        query = item["question"]
        relevant = set(item["answer_turn_ids"])
        t0 = time.perf_counter()
        if name == "dense":
            hits = dense_search(index, query, 10)
        elif name == "bm25":
            hits = bm25_search(index, query, 10)
        elif name == "hybrid":
            hits = hybrid_search(index, query, 10)
        elif name == "hybrid+rerank":
            candidates = hybrid_search(index, query, 20)
            hits = rerank(
                query,
                [tid for tid, _ in candidates],
                index.turns,
                top_n=10,
            )
        else:
            raise ValueError(f"Unknown config: {name}")
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        ranked_ids = [tid for tid, _ in hits]
        recalls.append(_recall_at_k(ranked_ids, relevant, 5))
        mrrs.append(_mrr_at_k(ranked_ids, relevant, 10))
        latencies_ms.append(elapsed_ms)

    n = len(questions)
    return {
        "config": name,
        "recall@5": sum(recalls) / n if n else 0.0,
        "MRR@10": sum(mrrs) / n if n else 0.0,
        "mean_latency_ms": sum(latencies_ms) / n if n else 0.0,
    }


def _run_rrf_sweep(index, questions: list[dict]) -> str:
    lines = ["| rrf_k | recall@5 | MRR@10 |", "|---:|---:|---:|"]
    n = len(questions)
    for rrf_k in RRF_K_SWEEP:
        recalls: list[float] = []
        mrrs: list[float] = []
        for item in questions:
            hits = hybrid_search(index, item["question"], 10, rrf_k=rrf_k)
            ranked_ids = [tid for tid, _ in hits]
            relevant = set(item["answer_turn_ids"])
            recalls.append(_recall_at_k(ranked_ids, relevant, 5))
            mrrs.append(_mrr_at_k(ranked_ids, relevant, 10))
        recall = sum(recalls) / n if n else 0.0
        mrr = sum(mrrs) / n if n else 0.0
        marker = " (default)" if rrf_k == 60 else ""
        lines.append(f"| {rrf_k}{marker} | {recall:.3f} | {mrr:.3f} |")
    return "\n".join(lines)


def _format_table(rows: list[dict]) -> str:
    header = (
        "| config | recall@5 | MRR@10 | mean latency (ms) |\n"
        "|---|---:|---:|---:|"
    )
    lines = [header]
    for row in rows:
        lines.append(
            f"| {row['config']} | {row['recall@5']:.3f} | "
            f"{row['MRR@10']:.3f} | {row['mean_latency_ms']:.1f} |"
        )
    return "\n".join(lines)


def main() -> None:
    started = time.perf_counter()
    turns = load_corpus(ROOT / "data")
    if not turns:
        raise RuntimeError(f"No transcripts found in {ROOT / 'data'}")

    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    if not isinstance(golden, list) or not golden:
        raise ValueError(f"Expected a non-empty JSON array in {GOLDEN_PATH}")

    print(f"Evaluating {len(golden)} golden questions")

    print(f"Building index with {EMBED_MODEL} ...")
    index = build_index(turns, EMBED_MODEL, PERSIST_DIR)
    print(f"Indexed turn count: {len(index.turn_ids)}")

    configs = ["dense", "bm25", "hybrid", "hybrid+rerank"]
    rows = [_run_config(name, index, golden) for name in configs]
    rows.sort(key=lambda r: r["recall@5"], reverse=True)

    table = _format_table(rows)
    print()
    print(table)

    print("\nSweeping rrf_k ...")
    sweep_table = _run_rrf_sweep(index, golden)
    print(sweep_table)

    n = len(golden)
    body = "\n\n".join(
        [
            "# Retrieval benchmark",
            f"{n} single-hop golden questions over "
            f"{len(index.turn_ids)} indexed turns.",
            "## Retrieval configurations",
            table,
            "## rrf_k sensitivity (hybrid fusion)",
            sweep_table,
            f"With n={n}, one question is worth {1.0 / n:.3f} of recall@5, "
            "so gaps of a single question are noise. Read these numbers as "
            "ruling out large regressions, not as fine-grained rankings.",
        ]
    )

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(body + "\n", encoding="utf-8")
    print(f"\nWrote {RESULTS_PATH}")

    total_s = time.perf_counter() - started
    print(f"Indexed turn count: {len(index.turn_ids)}")
    print(f"Total runtime: {total_s:.1f}s")
    print(f"LLM API calls made: {get_call_count()} (expected 0)")


if __name__ == "__main__":
    main()
