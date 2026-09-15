"""Calibrate the cross-encoder relevance floor against three score populations.

Makes zero LLM calls. Everything here is local retrieval.

The floor decides when the agent abstains. It has to sit above the best score an
off-topic question can reach and below the worst score a real in-corpus question
reaches, so it needs all three populations:

  golden      LLM-generated from the transcripts. Phrased in transcript
              vocabulary, so it scores far higher than a human would. Useless
              for setting the lower bound on its own.
  paraphrase  How a person actually asks. This is the population that sets the
              lower bound, and the one that catches a floor set too high.
  off_topic   Not answerable from the corpus. Sets the upper bound.

The value is corpus-specific: re-run this whenever the corpus changes.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from engine.agent import CANDIDATE_K
from engine.corpus import load_demo_corpus
from engine.llm import get_call_count
from engine.retrieval import RELEVANCE_FLOOR, build_index, hybrid_search, rerank

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
PERSIST_DIR = ROOT / "data" / "chroma_db"
EVAL_DIR = ROOT / "eval"
POPULATIONS = ("golden_set", "paraphrases", "off_topic")


def _top1_score(index, question: str) -> tuple[float, str]:
    """Best cross-encoder score for a question, with no floor applied."""
    candidates = hybrid_search(index, question, k=CANDIDATE_K)
    hits = rerank(question, [tid for tid, _ in candidates], index.turns, top_n=1)
    if not hits:
        return float("-inf"), ""
    return hits[0][1], hits[0][0]


def _summarise(name: str, scores: list[float], pairs: list) -> dict:
    return {
        "name": name,
        "n": len(scores),
        "min": min(scores),
        "max": max(scores),
        "mean": sum(scores) / len(scores),
        "scores": scores,
        "pairs": pairs,
    }


def main() -> None:
    started = time.perf_counter()
    turns = load_demo_corpus(ROOT / "data")
    if not turns:
        raise RuntimeError(f"No transcripts found in {ROOT / 'data'}")

    print(f"Building index with {EMBED_MODEL} ...")
    index = build_index(turns, EMBED_MODEL, PERSIST_DIR)
    print(f"Indexed turns: {len(index.turn_ids)}\n")

    summaries = []
    for name in POPULATIONS:
        items = json.loads((EVAL_DIR / f"{name}.json").read_text(encoding="utf-8"))
        scores = []
        pairs = []
        print(f"--- {name} ({len(items)} questions) ---")
        for item in items:
            score, tid = _top1_score(index, item["question"])
            scores.append(score)
            pairs.append((item, score))
            print(f"  {score:8.2f}  {tid:28}  {item['question'][:60]}")
        summaries.append(_summarise(name, scores, pairs))
        print()

    print("| population | n | min | max | mean |")
    print("|---|---:|---:|---:|---:|")
    for s in summaries:
        print(
            f"| {s['name']} | {s['n']} | {s['min']:.2f} | "
            f"{s['max']:.2f} | {s['mean']:.2f} |"
        )

    by_name = {s["name"]: s for s in summaries}
    # The floor must admit the weakest real question and reject the strongest
    # off-topic one. Anything in that gap works; the midpoint maximises margin
    # on both sides.
    lower = min(by_name["golden_set"]["min"], by_name["paraphrases"]["min"])
    upper = by_name["off_topic"]["max"]
    print()
    print(f"Weakest in-corpus top-1 (golden or paraphrase): {lower:.2f}")
    print(f"Strongest off-topic top-1:                      {upper:.2f}")

    in_corpus = by_name["golden_set"]["scores"] + by_name["paraphrases"]["scores"]
    off = by_name["off_topic"]["scores"]

    if upper < lower:
        suggested = round((lower + upper) / 2, 1)
        print(f"Usable gap: {upper:.2f} .. {lower:.2f} (width {lower - upper:.2f})")
        print(f"Suggested RELEVANCE_FLOOR: {suggested}")
        print(f"  margin below weakest real question: {lower - suggested:.2f}")
        print(f"  margin above strongest off-topic:   {suggested - upper:.2f}")
    else:
        # The populations overlap, so no floor separates them cleanly and the
        # choice becomes an explicit trade: every candidate wrongly rejects some
        # real questions or wrongly answers some off-topic ones. Sweep the
        # candidates and pick the one with the fewest total mistakes, breaking
        # ties toward answering real questions, since a wrong abstention is a
        # visibly broken product and a wrong answer is at least cited.
        print(
            f"\nPOPULATIONS OVERLAP: the weakest real question ({lower:.2f}) "
            f"scores below the strongest off-topic one ({upper:.2f}), so no "
            "floor divides them. Choosing by total misclassifications."
        )
        candidates = sorted({round(s - 0.5, 1) for s in in_corpus + off})
        best = None
        for floor in candidates:
            wrong_abstain = sum(1 for s in in_corpus if s < floor)
            wrong_answer = sum(1 for s in off if s >= floor)
            total = wrong_abstain + wrong_answer
            if best is None or (total, wrong_abstain) < (best[1], best[2]):
                best = (floor, total, wrong_abstain, wrong_answer)
        floor, total, wrong_abstain, wrong_answer = best
        print(f"\nBest RELEVANCE_FLOOR: {floor}")
        print(f"  real questions wrongly abstained: {wrong_abstain}/{len(in_corpus)}")
        print(f"  off-topic questions wrongly answered: {wrong_answer}/{len(off)}")
        print(f"  total misclassifications: {total}/{len(in_corpus) + len(off)}")
        print("\n  would wrongly ANSWER these off-topic questions:")
        for name in POPULATIONS:
            for item, score in by_name[name]["pairs"]:
                if name == "off_topic" and score >= floor:
                    print(f"    {score:7.2f}  {item['question']}")
        print("\n  would wrongly ABSTAIN on these real questions:")
        none_wrong = True
        for name in ("golden_set", "paraphrases"):
            for item, score in by_name[name]["pairs"]:
                if score < floor:
                    print(f"    {score:7.2f}  {item['question']}")
                    none_wrong = False
        if none_wrong:
            print("    (none)")

    print(f"\nCurrently set to {RELEVANCE_FLOOR} in engine/retrieval.py")

    print(f"\nTotal runtime: {time.perf_counter() - started:.1f}s")
    calls = get_call_count()
    print(f"LLM API calls made: {calls} (expected 0)")
    if calls:
        raise RuntimeError(f"Expected 0 API calls, made {calls}")


if __name__ == "__main__":
    main()
