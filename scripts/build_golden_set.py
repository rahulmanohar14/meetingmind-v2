"""Generate a single-hop golden question set from the meeting corpus.

Every question is answerable from exactly one turn. The script prints each
question next to the turn it was generated from so the set can be reviewed by
hand before it is trusted as ground truth.

Turns listed in data/inconsistent_turns.json are excluded. Those turns state
one of the corpus's deliberately contradictory facts, so a question drawn from
one of them ("how many accessibility findings were there") has several
disagreeing answer turns. Retrieval would return a defensible one and score as
a miss, making recall@5 a measure of label ambiguity rather than of retrieval.
The facts stay in the corpus doing their job as noise; they never become
questions.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.corpus import Turn, load_demo_corpus, turn_id
from engine.llm import complete, get_call_count

CORPUS_DIR = ROOT / "data"
OUT_PATH = ROOT / "eval" / "golden_set.json"
INCONSISTENT_PATH = ROOT / "data" / "inconsistent_turns.json"
BATCH_SIZE = 18
QUESTIONS_PER_BATCH = 3
TARGET_QUESTIONS = 28

SYSTEM = (
    "You write evaluation questions for a meeting-transcript retrieval system. "
    "Every question must be answerable from exactly one provided turn. "
    "Return strict JSON only."
)


def _format_batch(turns: list[Turn]) -> str:
    lines = []
    for t in turns:
        lines.append(
            f"- turn_id={turn_id(t)} | meeting={t.meeting_id} | "
            f"line={t.line_number} | speaker={t.speaker} | text={t.text}"
        )
    return "\n".join(lines)


def _ask_for_questions(turns: list[Turn], n: int) -> list[dict]:
    prompt = (
        f"Here are transcript turns:\n{_format_batch(turns)}\n\n"
        f"Generate exactly {n} factual questions. For each question the answer "
        "must be contained in exactly one of these turns.\n"
        "Return a JSON array of objects with keys:\n"
        '  question: string\n'
        '  answer_turn_ids: array with exactly one string like "meeting_id:turn_index"\n'
        '  type: the string "single_hop"\n'
        "Do not invent turn ids. Only use turn ids from the list above."
    )
    raw = complete(prompt, system=SYSTEM, json_mode=True)
    data = json.loads(raw)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        # A single question object rather than an array.
        if "question" in data:
            return [data]
        # Groq's json_object mode cannot return a bare top-level array, so the
        # model wraps it under a key of its choosing. Match on the shape of the
        # values, not on a key name it was never given: picking the first list
        # found grabs "answer_turn_ids" off a single-question object instead.
        for value in data.values():
            if isinstance(value, list) and all(isinstance(v, dict) for v in value):
                return value
    raise ValueError(
        f"Expected a JSON array of questions from the model, got: {type(data)}"
    )


def _index_turns(turns: list[Turn]) -> dict[str, Turn]:
    return {turn_id(t): t for t in turns}


def _excluded_turn_ids() -> set[str]:
    """Turn ids that state a deliberately contradictory fact."""
    if not INCONSISTENT_PATH.exists():
        return set()
    data = json.loads(INCONSISTENT_PATH.read_text(encoding="utf-8"))
    return {tid for ids in data.values() for tid in ids}


def main() -> None:
    turns = load_demo_corpus(CORPUS_DIR)
    if not turns:
        raise RuntimeError(f"No .txt/.vtt transcripts found in {CORPUS_DIR}")

    excluded = _excluded_turn_ids()
    # Excluded turns are withheld from the model entirely, not just filtered out
    # of its answers: a turn it cannot see is a turn it cannot build a question
    # from, which costs no extra API calls to enforce.
    turns = [t for t in turns if turn_id(t) not in excluded]
    print(f"Excluded {len(excluded)} turns that state a contradictory fact")
    print(f"Sampling questions from {len(turns)} remaining turns")

    by_id = _index_turns(turns)
    collected: list[dict] = []
    valid_ids = set(by_id)

    # Spread batches across the whole corpus so questions are not concentrated
    # in a few meetings. There is no designated subset: every meeting is fair
    # game, including the low-signal ones.
    n_batches = max(1, -(-TARGET_QUESTIONS // QUESTIONS_PER_BATCH))
    step = max(1, len(turns) // n_batches)
    batch_starts = list(range(0, len(turns), step))[:n_batches]

    for start in batch_starts:
        if len(collected) >= TARGET_QUESTIONS:
            break
        batch = turns[start : start + BATCH_SIZE]
        if not batch:
            continue
        need = min(QUESTIONS_PER_BATCH, TARGET_QUESTIONS - len(collected))
        items = _ask_for_questions(batch, need)
        for item in items:
            if not isinstance(item, dict):
                print(f"  skipped malformed item: {item!r:.80}")
                continue
            q = item.get("question")
            ids = item.get("answer_turn_ids")
            qtype = item.get("type", "single_hop")
            if not isinstance(q, str) or not q.strip():
                continue
            if not isinstance(ids, list) or len(ids) != 1 or ids[0] not in valid_ids:
                continue
            if qtype != "single_hop":
                continue
            collected.append(
                {
                    "question": q.strip(),
                    "answer_turn_ids": [ids[0]],
                    "type": "single_hop",
                }
            )
            if len(collected) >= TARGET_QUESTIONS:
                break

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(collected, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"Wrote {len(collected)} single_hop questions to {OUT_PATH}")
    print(f"API calls used: {get_call_count()}")
    print()
    # Review each question against its source turn before trusting the set.
    for i, item in enumerate(collected, start=1):
        tid = item["answer_turn_ids"][0]
        turn = by_id[tid]
        print(f"[{i}] {item['question']}")
        print(f"    answer_turn_id: {tid}")
        print(f"    turn text: [{turn.speaker}] {turn.text}")
        print()


if __name__ == "__main__":
    main()
