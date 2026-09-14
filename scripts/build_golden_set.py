"""Generate a single-hop golden question set from the meeting corpus.

After this script runs, hand-append about 5 multi_hop questions to
eval/golden_set.json. Those entries use the same schema with
type="multi_hop" and answer_turn_ids that may span multiple meetings, e.g.:

    {
      "question": "...",
      "answer_turn_ids": ["meeting_week1:8", "meeting_week2:3"],
      "type": "multi_hop"
    }
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.corpus import Turn, load_corpus, turn_id
from engine.llm import complete, get_call_count

CORPUS_DIR = ROOT / "data"
OUT_PATH = ROOT / "eval" / "golden_set.json"
BATCH_SIZE = 20
TARGET_QUESTIONS = 12

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
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array from the model, got: {type(data)}")
    return data


def _index_turns(turns: list[Turn]) -> dict[str, Turn]:
    return {turn_id(t): t for t in turns}


def main() -> None:
    turns = load_corpus(CORPUS_DIR)
    if not turns:
        raise RuntimeError(f"No .txt/.vtt transcripts found in {CORPUS_DIR}")

    by_id = _index_turns(turns)
    collected: list[dict] = []
    valid_ids = set(by_id)

    # Spread batches across the corpus so questions are not all from one meeting.
    step = max(1, len(turns) // max(1, (TARGET_QUESTIONS // 3)))
    batch_starts = list(range(0, len(turns), step))[:4]

    for start in batch_starts:
        if len(collected) >= TARGET_QUESTIONS:
            break
        batch = turns[start : start + BATCH_SIZE]
        need = min(3, TARGET_QUESTIONS - len(collected))
        items = _ask_for_questions(batch, need)
        for item in items:
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
    # Hand-edit next: append ~5 multi_hop items (see module docstring).
    for i, item in enumerate(collected, start=1):
        tid = item["answer_turn_ids"][0]
        turn = by_id[tid]
        print(f"[{i}] {item['question']}")
        print(f"    answer_turn_id: {tid}")
        print(f"    turn text: [{turn.speaker}] {turn.text}")
        print()


if __name__ == "__main__":
    main()
