"""Smoke test: load data/ corpus and print turn counts."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.corpus import load_demo_corpus, turn_id

turns = load_demo_corpus(ROOT / "data")
print(f"total turns: {len(turns)}")

by_meeting = Counter(t.meeting_id for t in turns)
print("per-meeting turn counts:")
for meeting_id, count in sorted(by_meeting.items()):
    print(f"  {meeting_id}: {count}")

print("\nfirst three turns:")
for turn in turns[:3]:
    print(f"--- {turn_id(turn)} ---")
    print(f"speaker: {turn.speaker}")
    print(f"line_number: {turn.line_number}")
    print(f"text: {turn.text}")
    print()
