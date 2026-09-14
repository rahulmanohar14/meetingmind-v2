"""Verify engine.llm disk cache: two identical calls should make one API request."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.llm import complete, get_call_count

PROMPT = "Reply with exactly one word: ping"

complete(PROMPT)
complete(PROMPT)
print(get_call_count())
