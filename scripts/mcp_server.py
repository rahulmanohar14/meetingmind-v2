"""MeetingMind MCP server — exposes the same engine as Streamlit to MCP clients.

Run (stdio):
  python scripts/mcp_server.py

Configure Cursor / Claude Desktop to spawn this script with the project venv.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mcp.server.mcpserver import MCPServer

from engine.agent import run
from engine.corpus import load_full_corpus, turn_id
from engine.db import list_meetings, seed_demo_meetings
from engine.retrieval import hybrid_search, load_index

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
DATA_DIR = ROOT / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
PERSIST_DIR = DATA_DIR / "chroma_db"

mcp = MCPServer(
    "meetingmind",
    instructions=(
        "MeetingMind answers questions over meeting transcripts using hybrid "
        "retrieval with a cross-encoder rerank. Prefer ask_meeting for Q&A."
    ),
)


@mcp.tool(
    description=(
        "Ask a question over the indexed meetings. Answers are grounded in "
        "retrieved turns and cited; abstains when nothing relevant is found."
    )
)
def ask_meeting(question: str) -> str:
    result = run(question)
    return json.dumps(
        {
            "answer": result.get("answer", ""),
            "standalone_question": result.get("standalone_question", ""),
            "decision_log": result.get("decision_log") or [],
            "documents": result.get("documents") or [],
        },
        ensure_ascii=False,
        indent=2,
    )


@mcp.tool(
    name="list_meetings",
    description="List demo and uploaded meetings in the corpus.",
)
def list_meetings_tool() -> str:
    seed_demo_meetings(DATA_DIR)
    rows = list_meetings()
    return json.dumps(rows, ensure_ascii=False, indent=2)


@mcp.tool(description="Hybrid search over meeting turns (no LLM). Returns top snippets.")
def search_meetings(query: str, k: int = 5) -> str:
    turns = load_full_corpus(DATA_DIR, UPLOADS_DIR)
    index = load_index(turns, EMBED_MODEL, PERSIST_DIR)
    hits = hybrid_search(index, query, k=max(1, min(k, 20)))
    by_id = {turn_id(t): t for t in turns}
    snippets = []
    for tid, score in hits:
        turn = by_id.get(tid)
        if turn is None:
            continue
        snippets.append(
            {
                "turn_id": tid,
                "score": score,
                "speaker": turn.speaker,
                "line_number": turn.line_number,
                "text": turn.text,
            }
        )
    return json.dumps(snippets, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    mcp.run(transport="stdio")
