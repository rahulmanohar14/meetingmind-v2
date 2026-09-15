"""LangGraph RAG agent: prepare the question, retrieve, generate.

Three nodes. Exactly two LLM calls on the answering path (rewrite, generate)
and one on the abstain path (rewrite only). The knowledge-graph routing layer
this replaced is preserved on the `main` branch as a recorded experiment;
neither store dominated the other, so the simpler one stays.

Nodes are thin wrappers: all retrieval lives in engine.retrieval.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from engine.corpus import load_full_corpus, turn_id
from engine.llm import complete
from engine.retrieval import hybrid_search, load_index, rerank_with_threshold

ROOT = Path(__file__).resolve().parents[1]
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
PERSIST_DIR = ROOT / "data" / "chroma_db"
DATA_DIR = ROOT / "data"
UPLOADS_DIR = ROOT / "data" / "uploads"

# Candidates fused by hybrid_search, then cut to TOP_N by the cross-encoder.
CANDIDATE_K = 20
TOP_N = 5

# Fixed string, not model output: the abstain path must cost zero API calls,
# and a generated refusal would vary run to run and defeat the eval.
ABSTAIN_ANSWER = (
    "I could not find supporting evidence in the retrieved documents, "
    "so I will not guess."
)

_index = None
_turns = None
_app = None


class AgentState(TypedDict):
    question: str
    history: list[dict]
    standalone_question: str
    documents: list[str]
    answer: str
    decision_log: list[str]


def _history_block(history: list[dict], limit: int) -> str:
    lines = []
    for turn in history[-limit:]:
        role = turn.get("role", "user")
        content = turn.get("content", "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines) if lines else "(none)"


def reload_stores() -> None:
    """Drop the cached corpus and index so the next run reloads from disk."""
    global _index, _turns
    _index = None
    _turns = None


def _ensure_stores() -> None:
    global _index, _turns
    if _turns is None:
        _turns = load_full_corpus(DATA_DIR, UPLOADS_DIR)
    if _index is None:
        _index = load_index(_turns, EMBED_MODEL, PERSIST_DIR)


def prepare_question(state: AgentState) -> AgentState:
    """Rewrite the question to stand alone. One LLM call.

    A follow-up like "who owns that?" carries no searchable terms, so retrieval
    on the raw text finds nothing. This resolves pronouns against the history
    before anything is embedded.

    `reasoning` is requested *before* `standalone_question` deliberately. The
    routing version of this node asked for its decision first and scored 0.40;
    moving the reasoning ahead of the decision took it to 0.94. The model
    commits to whatever it emits first, so the first field has to be thinking.
    """
    question = state["question"]
    history_block = _history_block(state.get("history") or [], 3)

    prompt = (
        "Rewrite the question below so it can be searched on its own against a "
        "meeting-transcript index.\n\n"
        "Resolve pronouns and back-references ('that', 'it', 'she', 'the same "
        "one') using the conversation history. Otherwise stay close to the "
        "original wording: do not answer the question, do not add facts, do "
        "not split it into several questions. If the question is already "
        "self-contained, repeat it unchanged.\n\n"
        f"Recent history (last 3 turns):\n{history_block}\n\n"
        f"Question: {question}\n\n"
        "Return JSON with these keys in this order: "
        '{"reasoning": "one short sentence on what the question refers to", '
        '"standalone_question": "..."}'
    )
    raw = complete(
        prompt,
        system=(
            "You rewrite meeting questions for a retrieval system. Respond "
            "with a single valid JSON object only, no markdown and no "
            "commentary."
        ),
        json_mode=True,
    )

    log = list(state.get("decision_log") or [])
    standalone = ""
    reasoning = ""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        # Not silent: recorded in the decision log the UI shows. Searching the
        # original question is a worse query but still a valid one, so this
        # degrades rather than fails.
        log.append(f"prepare_question: JSON invalid ({exc}); used question as-is")
        data = None
    if isinstance(data, dict):
        standalone = str(data.get("standalone_question") or "").strip()
        reasoning = str(data.get("reasoning") or "").strip()

    if not standalone:
        standalone = question

    if standalone != question:
        log.append(f"rewrote question for retrieval: {standalone}")
        if reasoning:
            log.append(f"rewrite reasoning: {reasoning}")
    else:
        log.append("question was already self-contained; searched as written")

    return {**state, "standalone_question": standalone, "decision_log": log}


def retrieve(state: AgentState) -> AgentState:
    """Hybrid search, cross-encoder rerank, relevance floor. No LLM call."""
    _ensure_stores()
    assert _index is not None and _turns is not None

    question = state.get("standalone_question") or state["question"]
    candidates = hybrid_search(_index, question, k=CANDIDATE_K)
    hits = rerank_with_threshold(
        question,
        [tid for tid, _ in candidates],
        _turns,
        top_n=TOP_N,
    )

    by_id = {turn_id(t): t for t in _turns}
    documents: list[str] = []
    for tid, _score in hits:
        turn = by_id.get(tid)
        if turn is None:
            continue
        documents.append(
            f"[{tid}] line {turn.line_number} | {turn.speaker}: {turn.text}"
        )

    log = list(state.get("decision_log") or [])
    if documents:
        log.append(
            f"retrieve: {len(candidates)} candidates -> {len(documents)} chunks "
            f"above the relevance floor (top score {hits[0][1]:.2f})"
        )
        return {**state, "documents": documents, "decision_log": log}

    # Nothing cleared the floor. The answer is set here rather than in a node
    # of its own so the graph can route straight to END without an LLM call;
    # generate() repeats the guard for callers that invoke it directly.
    log.append(
        "retrieve: no chunk cleared the relevance floor; "
        "question looks outside the corpus"
    )
    return {
        **state,
        "documents": [],
        "answer": ABSTAIN_ANSWER,
        "decision_log": log,
    }


def generate(state: AgentState) -> AgentState:
    """Answer from the retrieved documents only. One LLM call, or none."""
    docs = state.get("documents") or []
    log = list(state.get("decision_log") or [])
    if not docs:
        log.append("generate: no documents; declined to guess (0 API calls)")
        return {**state, "answer": ABSTAIN_ANSWER, "decision_log": log}

    doc_block = "\n".join(f"- {d}" for d in docs)
    history_block = _history_block(state.get("history") or [], 4)
    prompt = (
        "Answer the question using ONLY the documents below. "
        "Cite evidence inline as (meeting_id line N), taking both values from "
        "the document you are citing. If the documents are insufficient, say "
        "so instead of filling the gap.\n\n"
        "Use the conversation only to interpret what the question refers to; "
        "it is never evidence and must never be cited.\n\n"
        f"Conversation so far:\n{history_block}\n\n"
        f"Question: {state['question']}\n\n"
        f"Documents:\n{doc_block}\n"
    )
    answer = complete(
        prompt,
        system=(
            "You answer meeting questions strictly from provided documents. "
            "Never invent facts. Use inline citations."
        ),
    )
    log.append(f"generate: answered from {len(docs)} retrieved documents")
    return {**state, "answer": answer.strip(), "decision_log": log}


def _has_documents(state: AgentState) -> Literal["generate", "abstain"]:
    return "generate" if state.get("documents") else "abstain"


def _build_app():
    graph = StateGraph(AgentState)
    graph.add_node("prepare_question", prepare_question)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)

    graph.add_edge(START, "prepare_question")
    graph.add_edge("prepare_question", "retrieve")
    graph.add_conditional_edges(
        "retrieve",
        _has_documents,
        {"generate": "generate", "abstain": END},
    )
    graph.add_edge("generate", END)
    return graph.compile()


def run(question: str, history: list[dict] | None = None) -> dict:
    """Run the agent. Returns answer, documents, decision_log, standalone_question."""
    global _app
    _ensure_stores()
    if _app is None:
        _app = _build_app()

    final = _app.invoke(
        {
            "question": question,
            "history": history or [],
            "standalone_question": "",
            "documents": [],
            "answer": "",
            "decision_log": [],
        }
    )
    return {
        "answer": final.get("answer", ""),
        "documents": final.get("documents") or [],
        "decision_log": final.get("decision_log") or [],
        "standalone_question": final.get("standalone_question", ""),
    }
