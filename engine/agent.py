"""LangGraph routing agent: vector vs graph retrieval, then generate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from engine.corpus import load_full_corpus, turn_id
from engine.graph_store import find_entities, load_graph, local_search
from engine.llm import complete
from engine.retrieval import hybrid_search, load_index, rerank_with_threshold

ROOT = Path(__file__).resolve().parents[1]
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
PERSIST_DIR = ROOT / "data" / "chroma_db"
GRAPH_PATH = ROOT / "data" / "graph.json"
DATA_DIR = ROOT / "data"
UPLOADS_DIR = ROOT / "data" / "uploads"

_index = None
_graph = None
_turns = None
_app = None


class AgentState(TypedDict):
    question: str
    search_question: str
    history: list[dict]
    route: str
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
    """Drop cached corpus/index/graph so the next run reloads from disk."""
    global _index, _graph, _turns
    _index = None
    _graph = None
    _turns = None


def _ensure_stores() -> None:
    global _index, _graph, _turns
    if _turns is None:
        _turns = load_full_corpus(DATA_DIR, UPLOADS_DIR)
    if _index is None:
        _index = load_index(_turns, EMBED_MODEL, PERSIST_DIR)
    if _graph is None:
        _graph = load_graph(GRAPH_PATH)


def route_question(state: AgentState) -> AgentState:
    """Pick the store and rewrite the question to stand alone, in one call.

    Follow-ups like "who owns that?" are unsearchable on their own, so the
    router also resolves them against the history. Folding the rewrite into
    the routing call keeps the cost at two LLM calls per question.
    """
    question = state["question"]
    history_block = _history_block(state.get("history") or [], 3)

    prompt = (
        "Route this question to a retrieval store and rewrite it to stand "
        "alone.\n\n"
        "route=graph when answering needs more than one fact linked together: "
        "relationships, dependencies, blockers, ownership chains, knock-on "
        "effects, how one thing affects another, how something changed across "
        "weeks or meetings, or any question with two linked parts.\n"
        "route=vector when a single turn contains the answer: a date, the "
        "owner of one action item, one quoted status or decision.\n\n"
        "Examples:\n"
        '- "What is the agreed launch date?" -> vector (one stated fact)\n'
        '- "What action item did Bob commit to?" -> vector (one turn)\n'
        '- "What blocks launch and what else does that delay?" -> graph '
        "(blocker plus its downstream effects)\n"
        '- "How did the contract escalation change from week one to week '
        'three?" -> graph (spans meetings)\n\n'
        "standalone_question must resolve pronouns and references using the "
        "history so it can be searched on its own. If the question is already "
        "self-contained, repeat it unchanged.\n\n"
        f"Recent history (last 3 turns):\n{history_block}\n\n"
        f"Question: {question}\n\n"
        "Return JSON with these keys in this order: "
        '{"reasoning": "one short sentence on what answering requires", '
        '"route": "vector" or "graph", "standalone_question": "..."}'
    )
    raw = complete(
        prompt,
        system=(
            "You are a retrieval router. Respond with a single valid JSON "
            "object only, no markdown and no commentary."
        ),
        json_mode=True,
    )

    log = list(state.get("decision_log") or [])
    route_raw = ""
    standalone = ""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        log.append(f"route: router JSON invalid ({exc}); fell back to keyword match")
        data = None
    reasoning = ""
    if isinstance(data, dict):
        route_raw = str(data.get("route") or "")
        standalone = str(data.get("standalone_question") or "").strip()
        reasoning = str(data.get("reasoning") or "").strip()
    if not route_raw:
        # Keyword fallback over the raw text keeps routing working even if the
        # model ignores the JSON contract.
        route_raw = raw

    if "graph" in route_raw.strip().lower():
        route = "graph"
        reason = "Routed to graph for relational / multi-hop retrieval."
    else:
        route = "vector"
        reason = "Routed to vector for direct factual lookup."

    if not standalone:
        standalone = question

    log.append(f"route={route}: {reasoning or reason}")
    if standalone != question:
        log.append(f"rewrote question for retrieval: {standalone}")

    return {
        **state,
        "route": route,
        "search_question": standalone,
        "decision_log": log,
    }


def vector_search(state: AgentState) -> AgentState:
    _ensure_stores()
    assert _index is not None and _turns is not None
    question = state.get("search_question") or state["question"]
    hits = hybrid_search(_index, question, k=20)
    reranked = rerank_with_threshold(
        question,
        [tid for tid, _ in hits],
        _turns,
        top_n=5,
    )
    by_id = {turn_id(t): t for t in _turns}
    documents: list[str] = []
    for tid, _score in reranked:
        turn = by_id.get(tid)
        if turn is None:
            continue
        documents.append(
            f"[{tid}] line {turn.line_number} | {turn.speaker}: {turn.text}"
        )
    log = list(state.get("decision_log") or [])
    if documents:
        log.append(f"vector_search: retrieved {len(documents)} chunks above relevance floor")
    else:
        log.append(
            "vector_search: no chunk cleared the relevance floor; "
            "question looks outside the corpus"
        )
    return {**state, "documents": documents, "decision_log": log}


def graph_search(state: AgentState) -> AgentState:
    _ensure_stores()
    assert _graph is not None
    question = state.get("search_question") or state["question"]
    entities = find_entities(_graph, question)
    _sub, facts = local_search(_graph, entities, hops=2)
    log = list(state.get("decision_log") or [])
    log.append(
        f"graph_search: matched {len(entities)} entities, "
        f"returned {len(facts)} relation facts"
    )
    if facts:
        return {**state, "documents": facts, "decision_log": log}

    # The router sends a question to one store only. When the graph has no
    # path for it, answering "no evidence" would be wrong if the vector index
    # holds the answer, so fall back rather than decline.
    log.append("graph_search: no relation facts; falling back to vector retrieval")
    fallback = vector_search({**state, "decision_log": log})
    return {**fallback, "route": "graph->vector"}


def generate(state: AgentState) -> AgentState:
    docs = state.get("documents") or []
    if not docs:
        answer = (
            "I could not find supporting evidence in the retrieved documents, "
            "so I will not guess."
        )
        log = list(state.get("decision_log") or [])
        log.append("generate: no documents; declined to guess")
        return {**state, "answer": answer, "decision_log": log}

    doc_block = "\n".join(f"- {d}" for d in docs)
    history_block = _history_block(state.get("history") or [], 4)
    prompt = (
        "Answer the question using ONLY the documents below. "
        "Cite evidence inline as (meeting_id line N) when the document "
        "includes a meeting/line reference, or use the turn/citation already "
        "present in the document. If the documents are insufficient, say so.\n\n"
        "Use the conversation only to interpret what the question refers to; "
        "never treat it as evidence.\n\n"
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
    log = list(state.get("decision_log") or [])
    log.append("generate: answered from retrieved documents")
    return {**state, "answer": answer.strip(), "decision_log": log}


def _select_route(state: AgentState) -> Literal["vector_search", "graph_search"]:
    if state.get("route") == "graph":
        return "graph_search"
    return "vector_search"


def _build_app():
    graph = StateGraph(AgentState)
    graph.add_node("route_question", route_question)
    graph.add_node("vector_search", vector_search)
    graph.add_node("graph_search", graph_search)
    graph.add_node("generate", generate)

    graph.add_edge(START, "route_question")
    graph.add_conditional_edges(
        "route_question",
        _select_route,
        {
            "vector_search": "vector_search",
            "graph_search": "graph_search",
        },
    )
    graph.add_edge("vector_search", "generate")
    graph.add_edge("graph_search", "generate")
    graph.add_edge("generate", END)
    return graph.compile()


def run(question: str, history: list[dict] | None = None) -> dict:
    """Run the agent. Returns answer, documents, decision_log, route."""
    global _app
    _ensure_stores()
    if _app is None:
        _app = _build_app()

    final = _app.invoke(
        {
            "question": question,
            "search_question": "",
            "history": history or [],
            "route": "",
            "documents": [],
            "answer": "",
            "decision_log": [],
        }
    )
    return {
        "answer": final.get("answer", ""),
        "documents": final.get("documents") or [],
        "decision_log": final.get("decision_log") or [],
        "route": final.get("route", ""),
        "search_question": final.get("search_question", ""),
    }
