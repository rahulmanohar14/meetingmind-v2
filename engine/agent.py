"""LangGraph routing agent: vector vs graph retrieval, then generate."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from engine.corpus import load_corpus, turn_id
from engine.graph_store import find_entities, load_graph, local_search
from engine.llm import complete
from engine.retrieval import hybrid_search, load_index, rerank

ROOT = Path(__file__).resolve().parents[1]
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
PERSIST_DIR = ROOT / "data" / "chroma_db"
GRAPH_PATH = ROOT / "data" / "graph.json"

_index = None
_graph = None
_turns = None
_app = None


class AgentState(TypedDict):
    question: str
    history: list[dict]
    route: str
    documents: list[str]
    answer: str
    decision_log: list[str]


def _ensure_stores() -> None:
    global _index, _graph, _turns
    if _turns is None:
        _turns = load_corpus(ROOT / "data")
    if _index is None:
        _index = load_index(_turns, EMBED_MODEL, PERSIST_DIR)
    if _graph is None:
        _graph = load_graph(GRAPH_PATH)


def route_question(state: AgentState) -> AgentState:
    history = state.get("history") or []
    recent = history[-3:]
    history_lines = []
    for turn in recent:
        role = turn.get("role", "user")
        content = turn.get("content", "")
        history_lines.append(f"{role}: {content}")
    history_block = "\n".join(history_lines) if history_lines else "(none)"

    prompt = (
        "Choose the retrieval store for this question.\n"
        "Return exactly one word: vector OR graph.\n\n"
        "Choose graph for: relationships, dependencies, blockers, ownership "
        "chains, knock-on effects, or anything spanning multiple meetings.\n"
        "Choose vector for: direct factual lookup of what was said in a "
        "specific turn (dates, owners of a single action, quoted status).\n\n"
        f"Recent history (last 3 turns):\n{history_block}\n\n"
        f"Question: {state['question']}\n"
    )
    raw = complete(
        prompt,
        system=(
            "You are a retrieval router. Reply with only the word vector "
            "or the word graph."
        ),
    )
    token = raw.strip().lower().split()[0] if raw.strip() else ""
    if "graph" in token:
        route = "graph"
        reason = "Routed to graph for relational / multi-hop retrieval."
    else:
        route = "vector"
        reason = "Routed to vector for direct factual lookup."

    log = list(state.get("decision_log") or [])
    log.append(f"route={route}: {reason}")
    return {
        **state,
        "route": route,
        "decision_log": log,
    }


def vector_search(state: AgentState) -> AgentState:
    _ensure_stores()
    assert _index is not None and _turns is not None
    hits = hybrid_search(_index, state["question"], k=20)
    reranked = rerank(
        state["question"],
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
    log.append(f"vector_search: retrieved {len(documents)} chunks")
    return {**state, "documents": documents, "decision_log": log}


def graph_search(state: AgentState) -> AgentState:
    _ensure_stores()
    assert _graph is not None
    entities = find_entities(_graph, state["question"])
    _sub, facts = local_search(_graph, entities, hops=2)
    log = list(state.get("decision_log") or [])
    log.append(
        f"graph_search: matched {len(entities)} entities, "
        f"returned {len(facts)} relation facts"
    )
    return {**state, "documents": facts, "decision_log": log}


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
    prompt = (
        "Answer the question using ONLY the documents below. "
        "Cite evidence inline as (meeting_id line N) when the document "
        "includes a meeting/line reference, or use the turn/citation already "
        "present in the document. If the documents are insufficient, say so.\n\n"
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
    }
