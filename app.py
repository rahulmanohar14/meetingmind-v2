"""MeetingMind v2 Streamlit chat UI."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.agent import run
from engine.corpus import load_corpus
from engine.graph_store import load_graph

_VECTOR_DOC = re.compile(
    r"^\[(?P<meeting>[^:\]]+):\d+\]\s+line\s+(?P<line>\d+)\s+\|\s+(?P<speaker>[^:]+):"
)


@st.cache_resource
def _sidebar_stats() -> tuple[int, int, int]:
    turns = load_corpus(ROOT / "data")
    graph = load_graph(ROOT / "data" / "graph.json")
    return len(turns), graph.number_of_nodes(), graph.number_of_edges()


def _history_pairs(messages: list[dict]) -> list[dict]:
    """Last 3 user/assistant message pairs for the agent."""
    history = [{"role": m["role"], "content": m["content"]} for m in messages]
    return history[-6:]


def _format_source(doc: str) -> str:
    match = _VECTOR_DOC.match(doc)
    if match:
        return (
            f"{match.group('meeting')} line {match.group('line')}, "
            f"{match.group('speaker').strip()}"
        )
    return doc


def _render_assistant(message: dict) -> None:
    st.markdown(message["content"])
    documents = message.get("documents") or []
    if documents:
        st.markdown("**Sources**")
        for doc in documents:
            st.markdown(f"- {_format_source(doc)}")
    with st.expander("Decision log", expanded=False):
        log = message.get("decision_log") or []
        if not log:
            st.write("(empty)")
        else:
            for entry in log:
                st.markdown(f"- {entry}")


def main() -> None:
    st.set_page_config(page_title="MeetingMind", page_icon=None, layout="centered")

    turn_count, node_count, edge_count = _sidebar_stats()
    with st.sidebar:
        st.title("MeetingMind")
        st.write(f"Indexed turns: {turn_count}")
        st.write(f"Graph nodes: {node_count}")
        st.write(f"Graph edges: {edge_count}")

    st.header("MeetingMind")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                _render_assistant(message)
            else:
                st.markdown(message["content"])

    prompt = st.chat_input("Ask about the meetings")
    if not prompt:
        return

    st.session_state.messages.append(
        {"role": "user", "content": prompt, "documents": [], "decision_log": []}
    )
    with st.chat_message("user"):
        st.markdown(prompt)

    history = _history_pairs(st.session_state.messages[:-1])
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = run(prompt, history=history)
        assistant_msg = {
            "role": "assistant",
            "content": result["answer"],
            "documents": result.get("documents") or [],
            "decision_log": result.get("decision_log") or [],
        }
        _render_assistant(assistant_msg)
    st.session_state.messages.append(assistant_msg)


if __name__ == "__main__":
    main()
