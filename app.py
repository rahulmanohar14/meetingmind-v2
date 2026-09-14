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
from engine.corpus import load_full_corpus
from engine.db import list_meetings, seed_demo_meetings
from engine.graph_store import load_graph
from engine.ingest import ingest_upload, reset_uploads, save_upload

_VECTOR_DOC = re.compile(
    r"^\[(?P<meeting>[^:\]]+):\d+\]\s+line\s+(?P<line>\d+)\s+\|\s+(?P<speaker>[^:]+):"
)


@st.cache_resource
def _sidebar_stats() -> tuple[int, int, int]:
    turns = load_full_corpus(ROOT / "data", ROOT / "data" / "uploads")
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
    route = message.get("route") or ""
    if route:
        st.caption(f"Routed to **{route}**")
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


def _render_sidebar() -> None:
    seed_demo_meetings(ROOT / "data")
    turn_count, node_count, edge_count = _sidebar_stats()
    st.title("MeetingMind")
    st.write(f"Indexed turns: {turn_count}")
    st.write(f"Graph nodes: {node_count}")
    st.write(f"Graph edges: {edge_count}")

    st.divider()
    st.subheader("Upload")
    uploaded = st.file_uploader(
        "Transcript (.txt or .vtt)",
        type=["txt", "vtt"],
        accept_multiple_files=False,
    )
    if st.button("Ingest", disabled=uploaded is None):
        status = st.empty()
        try:
            path = save_upload(uploaded.name, uploaded.getvalue())
            status.info("saving")
            result = ingest_upload(
                path,
                status_callback=lambda msg: status.info(msg),
            )
            st.cache_resource.clear()
            status.success(
                f"Ingested {result['meeting_id']}: "
                f"{result['new_turns']} new turns, "
                f"{result['indexed_turns']} indexed, "
                f"{result['nodes']} nodes / {result['edges']} edges"
            )
            st.rerun()
        except Exception as exc:
            status.error(f"Ingest failed: {exc}")

    uploads = list_meetings(source="upload")
    if uploads:
        st.caption("Uploaded meetings")
        for row in uploads:
            st.write(f"- {row['meeting_id']} ({row['turn_count']} turns)")

    if st.button("Reset uploads"):
        status = st.empty()
        try:
            result = reset_uploads(status_callback=lambda msg: status.info(msg))
            st.cache_resource.clear()
            status.success(
                f"Reset done: {result['indexed_turns']} turns, "
                f"{result['nodes']} nodes / {result['edges']} edges"
            )
            st.rerun()
        except Exception as exc:
            status.error(f"Reset failed: {exc}")

    st.divider()
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

    results_path = ROOT / "benchmarks" / "results.md"
    agent_eval_path = ROOT / "benchmarks" / "agent_eval.md"
    with st.expander("Evals", expanded=False):
        if results_path.exists():
            st.markdown(results_path.read_text(encoding="utf-8"))
        else:
            st.write("Run `python scripts/benchmark.py` to generate results.")
        st.divider()
        if agent_eval_path.exists():
            st.markdown(agent_eval_path.read_text(encoding="utf-8"))
        else:
            st.write("Run `python scripts/eval_agent.py` for agent-level evals.")


def main() -> None:
    st.set_page_config(page_title="MeetingMind", page_icon=None, layout="centered")

    with st.sidebar:
        _render_sidebar()

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
        {
            "role": "user",
            "content": prompt,
            "documents": [],
            "decision_log": [],
            "route": "",
        }
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
            "route": result.get("route") or "",
        }
        _render_assistant(assistant_msg)
    st.session_state.messages.append(assistant_msg)


if __name__ == "__main__":
    main()
