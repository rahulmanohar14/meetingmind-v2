# MeetingMind v2

MeetingMind answers questions over meeting transcripts using two stores: a hybrid
vector index for factual lookup, and a typed knowledge graph for multi-hop
relationship questions (blockers, dependencies, ownership chains). A LangGraph
agent routes each question to the fitting store and records that choice in a
visible decision log.

## Architecture

- `engine/corpus.py` — transcript loading (`SPEAKER: text` and `.vtt`)
- `engine/retrieval.py` — Chroma dense search, BM25, RRF hybrid fusion, cross-encoder rerank
- `engine/extract.py` / `engine/graph_store.py` — LLM entity extraction and NetworkX graph
- `engine/agent.py` — LangGraph router (`vector` vs `graph`) then generate-with-citations
- `engine/llm.py` — sole Gemini entry point (disk cache, rate limit, backoff)
- `app.py` — Streamlit chat UI

## Setup

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows Git Bash
pip install -r requirements-v2.txt
```

Create a `.env` file at the repo root:

```
GEMINI_API_KEY=your_key_here
```

## Run order

Build artifacts first, then start the UI:

```bash
python scripts/build_index.py
python scripts/build_graph.py
python scripts/benchmark.py
streamlit run app.py
```

Optional checks:

```bash
python scripts/test_corpus.py
python scripts/test_llm.py
python scripts/test_graph.py
python scripts/test_agent.py
```

`scripts/build_golden_set.py` regenerates single-hop eval questions; hand-append
`multi_hop` entries to `eval/golden_set.json` afterward.

## Benchmark results

From `benchmarks/results.md` (12 single-hop questions; multi-hop excluded
because vector retrieval cannot answer them):

| config | recall@5 | MRR@10 | mean latency (ms) |
|---|---:|---:|---:|
| dense | 1.000 | 0.892 | 53.6 |
| hybrid+rerank | 1.000 | 0.917 | 1713.3 |
| hybrid | 0.917 | 0.844 | 66.2 |
| bm25 | 0.833 | 0.806 | 0.4 |
