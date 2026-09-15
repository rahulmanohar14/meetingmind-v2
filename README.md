# MeetingMind v2

> Note: this branch (`rag-only`) is mid-refactor. The architecture section below
> still describes the graph version and is stale. Stage 8 rewrites it.

MeetingMind answers questions over meeting transcripts using two stores: a hybrid
vector index for factual lookup, and a typed knowledge graph for multi-hop
relationship questions (blockers, dependencies, ownership chains). A LangGraph
agent routes each question to the fitting store and records that choice in a
visible decision log. The same engine is also exposed as an MCP server for
Cursor or Claude Desktop.

## Architecture

- `engine/corpus.py` — transcript loading (`SPEAKER: text` and `.vtt`)
- `engine/retrieval.py` — Chroma dense search, BM25, RRF hybrid fusion, cross-encoder rerank
- `engine/extract.py` / `engine/graph_store.py` — LLM entity extraction and NetworkX graph
- `engine/ingest.py` / `engine/db.py` — upload ingest, SQLite meeting registry, reset
- `engine/agent.py` — LangGraph router (`vector` vs `graph`), then generate-with-citations.
  The router also rewrites follow-ups into standalone questions in the same call, the
  graph route falls back to vector when it finds no relation path, and the vector route
  abstains when no chunk clears a calibrated relevance floor.
- `engine/llm.py` — sole Groq entry point (disk cache, multi-key fallback)
- `app.py` — Streamlit chat UI (upload, decision log, evals)
- `scripts/mcp_server.py` — MCP tools: `ask_meeting`, `list_meetings`, `search_meetings`

## Setup

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows Git Bash
pip install -r requirements-v2.txt
```

Create a `.env` file at the repo root with one or more Groq keys (comma-separated).
On 429/quota the client rotates to the next key:

```
GROQ_API_KEYS=key1,key2,key3
GROQ_MODEL=openai/gpt-oss-20b
```

## Run order

Build artifacts first, then start the UI:

```bash
python scripts/build_index.py
python scripts/build_graph.py
python scripts/benchmark.py
python scripts/eval_agent.py
streamlit run app.py
```

`benchmark.py` scores retrieval configurations on the single-hop questions.
`eval_agent.py` scores the parts a retrieval benchmark cannot see: multi-hop
coverage of the graph against vector search, abstention on off-topic questions,
and router accuracy.

In the sidebar you can upload a `.txt` / `.vtt` transcript (Ingest), reset uploads
back to the demo corpus, clear chat, and open the evals table. Uploads append to
the shared index and graph; Reset uploads restores `data/graph_baseline.json`
and rebuilds the index without extra API calls.

Optional checks:

```bash
python scripts/test_corpus.py
python scripts/test_llm.py
python scripts/test_graph.py
python scripts/test_agent.py
```

## MCP server

Same engine as Streamlit, callable from Cursor or Claude Desktop:

```bash
python scripts/mcp_server.py
```

### Cursor

Add to Cursor MCP settings (paths for this repo on Windows):

```json
{
  "mcpServers": {
    "meetingmind": {
      "command": "C:\\dev\\meetingmind-v2\\.venv\\Scripts\\python.exe",
      "args": ["C:\\dev\\meetingmind-v2\\scripts\\mcp_server.py"],
      "cwd": "C:\\dev\\meetingmind-v2"
    }
  }
}
```

### Claude Desktop

In `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "meetingmind": {
      "command": "C:\\dev\\meetingmind-v2\\.venv\\Scripts\\python.exe",
      "args": ["C:\\dev\\meetingmind-v2\\scripts\\mcp_server.py"],
      "cwd": "C:\\dev\\meetingmind-v2"
    }
  }
}
```

Tools: `ask_meeting`, `list_meetings`, `search_meetings`. Available in any
chat once the server is configured and the client has restarted.

A tiny sample upload for demos lives at `examples/sample_upload_orion.txt`.

## Benchmark results

From `benchmarks/results.md`, 12 single-hop questions. Multi-hop questions are
scored separately in `benchmarks/agent_eval.md`, because recall over ranked
chunks is the wrong metric for them:

| config | recall@5 | MRR@10 | mean latency (ms) |
|---|---:|---:|---:|
| dense | 1.000 | 0.892 | 18.8 |
| hybrid+rerank | 1.000 | 0.917 | 194.3 |
| bm25 | 0.917 | 0.883 | 0.1 |
| hybrid | 0.917 | 0.885 | 17.6 |

Latencies vary between runs; `benchmarks/results.md` holds the numbers from the
last local run.

Two findings worth stating plainly. First, `rrf_k` swept over 10/20/60/120 made
no difference on this corpus, so the paper default of 60 stands for lack of a
reason to change it, not because it was tuned. Second, dense alone already
reaches recall@5 1.000 here, so fusion earns its place only through MRR@10
(0.917 for hybrid+rerank against 0.892 for dense) — on 85 turns this corpus is
too small to show what hybrid retrieval is for.

With n=12, one question is worth 0.083 of recall@5. These numbers rule out
large regressions; they do not finely rank configurations.

## Agent results

From `benchmarks/agent_eval.md`. Full coverage means every turn the question
needs was retrieved:

| multi-hop arm (5 questions) | full coverage | mean partial |
|---|---:|---:|
| vector, hybrid+rerank top-5 | 3/5 | 0.80 |
| vector, hybrid+rerank top-10 | 4/5 | 0.90 |
| graph, 2-hop relation facts | 2/5 | 0.73 |

Neither store dominates, which is the case for routing rather than picking one.
Vector wins two questions the graph misses; the graph fully answers *"What beta
launch date did the team keep in week one, and what fallback release plan did
they adopt if blockers remain?"*, which vector cannot cover at top-5 or top-10
because the two halves of the answer never appear near each other.

| behaviour | result |
|---|---|
| router accuracy (17 questions) | 0.94 overall, 12/12 vector, 4/5 graph |
| answers golden-set questions | 17/17 |
| answers natural paraphrases | 8/8 |
| abstains on off-topic questions | 7/7 |

Abstention uses the cross-encoder score rather than a guessed constant. The
vector route drops every document below -7.0 and declines when none survive.
That value comes from the measured gap between natural in-corpus phrasings,
which bottom out near -5.0 at top-1, and off-topic questions, which peak near
-9.7.

Calibrating on the golden set alone was a trap worth recording: those questions
are generated from the transcripts and score 2.0 to 10.1, which suggested a
floor of 0.0 — and that floor silently refused real questions like "What did
Alice decide about beta seats?". `eval/paraphrases.json` exists to keep that
regression caught.

## What did not work

The extractor names entities as fragments of speech, so `DataCorp vendor
contract is still unsigned`, `DataCorp dependency` and `DataCorp slips further`
become three nodes for one blocker — 35 Blocker nodes for roughly six real
blockers. The obvious fix is to make the extraction prompt demand short
canonical names, so it was built and A/B tested against the current graph on
multi-hop coverage:

| graph | nodes | edges | multi-hop full | mean partial |
|---|---:|---:|---:|---:|
| current (verbose names) | 84 | 45 | 2/5 | 0.73 |
| canonical names, 10-turn batches | 72 | 13 | 1/5 | 0.30 |
| canonical names, 15-turn batches | 38 | 1 | 0/5 | 0.10 |

Canonicalization worked — Blocker nodes fell from 35 to 16 and `accessibility
audit` collected 4 source turns instead of being split across variants. But
adding naming rules to the prompt crowded out relation extraction, and edges
collapsed from 45 to 13 (to 1 at the larger batch size). A graph with clean
node names and no edges answers nothing, so the change was reverted.

The seeding fix in `find_entities` gets the same benefit from the other side:
IDF-weighted token overlap matches the verbose names without needing them to be
canonical. Fixing entity resolution properly needs a merge step after
extraction rather than instructions during it.
