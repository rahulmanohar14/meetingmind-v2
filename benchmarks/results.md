# Retrieval benchmark

12 single-hop golden questions. The 5 multi-hop questions are evaluated separately in `agent_eval.md`, because they need relation traversal rather than ranked chunks.

## Retrieval configurations

| config | recall@5 | MRR@10 | mean latency (ms) |
|---|---:|---:|---:|
| dense | 1.000 | 0.892 | 18.8 |
| hybrid+rerank | 1.000 | 0.917 | 194.3 |
| bm25 | 0.917 | 0.883 | 0.1 |
| hybrid | 0.917 | 0.885 | 17.6 |

## rrf_k sensitivity (hybrid fusion)

| rrf_k | recall@5 | MRR@10 |
|---:|---:|---:|
| 10 | 0.917 | 0.885 |
| 20 | 0.917 | 0.885 |
| 60 (default) | 0.917 | 0.885 |
| 120 | 0.917 | 0.885 |

With n=12, one question is worth 0.083 of recall@5, so gaps of a single question are noise. Read these numbers as ruling out large regressions, not as fine-grained rankings.
