# Retrieval benchmark

28 single-hop golden questions over 680 indexed turns.

## Retrieval configurations

| config | recall@5 | MRR@10 | mean latency (ms) |
|---|---:|---:|---:|
| hybrid | 0.964 | 0.805 | 14.7 |
| hybrid+rerank | 0.964 | 0.929 | 139.8 |
| bm25 | 0.929 | 0.795 | 0.7 |
| dense | 0.857 | 0.689 | 14.1 |

## rrf_k sensitivity (hybrid fusion)

| rrf_k | recall@5 | MRR@10 |
|---:|---:|---:|
| 10 | 0.929 | 0.804 |
| 20 | 0.929 | 0.804 |
| 60 (default) | 0.964 | 0.805 |
| 120 | 0.964 | 0.805 |

With n=28, one question is worth 0.036 of recall@5, so gaps of a single question are noise. Read these numbers as ruling out large regressions, not as fine-grained rankings.
