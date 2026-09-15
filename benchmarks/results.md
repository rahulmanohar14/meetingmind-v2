# Retrieval benchmark

12 single-hop golden questions over 85 indexed turns.

## Retrieval configurations

| config | recall@5 | MRR@10 | mean latency (ms) |
|---|---:|---:|---:|
| dense | 1.000 | 0.892 | 18.2 |
| hybrid+rerank | 1.000 | 0.917 | 357.6 |
| bm25 | 0.917 | 0.883 | 0.2 |
| hybrid | 0.917 | 0.885 | 16.0 |

## rrf_k sensitivity (hybrid fusion)

| rrf_k | recall@5 | MRR@10 |
|---:|---:|---:|
| 10 | 0.917 | 0.885 |
| 20 | 0.917 | 0.885 |
| 60 (default) | 0.917 | 0.885 |
| 120 | 0.917 | 0.885 |

With n=12, one question is worth 0.083 of recall@5, so gaps of a single question are noise. Read these numbers as ruling out large regressions, not as fine-grained rankings.
