# Retrieval benchmark

28 single-hop golden questions over 680 indexed turns.

## Retrieval configurations

| config | recall@5 | MRR@10 | mean latency (ms) |
|---|---:|---:|---:|
| hybrid | 0.929 | 0.787 | 35.1 |
| hybrid+rerank | 0.929 | 0.911 | 219.6 |
| bm25 | 0.893 | 0.759 | 1.8 |
| dense | 0.821 | 0.682 | 22.5 |

## rrf_k sensitivity (hybrid fusion)

| rrf_k | recall@5 | MRR@10 |
|---:|---:|---:|
| 10 | 0.893 | 0.786 |
| 20 | 0.893 | 0.786 |
| 60 (default) | 0.929 | 0.787 |
| 120 | 0.929 | 0.787 |

With n=28, one question is worth 0.036 of recall@5, so gaps of a single question are noise. Read these numbers as ruling out large regressions, not as fine-grained rankings.
