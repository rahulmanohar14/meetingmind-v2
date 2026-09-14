# Agent evals

Full coverage means every turn id the question needs was retrieved; partial coverage is the mean fraction retrieved. The vector arms mirror the agent's vector path (hybrid k=20 then cross-encoder rerank). The graph arm counts the turn ids cited by the relation facts that `graph_search` hands the generator.

## Multi-hop coverage (5 questions)

| arm | full coverage | mean partial coverage |
|---|---:|---:|
| vector, hybrid+rerank top-5 (agent path) | 3/5 | 0.80 |
| vector, hybrid+rerank top-10 (generous) | 4/5 | 0.90 |
| graph, 2-hop relation facts | 2/5 | 0.73 |

## Single-hop coverage (12 questions)

| arm | full coverage | mean partial coverage |
|---|---:|---:|
| vector, hybrid+rerank top-5 (agent path) | 12/12 | 1.00 |

## Abstention

The vector route passes the generator only documents scoring at least -7.0 on the cross-encoder, and abstains when none clear it. The value sits in the measured gap between natural in-corpus phrasings (down to -5.0 at top-1) and off-topic questions (up to -9.7).

| question set | n | desired behaviour | correct |
|---|---:|---|---:|
| golden set (in corpus) | 17 | answer | 17/17 |
| natural paraphrases (eval/paraphrases.json) | 8 | answer | 8/8 |
| off-topic (eval/off_topic.json) | 7 | abstain | 7/7 |

## Router accuracy (17 questions)

Expected route is derived from the golden set question type: `single_hop` -> vector, `multi_hop` -> graph.

| expected route | questions | correct | accuracy |
|---|---:|---:|---:|
| vector | 12 | 12 | 1.00 |
| graph | 5 | 4 | 0.80 |
| **overall** | 17 | 16 | **0.94** |

Sample sizes are small (12 single-hop, 5 multi-hop): one question moves single-hop metrics by 0.08 and multi-hop metrics by 0.20, so differences of a single question are not meaningful.
