# Agent evals

The agent has one retrieval path: hybrid fusion to 20 candidates, cross-encoder rerank to 5, then the relevance floor. Full coverage means every turn id the question needs was retrieved; partial coverage is the mean fraction retrieved.

## Retrieval coverage (28 questions)

| arm | full coverage | mean partial coverage |
|---|---:|---:|
| hybrid + rerank, top-5 (agent path) | 26/28 | 0.93 |

## Abstention

The agent passes the generator only documents scoring at least -6.0 on the cross-encoder, and abstains without an API call when none clear it.

| question set | n | desired behaviour | correct |
|---|---:|---|---:|
| golden set (in corpus) | 28 | answer | 28/28 |
| natural paraphrases (eval/paraphrases.json) | 8 | answer | 8/8 |
| off-topic (eval/off_topic.json) | 7 | abstain | 6/7 |
