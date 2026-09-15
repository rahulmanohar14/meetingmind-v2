# Agent evals

The agent has one retrieval path: hybrid fusion to 20 candidates, cross-encoder rerank to 5, then the relevance floor. Full coverage means every turn id the question needs was retrieved; partial coverage is the mean fraction retrieved.

## Retrieval coverage (12 questions)

| arm | full coverage | mean partial coverage |
|---|---:|---:|
| hybrid + rerank, top-5 (agent path) | 12/12 | 1.00 |

## Abstention

The agent passes the generator only documents scoring at least -7.0 on the cross-encoder, and abstains without an API call when none clear it.

| question set | n | desired behaviour | correct |
|---|---:|---|---:|
| golden set (in corpus) | 12 | answer | 12/12 |
| natural paraphrases (eval/paraphrases.json) | 8 | answer | 8/8 |
| off-topic (eval/off_topic.json) | 7 | abstain | 7/7 |
