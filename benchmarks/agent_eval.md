# Agent evals

The agent has one retrieval path: hybrid fusion to 20 candidates, cross-encoder rerank to 5, then the relevance floor. Full coverage means every turn id the question needs was retrieved; partial coverage is the mean fraction retrieved.

## Retrieval coverage (28 questions)

| arm | full coverage | mean partial coverage |
|---|---:|---:|
| hybrid + rerank, top-5 (agent path) | 27/28 | 0.96 |

## Abstention

The agent passes the generator only documents scoring at least -6.0 on the cross-encoder, and abstains without an API call when none clear it.

| question set | n | desired behaviour | correct |
|---|---:|---|---:|
| golden set (in corpus) | 28 | answer | 28/28 |
| natural paraphrases (eval/paraphrases.json) | 8 | answer | 8/8 |
| off-topic (eval/off_topic.json) | 7 | abstain | 6/7 |

One off-topic question is answered rather than abstained on: "how many story points did we burn down last sprint?". It is arguably mislabeled, because the generated corpus discusses sprints throughout, so it now behaves as a hard negative rather than an off-topic question. It is kept in the fixture deliberately rather than removed after it failed.

## Finding: adjacent-turn misses are a chunking limit, not a ranking bug

The remaining coverage miss is the most informative result in this file, and it is a property of the chunk strategy rather than a defect in retrieval.

- **By what day must the response to the DataCorp renewal notice be sent?**
  - needs `2025-03-10_standup:30`
  - retrieved `meeting_week1:9, 2025-03-10_standup:29, meeting_week2:9, 2025-03-24_standup:8, 2025-05-08_orion_review:12`

In the DataCorp renewal case the retriever returns turn `:29`, where Tomas *asks* the question, but not `:30`, where Marcus *answers* it. Ranking is working: `:29` is the turn most similar to the query, because a question resembles a question. The answer is simply in the next turn.

One speaker turn is the whole chunk, and on this corpus that averages about fifteen words. A question and its answer are routinely split across two turns, so no amount of reranking recovers the second one: the information needed is not in the unit being scored.

**Highest-value next change: sentence-window retrieval.** Embed the single turn, so retrieval precision is unchanged, but return the matched turn plus its neighbours as the context handed to the model. Citations still point at the matched turn, so the exact-citation property survives. The case above is the measured example to test it against: a window of one either side would include `:30` and close this miss. Any such comparison has to report context length in tokens alongside recall, since a larger window raises recall trivially by including more text.
