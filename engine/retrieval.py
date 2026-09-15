"""Local hybrid retrieval: Chroma dense + BM25 + RRF + cross-encoder rerank.

No LLM calls in this module.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import chromadb
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from engine.corpus import Turn, turn_id

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_COLLECTION_NAME = "turns"
_RERANKER: CrossEncoder | None = None

# BGE v1.5 retrieval models are trained asymmetrically: queries carry this
# instruction, passages are embedded bare. Omitting it costs ranking quality.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_WORD = re.compile(r"\w+", re.UNICODE)

# Cross-encoder relevance floor. Recalibrated on the 680-turn corpus by
# scripts/calibrate_floor.py; rerun that script whenever the corpus changes,
# because this value is a property of the corpus, not of the model.
#
# Measured top-1 score populations:
#   golden set   n=28  -5.50 .. 9.92  (mean  3.56)
#   paraphrases  n=8   -5.00 .. 6.14  (mean  0.34)
#   off-topic    n=7  -11.26 .. -3.03 (mean -8.88)
#
# Golden-set questions are generated from the transcripts and score far higher
# than real phrasings, so they cannot set the lower bound alone; the
# paraphrases are what catch a floor set too high.
#
# These populations OVERLAP, unlike on the old 85-turn corpus: the weakest real
# question (-5.50) scores below the strongest off-topic one (-3.03), so no
# value separates them cleanly and the choice is an explicit trade. -6.0
# minimises total errors at 1/43 — it wrongly abstains on nothing, and wrongly
# answers one off-topic question ("how many story points did we burn down last
# sprint?"), which this corpus arguably does discuss, since sprints are talked
# about constantly. A wrong abstention is a visibly broken product; a wrong
# answer is at least cited and checkable.
RELEVANCE_FLOOR = -6.0


@dataclass
class Index:
    collection: chromadb.Collection
    bm25: BM25Okapi
    turns: list[Turn]
    turn_ids: list[str]
    embed_model: SentenceTransformer
    embed_model_name: str
    persist_dir: str


def build_index(
    turns: list[Turn],
    embed_model_name: str,
    persist_dir: str | Path,
) -> Index:
    persist_dir = Path(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(persist_dir))
    existing = {c.name for c in client.list_collections()}
    if _COLLECTION_NAME in existing:
        client.delete_collection(_COLLECTION_NAME)
    collection = client.create_collection(
        name=_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    embed_model = SentenceTransformer(embed_model_name)
    texts = [t.text for t in turns]
    ids = [turn_id(t) for t in turns]
    embeddings = embed_model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=[
            {
                "meeting_id": t.meeting_id,
                "turn_index": t.turn_index,
                "speaker": t.speaker,
                "line_number": t.line_number,
            }
            for t in turns
        ],
    )

    tokenized = [_tokenize(text) for text in texts]
    bm25 = BM25Okapi(tokenized)

    return Index(
        collection=collection,
        bm25=bm25,
        turns=list(turns),
        turn_ids=ids,
        embed_model=embed_model,
        embed_model_name=embed_model_name,
        persist_dir=str(persist_dir),
    )


def load_index(
    turns: list[Turn],
    embed_model_name: str,
    persist_dir: str | Path,
) -> Index:
    """Open an existing Chroma collection and rebuild the in-memory BM25 index."""
    persist_dir = Path(persist_dir)
    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_collection(_COLLECTION_NAME)
    embed_model = SentenceTransformer(embed_model_name)
    ids = [turn_id(t) for t in turns]
    bm25 = BM25Okapi([_tokenize(t.text) for t in turns])
    return Index(
        collection=collection,
        bm25=bm25,
        turns=list(turns),
        turn_ids=ids,
        embed_model=embed_model,
        embed_model_name=embed_model_name,
        persist_dir=str(persist_dir),
    )


def _embed_query_text(index: Index, query: str) -> str:
    if "bge" in index.embed_model_name.lower():
        return BGE_QUERY_PREFIX + query
    return query


def dense_search(index: Index, query: str, k: int) -> list[tuple[str, float]]:
    if k <= 0 or not index.turn_ids:
        return []
    n = min(k, len(index.turn_ids))
    q_emb = index.embed_model.encode(
        [_embed_query_text(index, query)],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    result = index.collection.query(
        query_embeddings=np.asarray(q_emb, dtype=np.float32).tolist(),
        n_results=n,
        include=["distances"],
    )
    ids = (result.get("ids") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]
    # Cosine space: Chroma distance is 1 - cosine similarity.
    scored = [(tid, float(1.0 - dist)) for tid, dist in zip(ids, distances)]
    return scored


def bm25_search(index: Index, query: str, k: int) -> list[tuple[str, float]]:
    if k <= 0 or not index.turn_ids:
        return []
    scores = index.bm25.get_scores(_tokenize(query))
    ranked = sorted(
        zip(index.turn_ids, scores),
        key=lambda item: item[1],
        reverse=True,
    )
    return [(tid, float(score)) for tid, score in ranked[:k]]


def hybrid_search(
    index: Index,
    query: str,
    k: int,
    rrf_k: int = 60,
) -> list[tuple[str, float]]:
    if k <= 0 or not index.turn_ids:
        return []
    candidate_k = min(50, len(index.turn_ids))
    dense = dense_search(index, query, candidate_k)
    sparse = bm25_search(index, query, candidate_k)

    fused: dict[str, float] = {}
    for rank, (tid, _) in enumerate(dense, start=1):
        fused[tid] = fused.get(tid, 0.0) + 1.0 / (rrf_k + rank)
    for rank, (tid, _) in enumerate(sparse, start=1):
        fused[tid] = fused.get(tid, 0.0) + 1.0 / (rrf_k + rank)

    ranked = sorted(fused.items(), key=lambda item: item[1], reverse=True)
    return ranked[:k]


def rerank(
    query: str,
    turn_ids: list[str],
    turns: list[Turn],
    top_n: int,
) -> list[tuple[str, float]]:
    if top_n <= 0 or not turn_ids:
        return []
    by_id = {turn_id(t): t for t in turns}
    pairs: list[list[str]] = []
    valid_ids: list[str] = []
    for tid in turn_ids:
        turn = by_id.get(tid)
        if turn is None:
            continue
        pairs.append([query, turn.text])
        valid_ids.append(tid)
    if not pairs:
        return []

    model = _get_reranker()
    scores = model.predict(pairs)
    ranked = sorted(
        zip(valid_ids, scores),
        key=lambda item: float(item[1]),
        reverse=True,
    )
    return [(tid, float(score)) for tid, score in ranked[:top_n]]


def rerank_with_threshold(
    query: str,
    turn_ids: list[str],
    turns: list[Turn],
    top_n: int,
) -> list[tuple[str, float]]:
    """Rerank, then keep only documents above RELEVANCE_FLOOR.

    Returns [] when nothing clears the floor, which lets the agent decline
    instead of answering from noise. Without this, hybrid_search returns `k`
    documents for any query at all, so the agent's "no supporting evidence"
    path was unreachable.

    Dropping the sub-floor tail also matters when the answer *is* present: the
    launch-date question retrieves one document at 6.7 and four below -9, and
    passing those four to the generator is pure distraction.
    """
    ranked = rerank(query, turn_ids, turns, top_n)
    return [(tid, score) for tid, score in ranked if score >= RELEVANCE_FLOOR]


def _get_reranker() -> CrossEncoder:
    global _RERANKER
    if _RERANKER is None:
        _RERANKER = CrossEncoder(RERANKER_MODEL)
    return _RERANKER


def _tokenize(text: str) -> list[str]:
    """Word tokens for BM25.

    `str.split()` leaves punctuation attached, so "keys?" and "keys" were
    different terms and query punctuation silently lost matches.
    """
    return _WORD.findall(text.lower())
