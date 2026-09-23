"""Retrieval evaluation metrics."""

from __future__ import annotations

import math

from eval_harness.scorers.text_utils import tokenize


def _chunk_relevant(chunk: str, reference: str, *, min_overlap: float = 0.25) -> bool:
    ref_tokens = tokenize(reference)
    chunk_tokens = tokenize(chunk)
    if not ref_tokens or not chunk_tokens:
        return False
    overlap = len(ref_tokens & chunk_tokens) / len(ref_tokens)
    return overlap >= min_overlap


def _relevance_vector(retrieved: list[str], references: list[str]) -> list[float]:
    if not references:
        return [0.0] * len(retrieved)
    return [
        1.0 if any(_chunk_relevant(chunk, ref) for ref in references) else 0.0
        for chunk in retrieved
    ]


def recall_at_k(retrieved: list[str], references: list[str], k: int) -> float:
    """1.0 if any reference is hit in top-k retrieved chunks."""
    if not references:
        return 0.0
    top = retrieved[:k]
    for ref in references:
        if any(_chunk_relevant(chunk, ref) for chunk in top):
            return 1.0
    return 0.0


def mrr(retrieved: list[str], references: list[str]) -> float:
    """Reciprocal rank of the first relevant retrieved chunk."""
    if not references or not retrieved:
        return 0.0
    for rank, chunk in enumerate(retrieved, start=1):
        if any(_chunk_relevant(chunk, ref) for ref in references):
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], references: list[str], k: int) -> float:
    """Normalized DCG@k with binary relevance."""
    rel = _relevance_vector(retrieved[:k], references)
    if not rel or not any(rel):
        return 0.0
    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rel))
    ideal = sorted(rel, reverse=True)
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0
