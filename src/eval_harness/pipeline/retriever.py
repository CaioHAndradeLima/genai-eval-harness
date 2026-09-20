"""Retrieval abstractions and implementations."""

from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from collections import Counter
from pathlib import Path

from eval_harness.config import resolve_path
from eval_harness.models import RetrieverConfig, RetrieverType
from eval_harness.pipeline.chunking import chunk_text


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class Retriever(ABC):
    """Abstract retriever."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int | None = None) -> list[str]:
        """Return ranked document chunks for a query."""


class NullRetriever(Retriever):
    """No retrieval — for pure generation tasks."""

    def retrieve(self, query: str, top_k: int | None = None) -> list[str]:
        return []


class BM25Retriever(Retriever):
    """BM25 lexical retriever over a chunked corpus."""

    def __init__(self, chunks: list[str], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self._tokenized = [_tokenize(chunk) for chunk in chunks]
        self._doc_freq: Counter[str] = Counter()
        for tokens in self._tokenized:
            self._doc_freq.update(set(tokens))
        self._avg_dl = sum(len(tokens) for tokens in self._tokenized) / max(len(self._tokenized), 1)
        self._n_docs = len(chunks)

    def _score(self, query_tokens: list[str], doc_index: int) -> float:
        doc_tokens = self._tokenized[doc_index]
        doc_len = len(doc_tokens)
        if doc_len == 0:
            return 0.0
        term_freq = Counter(doc_tokens)
        score = 0.0
        for term in query_tokens:
            if term not in term_freq:
                continue
            df = self._doc_freq.get(term, 0)
            idf = math.log(1 + (self._n_docs - df + 0.5) / (df + 0.5))
            tf = term_freq[term]
            denom = tf + self.k1 * (1 - self.b + self.b * doc_len / self._avg_dl)
            score += idf * (tf * (self.k1 + 1)) / denom
        return score

    def retrieve(self, query: str, top_k: int | None = None) -> list[str]:
        if not self.chunks:
            return []
        k = top_k or len(self.chunks)
        query_tokens = _tokenize(query)
        scored = [
            (self._score(query_tokens, index), index) for index in range(len(self.chunks))
        ]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [self.chunks[index] for score, index in scored[:k] if score > 0]


class HybridRetriever(Retriever):
    """Combine BM25 with dense-like keyword overlap (no embedding API required)."""

    def __init__(self, chunks: list[str]) -> None:
        self.bm25 = BM25Retriever(chunks)
        self.chunks = chunks

    def _dense_proxy_score(self, query: str, chunk: str) -> float:
        query_tokens = set(_tokenize(query))
        chunk_tokens = set(_tokenize(chunk))
        if not query_tokens or not chunk_tokens:
            return 0.0
        overlap = len(query_tokens & chunk_tokens)
        return overlap / len(query_tokens)

    def retrieve(self, query: str, top_k: int | None = None) -> list[str]:
        if not self.chunks:
            return []
        k = top_k or len(self.chunks)
        bm25_hits = self.bm25.retrieve(query, top_k=len(self.chunks))
        bm25_rank = {chunk: rank for rank, chunk in enumerate(bm25_hits)}
        max_bm25_rank = max(bm25_rank.values()) if bm25_rank else 1

        scored: list[tuple[float, str]] = []
        for chunk in self.chunks:
            bm25_component = 1 - (bm25_rank.get(chunk, max_bm25_rank) / (max_bm25_rank + 1))
            dense_component = self._dense_proxy_score(query, chunk)
            combined = 0.6 * bm25_component + 0.4 * dense_component
            scored.append((combined, chunk))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [chunk for score, chunk in scored[:k] if score > 0]


def load_corpus_chunks(corpus_path: Path | str, config: RetrieverConfig, base_dir: Path) -> list[str]:
    """Load and chunk a corpus file."""
    path = resolve_path(base_dir, str(corpus_path))
    text = path.read_text(encoding="utf-8")
    return chunk_text(text, chunk_size=config.chunk_size, overlap=config.chunk_overlap)


def create_retriever(config: RetrieverConfig, base_dir: Path) -> Retriever:
    """Factory for retriever implementations."""
    if config.type == RetrieverType.NONE:
        return NullRetriever()

    if not config.corpus_path:
        msg = f"corpus_path required for retriever type '{config.type.value}'"
        raise ValueError(msg)

    chunks = load_corpus_chunks(config.corpus_path, config, base_dir)
    if config.type == RetrieverType.BM25:
        retriever: Retriever = BM25Retriever(chunks)
    elif config.type == RetrieverType.DENSE:
        retriever = HybridRetriever(chunks)  # dense proxy until embedding provider lands
    elif config.type == RetrieverType.HYBRID:
        retriever = HybridRetriever(chunks)
    else:
        msg = f"unsupported retriever type: {config.type}"
        raise ValueError(msg)

    if config.rerank and isinstance(retriever, HybridRetriever):
        return retriever
    if config.rerank:
        return HybridRetriever(chunks)
    return retriever
