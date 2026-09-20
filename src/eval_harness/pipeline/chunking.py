"""Corpus chunking utilities."""

from __future__ import annotations


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    """Split text into overlapping character chunks."""
    if chunk_size <= 0:
        msg = "chunk_size must be positive"
        raise ValueError(msg)
    if overlap >= chunk_size:
        msg = "overlap must be smaller than chunk_size"
        raise ValueError(msg)

    normalized = text.strip()
    if not normalized:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        end = start + chunk_size
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(normalized):
            break
        start = end - overlap

    return chunks
