"""Shared text normalization for metrics."""

from __future__ import annotations

import re


def tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())
