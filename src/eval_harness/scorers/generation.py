"""Generation evaluation metrics."""

from __future__ import annotations

from eval_harness.scorers.text_utils import normalize_text, tokenize


def exact_match(output: str, expected: str | None) -> float:
    """1.0 if normalized expected answer appears in output."""
    if not expected:
        return 0.0
    out = normalize_text(output)
    exp = normalize_text(expected)
    if not exp:
        return 0.0
    return 1.0 if exp in out or out == exp else 0.0


def faithfulness(output: str, contexts: list[str]) -> float:
    """Heuristic faithfulness: share of output tokens supported by retrieved context."""
    out_tokens = tokenize(output)
    if not out_tokens:
        return 0.0
    context_tokens = tokenize(" ".join(contexts))
    if not context_tokens:
        return 0.0
    supported = len(out_tokens & context_tokens)
    return supported / len(out_tokens)


def answer_relevance(output: str, expected: str | None) -> float:
    """Token overlap between output and expected answer."""
    if not expected:
        return 0.0
    out_tokens = tokenize(output)
    exp_tokens = tokenize(expected)
    if not out_tokens or not exp_tokens:
        return 0.0
    return len(out_tokens & exp_tokens) / len(out_tokens)
