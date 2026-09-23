"""Score variant runs using configured metrics."""

from __future__ import annotations

import statistics
from pathlib import Path

from eval_harness.models import (
    EvalDataset,
    EvalSample,
    ExperimentMatrix,
    ExperimentVariant,
    MatrixRunResult,
    MetricScore,
    SampleResult,
    ScorerConfig,
    VariantRunResult,
)
from eval_harness.scorers import generation as gen_metrics
from eval_harness.scorers import retrieval as ret_metrics


def _sample_by_id(dataset: EvalDataset) -> dict[str, EvalSample]:
    return {sample.id: sample for sample in dataset.samples}


def _mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _p50(values: list[float]) -> float:
    if not values:
        return 0.0
    return statistics.median(values)


def score_sample(
    sample: EvalSample,
    result: SampleResult,
    scorers: ScorerConfig,
    *,
    top_k: int = 5,
) -> list[MetricScore]:
    """Compute per-sample metrics."""
    if result.error:
        return []

    scores: list[MetricScore] = []
    refs = sample.reference_contexts
    retrieved = result.retrieved_contexts

    for name in scorers.retrieval:
        if name == "recall_at_k":
            value = ret_metrics.recall_at_k(retrieved, refs, top_k)
        elif name == "mrr":
            value = ret_metrics.mrr(retrieved, refs)
        elif name == "ndcg":
            value = ret_metrics.ndcg_at_k(retrieved, refs, top_k)
        else:
            continue
        scores.append(MetricScore(name=name, value=value))

    for name in scorers.generation:
        if name == "exact_match":
            value = gen_metrics.exact_match(result.output, sample.expected_output)
        elif name == "faithfulness":
            value = gen_metrics.faithfulness(result.output, retrieved)
        elif name == "answer_relevance":
            value = gen_metrics.answer_relevance(result.output, sample.expected_output)
        else:
            continue
        scores.append(MetricScore(name=name, value=value))

    for name in scorers.ops:
        if name == "latency_ms":
            scores.append(MetricScore(name=name, value=result.latency_ms))
        elif name == "token_count":
            scores.append(MetricScore(name=name, value=float(result.token_count)))

    return scores


def score_variant_result(
    variant: ExperimentVariant,
    dataset: EvalDataset,
    variant_result: VariantRunResult,
) -> VariantRunResult:
    """Attach per-sample metrics and aggregate scores to a variant run."""
    lookup = _sample_by_id(dataset)
    top_k = variant.retriever.top_k

    for sample_result in variant_result.sample_results:
        sample = lookup.get(sample_result.sample_id)
        if sample is None:
            continue
        sample_result.metrics = score_sample(
            sample,
            sample_result,
            variant.scorers,
            top_k=top_k,
        )

    aggregate: dict[str, list[float]] = {}
    for sample_result in variant_result.sample_results:
        for metric in sample_result.metrics:
            aggregate.setdefault(metric.name, []).append(metric.value)

    variant_result.aggregate_metrics = [
        MetricScore(name=name, value=_mean(values)) for name, values in sorted(aggregate.items())
    ]

    # Ops aggregates: latency p50 as separate metric
    latencies = [r.latency_ms for r in variant_result.sample_results if r.error is None]
    if latencies and "latency_ms" in aggregate:
        variant_result.aggregate_metrics.append(
            MetricScore(name="latency_p50_ms", value=_p50(latencies))
        )

    return variant_result


def score_matrix_run(
    matrix: ExperimentMatrix,
    matrix_path: str | Path,
    result: MatrixRunResult,
) -> MatrixRunResult:
    """Score all variants in a matrix run."""
    from eval_harness.config import load_dataset, resolve_path

    base = Path(matrix_path).parent
    scored_variants: list[VariantRunResult] = []

    for variant_result in result.variant_results:
        variant = matrix.get_variant(variant_result.variant_name)
        dataset_path = resolve_path(base, variant.data.path)
        dataset = load_dataset(dataset_path)
        scored_variants.append(score_variant_result(variant, dataset, variant_result))

    result.variant_results = scored_variants
    return result
