"""Regression checks against committed baseline metrics."""

from __future__ import annotations

import json
from pathlib import Path

from eval_harness.models import MatrixRunResult, MetricScore


class RegressionError(Exception):
    """Raised when metrics fall below baseline tolerances."""


def _metrics_dict(metrics: list[MetricScore]) -> dict[str, float]:
    return {m.name: m.value for m in metrics}


def load_baseline(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        msg = "baseline file must be a JSON object"
        raise ValueError(msg)
    return data


def check_regression(
    result: MatrixRunResult,
    baseline_path: Path,
    *,
    default_tolerance: float = 0.05,
) -> list[str]:
    """
    Compare scored matrix run to baseline.

    Returns list of failure messages (empty if all checks pass).
    """
    baseline = load_baseline(baseline_path)
    tolerance = float(baseline.get("tolerance", default_tolerance))
    expected_variants: dict = baseline.get("variants", {})
    failures: list[str] = []

    for variant_result in result.variant_results:
        name = variant_result.variant_name
        if name not in expected_variants:
            failures.append(f"variant '{name}' missing from baseline")
            continue

        current = _metrics_dict(variant_result.aggregate_metrics)
        expected = expected_variants[name]
        for metric_name, min_value in expected.items():
            if metric_name == "description":
                continue
            if not isinstance(min_value, (int, float)):
                continue
            actual = current.get(metric_name)
            if actual is None:
                failures.append(f"{name}/{metric_name}: metric not computed")
                continue
            floor = float(min_value) - tolerance
            if actual < floor:
                failures.append(
                    f"{name}/{metric_name}: {actual:.4f} < floor {floor:.4f} "
                    f"(baseline {float(min_value):.4f}, tolerance {tolerance})"
                )

    return failures


def assert_regression(result: MatrixRunResult, baseline_path: Path) -> None:
    """Raise RegressionError if any metric is below baseline tolerance."""
    failures = check_regression(result, baseline_path)
    if failures:
        raise RegressionError("\n".join(failures))
