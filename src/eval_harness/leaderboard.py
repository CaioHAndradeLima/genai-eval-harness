"""Leaderboard generation from scored matrix runs."""

from __future__ import annotations

import csv
from io import StringIO
from pathlib import Path

from eval_harness.models import LeaderboardRow, MatrixRunResult, MetricScore


def _metrics_dict(metrics: list[MetricScore]) -> dict[str, float]:
    return {m.name: m.value for m in metrics}


def _primary_score(metrics: dict[str, float]) -> float:
    """Ranking key: prefer recall, then faithfulness, then exact_match."""
    for key in ("recall_at_k", "faithfulness", "exact_match", "mrr"):
        if key in metrics:
            return metrics[key]
    return 0.0


def build_leaderboard(result: MatrixRunResult) -> list[LeaderboardRow]:
    """Rank variants by aggregate metrics."""
    rows: list[LeaderboardRow] = []
    for variant_result in result.variant_results:
        metrics = _metrics_dict(variant_result.aggregate_metrics)
        latency_p50 = metrics.get("latency_p50_ms", metrics.get("latency_ms", 0.0))
        rows.append(
            LeaderboardRow(
                rank=0,
                variant_name=variant_result.variant_name,
                is_baseline=variant_result.variant_name == result.baseline,
                metrics=metrics,
                latency_p50_ms=latency_p50,
            )
        )

    rows.sort(key=lambda row: _primary_score(row.metrics), reverse=True)
    for index, row in enumerate(rows, start=1):
        row.rank = index
    return rows


def leaderboard_to_csv(rows: list[LeaderboardRow]) -> str:
    """Render leaderboard as CSV string."""
    if not rows:
        return "rank,variant,is_baseline\n"

    metric_names = sorted({key for row in rows for key in row.metrics})
    fieldnames = ["rank", "variant", "is_baseline", "latency_p50_ms", *metric_names]
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        record = {
            "rank": row.rank,
            "variant": row.variant_name,
            "is_baseline": row.is_baseline,
            "latency_p50_ms": f"{row.latency_p50_ms:.2f}",
        }
        for name in metric_names:
            value = row.metrics.get(name)
            record[name] = f"{value:.4f}" if value is not None else ""
        writer.writerow(record)
    return buffer.getvalue()


def leaderboard_to_html(rows: list[LeaderboardRow], *, title: str = "Eval Leaderboard") -> str:
    """Render a simple HTML leaderboard table."""
    if not rows:
        return f"<html><body><h1>{title}</h1><p>No results.</p></body></html>"

    metric_names = sorted({key for row in rows for key in row.metrics})
    headers = ["Rank", "Variant", "Baseline", "Latency p50 (ms)", *metric_names]
    head_html = "".join(f"<th>{h}</th>" for h in headers)

    body_rows = []
    for row in rows:
        cells = [
            str(row.rank),
            row.variant_name,
            "yes" if row.is_baseline else "",
            f"{row.latency_p50_ms:.1f}",
        ]
        cells.extend(f"{row.metrics.get(n, 0.0):.4f}" for n in metric_names)
        body_rows.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{title}</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 2rem; }}
    table {{ border-collapse: collapse; width: 100%; max-width: 960px; }}
    th, td {{ border: 1px solid #ddd; padding: 0.5rem 0.75rem; text-align: left; }}
    th {{ background: #f4f4f5; }}
    tr:nth-child(even) {{ background: #fafafa; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <table>
    <thead><tr>{head_html}</tr></thead>
    <tbody>
      {"".join(body_rows)}
    </tbody>
  </table>
</body>
</html>
"""


def save_leaderboard(rows: list[LeaderboardRow], run_dir: Path, *, matrix_name: str) -> None:
    """Write CSV and HTML leaderboard files."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "leaderboard.csv").write_text(
        leaderboard_to_csv(rows),
        encoding="utf-8",
    )
    (run_dir / "leaderboard.html").write_text(
        leaderboard_to_html(rows, title=f"Leaderboard — {matrix_name}"),
        encoding="utf-8",
    )
