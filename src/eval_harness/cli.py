"""Command-line interface for the evaluation harness."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from eval_harness import __version__
from eval_harness.config import (
    ConfigError,
    find_project_root,
    load_dataset,
    load_experiment_matrix,
    resolve_path,
)
from eval_harness.leaderboard import build_leaderboard
from eval_harness.pipeline.factory import build_rag_pipeline
from eval_harness.regression import check_regression
from eval_harness.runner import ExperimentRunner, save_run_result
from eval_harness.scorers import score_matrix_run

app = typer.Typer(
    name="eval-harness",
    help="Reproducible LLM evaluation — sweep configs, score setups, rank results.",
    no_args_is_help=True,
)
console = Console()


@app.callback()
def main() -> None:
    """GenAI Eval Harness CLI."""


@app.command("version")
def version_cmd() -> None:
    """Print package version."""
    console.print(f"genai-eval-harness v{__version__}")


@app.command("validate")
def validate_cmd(
    matrix: Path = typer.Argument(..., help="Path to experiment matrix YAML"),
) -> None:
    """Validate an experiment matrix and its referenced datasets."""
    try:
        experiment = load_experiment_matrix(matrix)
    except ConfigError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    table = Table(title=f"Matrix: {experiment.name}")
    table.add_column("Variant")
    table.add_column("Task")
    table.add_column("Model")
    table.add_column("Retriever")
    table.add_column("Dataset")

    for variant in experiment.variants:
        dataset_path = resolve_path(matrix.parent, variant.data.path)
        try:
            dataset = load_dataset(dataset_path)
            dataset_label = f"{dataset.name}@{dataset.version} ({dataset.size} samples)"
        except ConfigError:
            dataset_label = f"[red]missing: {variant.data.path}[/red]"

        table.add_row(
            variant.name,
            variant.task.value,
            variant.model.name,
            variant.retriever.type.value,
            dataset_label,
        )

    console.print(table)
    console.print(
        f"\n[green]✓[/green] Validated {len(experiment.variants)} variants "
        f"(baseline: [bold]{experiment.baseline}[/bold])"
    )


@app.command("run-sample")
def run_sample_cmd(
    matrix: Path = typer.Argument(..., help="Path to experiment matrix YAML"),
    variant_name: str = typer.Option(..., "--variant", "-v", help="Variant name to run"),
    sample_id: str = typer.Option(..., "--sample-id", "-s", help="Sample ID from the dataset"),
    mock: bool = typer.Option(True, "--mock/--live", help="Use mock model (no API key)"),
) -> None:
    """Run a single eval sample through the pipeline (Phase 2 smoke test)."""
    try:
        experiment = load_experiment_matrix(matrix)
        variant = experiment.get_variant(variant_name)
        dataset_path = resolve_path(matrix.parent, variant.data.path)
        dataset = load_dataset(dataset_path)
    except (ConfigError, KeyError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    sample = next((s for s in dataset.samples if s.id == sample_id), None)
    if sample is None:
        console.print(f"[red]Error:[/red] sample '{sample_id}' not found in dataset")
        raise typer.Exit(code=1)

    pipeline = build_rag_pipeline(
        variant, find_project_root(matrix.parent), use_mock_model=mock
    )
    result = pipeline.run(sample)

    if result.error:
        console.print(f"[red]Pipeline error:[/red] {result.error}")
        raise typer.Exit(code=1)

    console.print(f"[bold]Variant:[/bold] {variant_name}")
    console.print(f"[bold]Sample:[/bold] {sample_id}")
    console.print(f"[bold]Retriever:[/bold] {result.retriever_type}")
    console.print(f"[bold]Retrieved ({len(result.retrieved_contexts)}):[/bold]")
    for index, ctx in enumerate(result.retrieved_contexts, 1):
        preview = ctx[:120] + ("..." if len(ctx) > 120 else "")
        console.print(f"  [{index}] {preview}")
    console.print(f"\n[bold]Output:[/bold]\n{result.output}")
    console.print(
        f"\n[dim]latency={result.latency_ms:.1f}ms tokens={result.token_count} "
        f"model={result.model_name}[/dim]"
    )


@app.command("run")
def run_cmd(
    matrix: Path = typer.Argument(..., help="Path to experiment matrix YAML"),
    variant_name: list[str] = typer.Option(
        None,
        "--variant",
        "-v",
        help="Run only these variants (repeatable). Default: all variants.",
    ),
    mock: bool = typer.Option(True, "--mock/--live", help="Use mock model (no API key)"),
    output: Path = typer.Option(
        None,
        "--output",
        "-o",
        help="Directory to save run results (default: results/runs/)",
    ),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Disable progress bars"),
    baseline: Path = typer.Option(
        None,
        "--baseline",
        help="Regression baseline JSON; fail if metrics drop below thresholds",
    ),
    no_score: bool = typer.Option(False, "--no-score", help="Skip metric scoring"),
) -> None:
    """Run an experiment matrix — all variants × all dataset samples."""
    try:
        experiment = load_experiment_matrix(matrix)
        if variant_name:
            for name in variant_name:
                experiment.get_variant(name)
    except (ConfigError, KeyError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    root = find_project_root(matrix.parent)
    runner = ExperimentRunner(
        root,
        use_mock_model=mock,
        show_progress=not quiet,
    )

    console.print(
        f"[bold]Running matrix:[/bold] {experiment.name} "
        f"({len(variant_name) if variant_name else len(experiment.variants)} variant(s), "
        f"{'mock' if mock else 'live'} model)"
    )

    result = runner.run_matrix(
        experiment,
        matrix,
        variant_names=variant_name or None,
    )

    if not no_score:
        result = score_matrix_run(experiment, matrix, result)

    leaderboard = build_leaderboard(result) if not no_score else []

    table = Table(title="Run Summary")
    table.add_column("Variant")
    table.add_column("Samples")
    table.add_column("OK")
    table.add_column("Errors")
    if not no_score:
        table.add_column("recall@k")
        table.add_column("faithfulness")
    table.add_column("Avg latency (ms)")

    for variant_result in result.variant_results:
        latencies = [r.latency_ms for r in variant_result.sample_results if r.error is None]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
        is_baseline_row = variant_result.variant_name == experiment.baseline
        name = variant_result.variant_name
        if is_baseline_row:
            name = f"{name} [baseline]"
        metrics = {m.name: m.value for m in variant_result.aggregate_metrics}
        row = [
            name,
            str(len(variant_result.sample_results)),
            str(variant_result.success_count),
            str(variant_result.error_count),
        ]
        if not no_score:
            row.extend(
                [
                    f"{metrics.get('recall_at_k', 0.0):.3f}",
                    f"{metrics.get('faithfulness', 0.0):.3f}",
                ]
            )
        row.append(f"{avg_latency:.1f}")
        table.add_row(*row)

    console.print(table)

    if leaderboard:
        lb_table = Table(title="Leaderboard")
        lb_table.add_column("Rank")
        lb_table.add_column("Variant")
        lb_table.add_column("Key metrics")
        for row in leaderboard:
            keys = ", ".join(
                f"{k}={row.metrics[k]:.3f}"
                for k in ("recall_at_k", "mrr", "faithfulness", "exact_match")
                if k in row.metrics
            )
            lb_table.add_row(str(row.rank), row.variant_name, keys or "—")
        console.print(lb_table)

    out_dir = output or (root / "results" / "runs")
    run_dir = save_run_result(result, out_dir, leaderboard=leaderboard or None)
    console.print(f"\n[green]✓[/green] Results saved to [bold]{run_dir}[/bold]")
    if result.git_commit:
        console.print(f"[dim]git commit: {result.git_commit[:8]}[/dim]")

    if baseline:
        failures = check_regression(result, baseline)
        if failures:
            for msg in failures:
                console.print(f"[red]Regression:[/red] {msg}")
            raise typer.Exit(code=1)
        console.print(f"[green]✓[/green] Regression check passed ({baseline})")

    if result.total_errors > 0:
        raise typer.Exit(code=1)


@app.command("info")
def info_cmd() -> None:
    """Show project capabilities and roadmap status."""
    table = Table(title="Implementation Roadmap")
    table.add_column("Phase")
    table.add_column("Status")
    table.add_column("Scope")

    phases = [
        ("1", "[green]done[/green]", "Foundation — models, config schema, CLI validate"),
        ("2", "[green]done[/green]", "Pipeline — model, prompt, retriever, run-sample"),
        ("3", "[green]done[/green]", "Runner — matrix executor, run CLI, JSON output"),
        ("4", "[green]done[/green]", "Scorers — retrieval + generation + ops metrics"),
        ("5", "[green]done[/green]", "Leaderboard — CSV/HTML in run directory"),
        ("6", "[green]done[/green]", "Example baseline — examples/baseline_metrics.json"),
        ("7", "[green]done[/green]", "CI — GitHub Actions + regression tests"),
    ]
    for phase, status, scope in phases:
        table.add_row(phase, status, scope)

    console.print(table)


if __name__ == "__main__":
    app()
