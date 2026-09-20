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
from eval_harness.pipeline.factory import build_rag_pipeline

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
        ("3", "[yellow]planned[/yellow]", "Runner — config matrix executor"),
        ("4", "[yellow]planned[/yellow]", "Scorers — retrieval + generation metrics"),
        ("5", "[yellow]planned[/yellow]", "Store — results logging + leaderboard"),
        ("6", "[yellow]planned[/yellow]", "Example — full RAG benchmark walkthrough"),
        ("7", "[yellow]planned[/yellow]", "CI — regression tests + GitHub Actions"),
    ]
    for phase, status, scope in phases:
        table.add_row(phase, status, scope)

    console.print(table)


if __name__ == "__main__":
    app()
