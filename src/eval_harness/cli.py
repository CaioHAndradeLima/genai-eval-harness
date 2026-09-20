"""Command-line interface for the evaluation harness."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from eval_harness import __version__
from eval_harness.config import ConfigError, load_dataset, load_experiment_matrix, resolve_path

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


@app.command("info")
def info_cmd() -> None:
    """Show project capabilities and roadmap status."""
    table = Table(title="Implementation Roadmap")
    table.add_column("Phase")
    table.add_column("Status")
    table.add_column("Scope")

    phases = [
        ("1", "[green]done[/green]", "Foundation — models, config schema, CLI validate"),
        ("2", "[yellow]planned[/yellow]", "Pipeline — model, prompt, retriever abstractions"),
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
