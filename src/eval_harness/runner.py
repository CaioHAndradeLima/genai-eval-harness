"""Experiment runner — execute config matrix over golden datasets."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from eval_harness.config import load_dataset, resolve_path
from eval_harness.models import (
    EvalDataset,
    ExperimentMatrix,
    ExperimentVariant,
    MatrixRunResult,
    SampleResult,
    VariantRunResult,
)
from eval_harness.pipeline.factory import build_rag_pipeline
from eval_harness.pipeline.types import PipelineOutput

console = Console()


def get_git_commit(base_dir: Path) -> str | None:
    """Return current git commit hash if available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=base_dir,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip() or None
    except (subprocess.SubprocessError, OSError):
        return None


def pipeline_output_to_sample_result(output: PipelineOutput) -> SampleResult:
    """Convert raw pipeline output to a sample result (pre-scoring)."""
    return SampleResult(
        sample_id=output.sample_id,
        input=output.input,
        output=output.output,
        retrieved_contexts=output.retrieved_contexts,
        latency_ms=output.latency_ms,
        token_count=output.token_count,
        error=output.error,
    )


class ExperimentRunner:
    """Orchestrate matrix execution: variants × samples → raw results."""

    def __init__(
        self,
        base_dir: Path,
        *,
        use_mock_model: bool = True,
        show_progress: bool = True,
    ) -> None:
        self.base_dir = base_dir
        self.use_mock_model = use_mock_model
        self.show_progress = show_progress

    def run_variant(
        self,
        matrix: ExperimentMatrix,
        variant: ExperimentVariant,
        dataset: EvalDataset,
        *,
        git_commit: str | None = None,
    ) -> VariantRunResult:
        """Run one variant over every sample in its dataset."""
        pipeline = build_rag_pipeline(
            variant,
            self.base_dir,
            use_mock_model=self.use_mock_model,
        )
        started_at = datetime.now(timezone.utc)
        sample_results: list[SampleResult] = []

        iterator = dataset.samples
        if self.show_progress:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console,
            ) as progress:
                task = progress.add_task(
                    f"[cyan]{variant.name}[/cyan]",
                    total=len(dataset.samples),
                )
                for sample in iterator:
                    output = pipeline.run(sample)
                    sample_results.append(pipeline_output_to_sample_result(output))
                    progress.advance(task)
        else:
            for sample in iterator:
                output = pipeline.run(sample)
                sample_results.append(pipeline_output_to_sample_result(output))

        return VariantRunResult(
            variant_name=variant.name,
            matrix_name=matrix.name,
            dataset_name=dataset.name,
            dataset_version=dataset.version,
            started_at=started_at,
            finished_at=datetime.now(timezone.utc),
            git_commit=git_commit,
            sample_results=sample_results,
        )

    def run_matrix(
        self,
        matrix: ExperimentMatrix,
        matrix_path: Path,
        *,
        variant_names: list[str] | None = None,
    ) -> MatrixRunResult:
        """Execute all (or selected) variants in an experiment matrix."""
        git_commit = get_git_commit(self.base_dir)
        started_at = datetime.now(timezone.utc)

        if variant_names:
            variants = [matrix.get_variant(name) for name in variant_names]
        else:
            variants = matrix.variants

        variant_results: list[VariantRunResult] = []
        for variant in variants:
            dataset_path = resolve_path(matrix_path.parent, variant.data.path)
            dataset = load_dataset(dataset_path)
            result = self.run_variant(matrix, variant, dataset, git_commit=git_commit)
            variant_results.append(result)

        return MatrixRunResult(
            matrix_name=matrix.name,
            baseline=matrix.baseline,
            started_at=started_at,
            finished_at=datetime.now(timezone.utc),
            git_commit=git_commit,
            variant_results=variant_results,
        )


def save_run_result(result: MatrixRunResult, output_dir: Path) -> Path:
    """Persist matrix run results as JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = result.started_at.strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_dir / f"{result.matrix_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    summary_path = run_dir / "matrix_run.json"
    summary_path.write_text(
        result.model_dump_json(indent=2),
        encoding="utf-8",
    )

    for variant_result in result.variant_results:
        variant_path = run_dir / f"{variant_result.variant_name}.json"
        variant_path.write_text(
            variant_result.model_dump_json(indent=2),
            encoding="utf-8",
        )

    manifest = {
        "matrix_name": result.matrix_name,
        "baseline": result.baseline,
        "git_commit": result.git_commit,
        "variants": result.variant_names,
        "total_samples": result.total_samples,
        "total_errors": result.total_errors,
        "run_dir": str(run_dir),
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return run_dir
