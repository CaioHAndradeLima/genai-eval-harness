"""Tests for the experiment runner."""

from pathlib import Path

from eval_harness.config import load_dataset, load_experiment_matrix, resolve_path
from eval_harness.models import MatrixRunResult
from eval_harness.runner import ExperimentRunner, pipeline_output_to_sample_result, save_run_result
from eval_harness.pipeline.types import PipelineOutput

ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "configs/matrices/rag_qa_baseline.yaml"


def test_pipeline_output_to_sample_result():
    output = PipelineOutput(
        sample_id="s1",
        input="q",
        output="a",
        prompt="p",
        retrieved_contexts=["ctx"],
        latency_ms=12.5,
        token_count=10,
    )
    sample = pipeline_output_to_sample_result(output)
    assert sample.sample_id == "s1"
    assert sample.output == "a"
    assert sample.retrieved_contexts == ["ctx"]
    assert sample.error is None


def test_run_single_variant():
    matrix = load_experiment_matrix(MATRIX)
    variant = matrix.get_variant("bm25-only")
    dataset = load_dataset(resolve_path(MATRIX.parent, variant.data.path))

    runner = ExperimentRunner(ROOT, use_mock_model=True, show_progress=False)
    result = runner.run_variant(matrix, variant, dataset)

    assert result.variant_name == "bm25-only"
    assert len(result.sample_results) == dataset.size
    assert result.success_count == dataset.size
    assert all(r.output for r in result.sample_results)


def test_run_full_matrix_mock():
    matrix = load_experiment_matrix(MATRIX)
    runner = ExperimentRunner(ROOT, use_mock_model=True, show_progress=False)
    result = runner.run_matrix(matrix, MATRIX)

    assert isinstance(result, MatrixRunResult)
    assert len(result.variant_results) == 2
    assert result.total_samples == 10  # 5 samples × 2 variants
    assert result.total_errors == 0


def test_run_matrix_variant_filter():
    matrix = load_experiment_matrix(MATRIX)
    runner = ExperimentRunner(ROOT, use_mock_model=True, show_progress=False)
    result = runner.run_matrix(matrix, MATRIX, variant_names=["hybrid-rerank"])

    assert len(result.variant_results) == 1
    assert result.variant_results[0].variant_name == "hybrid-rerank"


def test_save_run_result(tmp_path: Path):
    matrix = load_experiment_matrix(MATRIX)
    runner = ExperimentRunner(ROOT, use_mock_model=True, show_progress=False)
    result = runner.run_matrix(matrix, MATRIX, variant_names=["bm25-only"])

    run_dir = save_run_result(result, tmp_path)
    assert run_dir.exists()
    assert (run_dir / "matrix_run.json").exists()
    assert (run_dir / "bm25-only.json").exists()
    assert (run_dir / "manifest.json").exists()
