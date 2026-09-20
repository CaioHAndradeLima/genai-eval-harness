"""Tests for domain models."""

import pytest
from pydantic import ValidationError

from eval_harness.models import EvalDataset, ExperimentMatrix, ExperimentVariant, TaskType


def test_eval_dataset_size():
    dataset = EvalDataset(
        name="test",
        version="1.0",
        task=TaskType.QA,
        samples=[
            {"id": "1", "input": "hello", "expected_output": "world"},
        ],
    )
    assert dataset.size == 1


def test_variant_name_must_be_slug():
    with pytest.raises(ValidationError):
        ExperimentVariant(
            name="bad name!",
            task=TaskType.RAG,
            model={"name": "gpt-4o-mini"},
            prompt={"path": "prompts/test.jinja"},
            data={"path": "datasets/test.json"},
        )


def test_matrix_get_variant():
    matrix = ExperimentMatrix(
        name="test-matrix",
        baseline="a",
        variants=[
            ExperimentVariant(
                name="a",
                task=TaskType.RAG,
                model={"name": "gpt-4o-mini"},
                prompt={"path": "prompts/test.jinja"},
                data={"path": "datasets/test.json"},
            ),
        ],
    )
    assert matrix.get_variant("a").name == "a"
    with pytest.raises(KeyError):
        matrix.get_variant("missing")
