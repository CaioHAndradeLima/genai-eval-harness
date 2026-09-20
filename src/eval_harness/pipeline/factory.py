"""Build pipeline instances from experiment variant configs."""

from __future__ import annotations

from pathlib import Path

from eval_harness.models import ExperimentVariant
from eval_harness.pipeline.model import create_model_provider
from eval_harness.pipeline.prompt import PromptRenderer
from eval_harness.pipeline.rag import RAGPipeline
from eval_harness.pipeline.retriever import create_retriever


def build_rag_pipeline(
    variant: ExperimentVariant,
    base_dir: Path,
    *,
    use_mock_model: bool = False,
) -> RAGPipeline:
    """Construct a RAG pipeline from a validated experiment variant."""
    retriever = create_retriever(variant.retriever, base_dir)
    prompt_renderer = PromptRenderer(variant.prompt.path, base_dir=base_dir)
    model = create_model_provider(variant.model, use_mock=use_mock_model)
    return RAGPipeline(
        variant=variant,
        retriever=retriever,
        prompt_renderer=prompt_renderer,
        model=model,
        base_dir=base_dir,
    )
