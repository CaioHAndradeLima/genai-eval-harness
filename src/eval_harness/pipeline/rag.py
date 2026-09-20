"""RAG pipeline orchestration."""

from __future__ import annotations

import time
from pathlib import Path

from eval_harness.models import EvalSample, ExperimentVariant
from eval_harness.pipeline.model import ModelProvider
from eval_harness.pipeline.prompt import PromptRenderer
from eval_harness.pipeline.retriever import Retriever
from eval_harness.pipeline.types import PipelineOutput


class RAGPipeline:
    """End-to-end RAG pipeline: retrieve → render prompt → generate."""

    def __init__(
        self,
        variant: ExperimentVariant,
        *,
        retriever: Retriever,
        prompt_renderer: PromptRenderer,
        model: ModelProvider,
        base_dir: Path | None = None,
    ) -> None:
        self.variant = variant
        self.retriever = retriever
        self.prompt_renderer = prompt_renderer
        self.model = model
        self.base_dir = base_dir or Path.cwd()

    def run(self, sample: EvalSample) -> PipelineOutput:
        """Run one eval sample through the pipeline."""
        start = time.perf_counter()
        try:
            contexts = self.retriever.retrieve(
                sample.input,
                top_k=self.variant.retriever.top_k,
            )
            prompt = self.prompt_renderer.render(
                question=sample.input,
                contexts=contexts,
                **self.variant.prompt.variables,
            )
            generation = self.model.generate(prompt)
            latency_ms = (time.perf_counter() - start) * 1000

            return PipelineOutput(
                sample_id=sample.id,
                input=sample.input,
                output=generation.text,
                prompt=prompt,
                retrieved_contexts=contexts,
                latency_ms=latency_ms,
                token_count=generation.token_count,
                model_name=generation.model_name,
                retriever_type=self.variant.retriever.type.value,
            )
        except Exception as exc:
            latency_ms = (time.perf_counter() - start) * 1000
            return PipelineOutput(
                sample_id=sample.id,
                input=sample.input,
                output="",
                prompt="",
                retrieved_contexts=[],
                latency_ms=latency_ms,
                model_name=self.variant.model.name,
                retriever_type=self.variant.retriever.type.value,
                error=str(exc),
            )
