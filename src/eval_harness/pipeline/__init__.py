"""Pipeline layer — model, prompt, retriever, and RAG orchestration."""

from eval_harness.pipeline.factory import build_rag_pipeline
from eval_harness.pipeline.rag import RAGPipeline
from eval_harness.pipeline.types import PipelineOutput

__all__ = ["RAGPipeline", "PipelineOutput", "build_rag_pipeline"]
