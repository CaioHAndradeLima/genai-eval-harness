"""Pipeline input/output types."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PipelineOutput(BaseModel):
    """Raw output from running one sample through the pipeline (pre-scoring)."""

    sample_id: str
    input: str
    output: str
    prompt: str
    retrieved_contexts: list[str] = Field(default_factory=list)
    latency_ms: float = 0.0
    token_count: int = 0
    model_name: str = ""
    retriever_type: str = ""
    error: str | None = None
