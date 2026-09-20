"""Core domain models for experiments, datasets, and results."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class TaskType(StrEnum):
    """Supported evaluation task families."""

    QA = "qa"
    RAG = "rag"
    CLASSIFICATION = "classification"
    GENERATION = "generation"


class RetrieverType(StrEnum):
    """Retrieval strategy identifiers."""

    NONE = "none"
    BM25 = "bm25"
    DENSE = "dense"
    HYBRID = "hybrid"


class EvalSample(BaseModel):
    """One labeled example in the golden evaluation set."""

    id: str
    input: str
    expected_output: str | None = None
    reference_contexts: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvalDataset(BaseModel):
    """Versioned collection of evaluation samples."""

    name: str
    version: str
    task: TaskType
    description: str = ""
    samples: list[EvalSample]

    @property
    def size(self) -> int:
        return len(self.samples)


class ModelConfig(BaseModel):
    """LLM provider settings for one experiment variant."""

    provider: str = "openai"
    name: str
    temperature: float = 0.0
    max_tokens: int = 1024


class PromptConfig(BaseModel):
    """Versioned prompt template reference."""

    path: str
    version: str = "v1"
    variables: dict[str, str] = Field(default_factory=dict)


class RetrieverConfig(BaseModel):
    """Retrieval stack settings (used by RAG tasks)."""

    type: RetrieverType = RetrieverType.NONE
    corpus_path: str | None = None
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k: int = 5
    rerank: bool = False
    embedding_model: str = "text-embedding-3-small"


class DataConfig(BaseModel):
    """Dataset binding for an experiment."""

    path: str
    name: str | None = None
    version: str | None = None


class ScorerConfig(BaseModel):
    """Metrics to compute for a run."""

    retrieval: list[str] = Field(default_factory=list)
    generation: list[str] = Field(default_factory=list)
    ops: list[str] = Field(default_factory=lambda: ["latency_ms", "token_count"])


class ExperimentVariant(BaseModel):
    """One fully-specified setup in the config matrix."""

    name: str
    task: TaskType
    model: ModelConfig
    prompt: PromptConfig
    retriever: RetrieverConfig = Field(default_factory=RetrieverConfig)
    data: DataConfig
    scorers: ScorerConfig = Field(default_factory=ScorerConfig)
    tags: dict[str, str] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def name_must_be_slug(cls, value: str) -> str:
        if not value.replace("-", "").replace("_", "").isalnum():
            msg = "variant name must be alphanumeric with dashes/underscores"
            raise ValueError(msg)
        return value


class ExperimentMatrix(BaseModel):
    """Sweep definition: baseline + variants to compare."""

    name: str
    description: str = ""
    baseline: str
    variants: list[ExperimentVariant]

    def get_variant(self, name: str) -> ExperimentVariant:
        for variant in self.variants:
            if variant.name == name:
                return variant
        msg = f"variant '{name}' not found in matrix '{self.name}'"
        raise KeyError(msg)

    @property
    def variant_names(self) -> list[str]:
        return [v.name for v in self.variants]


class MetricScore(BaseModel):
    """Single metric value for one sample or aggregated."""

    name: str
    value: float
    details: dict[str, Any] = Field(default_factory=dict)


class SampleResult(BaseModel):
    """Pipeline output + scores for one eval sample."""

    sample_id: str
    input: str
    output: str
    retrieved_contexts: list[str] = Field(default_factory=list)
    metrics: list[MetricScore] = Field(default_factory=list)
    latency_ms: float = 0.0
    token_count: int = 0
    error: str | None = None


class VariantRunResult(BaseModel):
    """Complete result of running one variant over the full dataset."""

    variant_name: str
    matrix_name: str
    dataset_name: str
    dataset_version: str
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime | None = None
    git_commit: str | None = None
    sample_results: list[SampleResult] = Field(default_factory=list)
    aggregate_metrics: list[MetricScore] = Field(default_factory=list)

    @property
    def success_count(self) -> int:
        return sum(1 for r in self.sample_results if r.error is None)

    @property
    def error_count(self) -> int:
        return sum(1 for r in self.sample_results if r.error is not None)


class LeaderboardRow(BaseModel):
    """One row in the experiment comparison table."""

    rank: int
    variant_name: str
    is_baseline: bool
    metrics: dict[str, float]
    latency_p50_ms: float = 0.0
    cost_usd: float = 0.0
