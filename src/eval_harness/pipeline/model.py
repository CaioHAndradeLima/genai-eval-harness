"""LLM provider abstractions."""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import httpx

from eval_harness.models import ModelConfig


@dataclass
class GenerationResult:
    """Structured result from an LLM call."""

    text: str
    token_count: int
    latency_ms: float
    model_name: str


class ModelProvider(ABC):
    """Abstract LLM provider."""

    @abstractmethod
    def generate(self, prompt: str) -> GenerationResult:
        """Generate text from a prompt."""


class MockModelProvider(ModelProvider):
    """Deterministic model for tests and dry-runs — no API key required."""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config

    def generate(self, prompt: str) -> GenerationResult:
        start = time.perf_counter()
        # Echo the question line when present for predictable test output.
        answer_line = "mock-response"
        for line in prompt.splitlines():
            stripped = line.strip()
            if stripped.lower().startswith(("question:", "user question:")):
                answer_line = f"Mock answer for: {stripped.split(':', 1)[1].strip()}"
                break
        latency_ms = (time.perf_counter() - start) * 1000
        token_count = max(1, len(prompt.split()) // 4)
        return GenerationResult(
            text=answer_line,
            token_count=token_count,
            latency_ms=latency_ms,
            model_name=f"mock/{self.config.name}",
        )


class OpenAIModelProvider(ModelProvider):
    """OpenAI chat completions via HTTP — requires OPENAI_API_KEY."""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.api_key = os.environ.get("OPENAI_API_KEY", "")
        if not self.api_key:
            msg = "OPENAI_API_KEY is required for OpenAI model provider"
            raise ValueError(msg)

    def generate(self, prompt: str) -> GenerationResult:
        start = time.perf_counter()
        payload = {
            "model": self.config.name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        with httpx.Client(timeout=60.0) as client:
            response = client.post(
                "https://api.openai.com/v1/chat/completions",
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
            data = response.json()

        text = data["choices"][0]["message"]["content"].strip()
        usage = data.get("usage", {})
        token_count = int(usage.get("total_tokens", len(text.split())))
        latency_ms = (time.perf_counter() - start) * 1000
        return GenerationResult(
            text=text,
            token_count=token_count,
            latency_ms=latency_ms,
            model_name=self.config.name,
        )


def create_model_provider(config: ModelConfig, *, use_mock: bool = False) -> ModelProvider:
    """Factory for model providers."""
    if use_mock:
        return MockModelProvider(config)
    if config.provider == "openai":
        return OpenAIModelProvider(config)
    if config.provider == "mock":
        return MockModelProvider(config)
    msg = f"unsupported model provider: {config.provider}"
    raise ValueError(msg)
