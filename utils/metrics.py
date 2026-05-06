"""
Shared metrics tracking for all three pipelines.
Captures tokens, latency, and cost per query.
"""

import time
from dataclasses import dataclass, field, asdict
from typing import Optional
import config


@dataclass
class QueryMetrics:
    """Metrics captured for a single query across any pipeline."""
    pipeline_name: str
    question: str
    answer: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    latency_seconds: float = 0.0
    cost_usd: float = 0.0
    context_chunks: int = 0          # Number of retrieved chunks (RAG pipelines)
    context_tokens: int = 0          # Tokens in the retrieved context

    def calculate_cost(self):
        """Calculate cost based on Gemini pricing."""
        input_cost = (self.input_tokens / 1_000_000) * config.COST_PER_1M_INPUT_TOKENS
        output_cost = (self.output_tokens / 1_000_000) * config.COST_PER_1M_OUTPUT_TOKENS
        self.cost_usd = round(input_cost + output_cost, 8)

    def to_dict(self) -> dict:
        return asdict(self)


class Timer:
    """Simple context manager to measure elapsed time."""

    def __init__(self):
        self.elapsed = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self._start


def create_metrics(
    pipeline_name: str,
    question: str,
    llm_response: dict,
    latency: float,
    context_chunks: int = 0,
    context_tokens: int = 0,
) -> QueryMetrics:
    """
    Build a QueryMetrics object from an LLM response dict.

    Args:
        pipeline_name: "LLM-Only", "Basic RAG", or "GraphRAG"
        question: the original question
        llm_response: dict from gemini_client.generate()
        latency: seconds elapsed
        context_chunks: number of chunks retrieved (0 for LLM-Only)
        context_tokens: tokens in retrieved context (0 for LLM-Only)

    Returns:
        QueryMetrics with all fields populated
    """
    metrics = QueryMetrics(
        pipeline_name=pipeline_name,
        question=question,
        answer=llm_response["answer"],
        input_tokens=llm_response["input_tokens"],
        output_tokens=llm_response["output_tokens"],
        total_tokens=llm_response["total_tokens"],
        latency_seconds=round(latency, 4),
        context_chunks=context_chunks,
        context_tokens=context_tokens,
    )
    metrics.calculate_cost()
    return metrics
