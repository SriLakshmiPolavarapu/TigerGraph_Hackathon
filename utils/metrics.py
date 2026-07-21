"""
Shared metrics tracking for all three pipelines.
Captures tokens, latency, cost, retrieved evidence, and citations per query.
"""

import time
import re
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

    # ── Round 3 additions ──
    system_tokens: int = 0           # Tokens in the system instruction
    question_tokens: int = 0         # Tokens in the question itself
    evidence: list = field(default_factory=list)   # [{"source_id": str, "text": str}, ...]
    citations: list = field(default_factory=list)  # source_ids the answer appears to reference
    error: Optional[str] = None      # set if the query failed or returned malformed output

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


def extract_citations(answer: str, evidence: list) -> list:
    """
    Heuristic citation extraction: which evidence source_ids does the
    answer appear to reference? Checks if a source_id (e.g. a filing_id
    like "AAPL_8-K_2026-04-30", or a chunk source filename) appears
    verbatim in the answer, or if enough distinctive words from that
    evidence's text appear in the answer.
    """
    if not evidence:
        return []

    citations = []
    answer_lower = answer.lower()

    for item in evidence:
        source_id = item.get("source_id", "")
        text = item.get("text", "")

        # Direct mention of the source id (e.g. ticker or filing id fragment)
        if source_id and source_id.lower() in answer_lower:
            citations.append(source_id)
            continue

        # Fallback: does the answer share a distinctive rare word/number
        # with this piece of evidence? (very light heuristic, numbers and
        # capitalized tokens are the most distinctive in financial text)
        evidence_numbers = set(re.findall(r'\b\d[\d,\.]*\b', text))
        answer_numbers = set(re.findall(r'\b\d[\d,\.]*\b', answer))
        if evidence_numbers & answer_numbers:
            citations.append(source_id)

    return citations


def create_metrics(
    pipeline_name: str,
    question: str,
    llm_response: dict,
    latency: float,
    context_chunks: int = 0,
    context_tokens: int = 0,
    system_instruction: str = "",
    evidence: list = None,
    error: str = None,
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
        system_instruction: the system prompt text, used to compute
            system_tokens separately (round 3 granular token accounting)
        evidence: list of {"source_id": str, "text": str} dicts, the
            actual retrieved evidence supplied to the LLM (round 3
            "Required Results" - retrieved document/chunk identifiers
            and evidence supplied to the LLM)
        error: error message if this query failed, else None

    Returns:
        QueryMetrics with all fields populated
    """
    evidence = evidence or []
    answer_text = llm_response.get("answer", "") if llm_response else ""

    system_tokens = 0
    question_tokens = 0
    if system_instruction or question:
        try:
            import tiktoken
            encoder = tiktoken.get_encoding("cl100k_base")
            if system_instruction:
                system_tokens = len(encoder.encode(system_instruction))
            if question:
                question_tokens = len(encoder.encode(question))
        except Exception:
            pass

    citations = extract_citations(answer_text, evidence)

    metrics = QueryMetrics(
        pipeline_name=pipeline_name,
        question=question,
        answer=answer_text,
        input_tokens=llm_response.get("input_tokens", 0) if llm_response else 0,
        output_tokens=llm_response.get("output_tokens", 0) if llm_response else 0,
        total_tokens=llm_response.get("total_tokens", 0) if llm_response else 0,
        latency_seconds=round(latency, 4),
        context_chunks=context_chunks,
        context_tokens=context_tokens,
        system_tokens=system_tokens,
        question_tokens=question_tokens,
        evidence=evidence,
        citations=citations,
        error=error,
    )
    metrics.calculate_cost()
    return metrics