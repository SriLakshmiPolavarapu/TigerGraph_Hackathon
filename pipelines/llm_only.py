"""
Pipeline 1: LLM-Only

The simplest pipeline. Sends the question directly to Gemini
with no retrieval, no context. This is the worst-case baseline.
"""

import json
import sys
import os
import time

# Add project root to Python path so imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.gemini_client import generate
from utils.metrics import Timer, create_metrics, QueryMetrics


SYSTEM_PROMPT = """You are a helpful research assistant specializing in AI and machine learning.
Answer questions accurately and concisely based on your training knowledge.
If you are unsure about something, say so rather than guessing."""


def run_query(question: str) -> QueryMetrics:
    """
    Run a single question through the LLM-Only pipeline.
    """
    with Timer() as timer:
        response = generate(
            prompt=question,
            system_instruction=SYSTEM_PROMPT,
        )

    metrics = create_metrics(
        pipeline_name="LLM-Only",
        question=question,
        llm_response=response,
        latency=timer.elapsed,
    )

    return metrics


def run_benchmark(questions: list) -> list:
    """
    Run a list of questions through Pipeline 1 and collect metrics.
    """
    results = []
    print(f"\nRunning Pipeline 1 (LLM-Only) on {len(questions)} questions...")
    print("-" * 60)

    for i, question in enumerate(questions, 1):
        print(f"\n[{i}/{len(questions)}] {question[:80]}...")
        metrics = run_query(question)

        print(f"  Tokens: {metrics.total_tokens} "
              f"(in: {metrics.input_tokens}, out: {metrics.output_tokens})")
        print(f"  Latency: {metrics.latency_seconds:.2f}s")
        print(f"  Cost: ${metrics.cost_usd:.6f}")

        results.append(metrics)

        # Small delay to respect rate limits on free tier
        time.sleep(1)

    # Print summary
    total_tokens = sum(m.total_tokens for m in results)
    avg_latency = sum(m.latency_seconds for m in results) / len(results)
    total_cost = sum(m.cost_usd for m in results)

    print(f"\n{'=' * 60}")
    print(f"Pipeline 1 (LLM-Only) Summary")
    print(f"{'=' * 60}")
    print(f"  Questions:     {len(results)}")
    print(f"  Total tokens:  {total_tokens:,}")
    print(f"  Avg latency:   {avg_latency:.2f}s")
    print(f"  Total cost:    ${total_cost:.6f}")
    print(f"{'=' * 60}")

    return results


# Sample benchmark questions
SAMPLE_QUESTIONS = [
    "What is retrieval augmented generation and how does it improve LLM accuracy?",
    "Explain the transformer architecture and the role of self-attention mechanisms.",
    "How do graph neural networks differ from traditional neural networks?",
    "What are the main challenges in federated learning for healthcare applications?",
    "Compare collaborative filtering and content-based recommendation systems.",
    "How does RLHF work in training large language models?",
    "What are the key differences between BERT and GPT architectures?",
    "Explain knowledge graph embedding methods and their applications.",
    "What are the privacy concerns with large language models and how can they be mitigated?",
    "How does multi-hop reasoning work in knowledge graphs for question answering?",
]


if __name__ == "__main__":
    print("=" * 60)
    print("Pipeline 1: LLM-Only Baseline")
    print("=" * 60)

    # Check for custom questions file
    questions_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "benchmark_questions.json"
    )

    if os.path.exists(questions_file):
        with open(questions_file, "r") as f:
            data = json.load(f)
            questions = [q["question"] for q in data]
        print(f"Loaded {len(questions)} questions from {questions_file}")
    else:
        questions = SAMPLE_QUESTIONS
        print(f"Using {len(questions)} sample questions")

    results = run_benchmark(questions)

    # Save results
    results_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results"
    )
    os.makedirs(results_dir, exist_ok=True)
    output = [m.to_dict() for m in results]
    output_path = os.path.join(results_dir, "pipeline1_llm_only.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")
