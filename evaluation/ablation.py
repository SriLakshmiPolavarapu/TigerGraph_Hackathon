"""
ablation.py - Ablation analysis for GraphRAG (round 3 requirement).

Runs the 50-question benchmark three ways:
  1. FULL: vector search + graph traversal (current graph_rag.py behavior)
  2. VECTOR_ONLY: vector search only, graph/topic traversal disabled
  3. GRAPH_ONLY: topic/ticker graph traversal only, vector search disabled

This isolates whether each retrieval mechanism is actually contributing,
required by round 3's "Graph Contribution and Experimental Rigor" criteria.

Usage:
    python -m evaluation.ablation
"""

import json
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipelines.graph_rag import (
    get_connection, TG_GRAPH, extract_entities_from_question,
    vector_search_filings, build_graph_prompt, MAX_TOPIC_RESULTS,
    MAX_TOTAL_RESULTS, EXCERPT_CHARS,
)
from utils.gemini_client import generate
from utils.metrics import Timer, create_metrics
import tiktoken


def query_graph_ablation(question: str, conn, mode: str) -> dict:
    """
    Same retrieval logic as query_graph() in graph_rag.py, but with a
    mode switch to isolate vector search vs. graph traversal.

    mode: "full" | "vector_only" | "graph_only"
    """
    context_parts = []
    found_filings = set()
    found_companies = set()
    found_topics = set()

    # ── Vector search (skipped in graph_only mode) ──
    if mode in ("full", "vector_only"):
        vector_matches = vector_search_filings(question, conn, k=5)
        for match in vector_matches:
            filing_id = match["filing_id"]
            if filing_id and filing_id not in found_filings:
                found_filings.add(filing_id)
                context_parts.append(
                    f"Filing: {filing_id} ({match['filing_type']}) [vector match]\n"
                    f"Excerpt: {match['content']}"
                )

    # ── Graph/topic traversal (skipped in vector_only mode) ──
    if mode in ("full", "graph_only"):
        entities = extract_entities_from_question(question)

        for entity in entities:
            if len(context_parts) >= MAX_TOTAL_RESULTS:
                break
            try:
                topic_data = conn.getVerticesById("Topic", entity)
                if topic_data:
                    found_topics.add(entity)
                    edges = conn.getEdges("Topic", entity, "COVERS_TOPIC")
                    for edge in edges[:MAX_TOPIC_RESULTS]:
                        filing_id = edge.get("to_id", edge.get("from_id", ""))
                        if filing_id and filing_id not in found_filings:
                            try:
                                filing_data = conn.getVerticesById("Filing", filing_id)
                                if filing_data:
                                    f = filing_data[0] if isinstance(filing_data, list) else filing_data
                                    attrs = f.get("attributes", f)
                                    filing_type = attrs.get("filing_type", "")
                                    content = attrs.get("content", "")[:EXCERPT_CHARS]
                                    found_filings.add(filing_id)
                                    context_parts.append(
                                        f"Filing: {filing_id} ({filing_type}) [topic match]\n"
                                        f"Excerpt: {content}"
                                    )
                            except Exception:
                                pass
            except Exception:
                pass

            try:
                company_data = conn.getVerticesById("Company", entity.upper())
                if company_data:
                    found_companies.add(entity.upper())
                    edges = conn.getEdges("Company", entity.upper(), "FILED")
                    for edge in edges[:MAX_TOPIC_RESULTS]:
                        filing_id = edge.get("to_id", edge.get("from_id", ""))
                        if filing_id and filing_id not in found_filings:
                            try:
                                filing_data = conn.getVerticesById("Filing", filing_id)
                                if filing_data:
                                    f = filing_data[0] if isinstance(filing_data, list) else filing_data
                                    attrs = f.get("attributes", f)
                                    filing_type = attrs.get("filing_type", "")
                                    content = attrs.get("content", "")[:EXCERPT_CHARS]
                                    found_filings.add(filing_id)
                                    context_parts.append(
                                        f"Filing: {filing_id} ({filing_type}) [company match]\n"
                                        f"Excerpt: {content}"
                                    )
                            except Exception:
                                pass
            except Exception:
                pass

    if not context_parts:
        try:
            filings = conn.getVertices("Filing", limit=3)
            for f in filings:
                attrs = f.get("attributes", f)
                filing_type = attrs.get("filing_type", "")
                content = attrs.get("content", "")[:EXCERPT_CHARS]
                if content:
                    context_parts.append(f"Filing: ({filing_type})\nExcerpt: {content}")
        except Exception:
            pass

    return {
        "filings": list(found_filings),
        "companies": list(found_companies),
        "topics": list(found_topics),
        "details": context_parts[:MAX_TOTAL_RESULTS],
    }


def run_ablation_mode(questions: list, conn, mode: str) -> list:
    """Run all 50 questions through GraphRAG with a specific ablation mode."""
    results = []
    encoder = tiktoken.get_encoding("cl100k_base")

    print(f"\nRunning ablation mode: {mode.upper()}")
    print("-" * 60)

    for i, question in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {question[:70]}...")

        with Timer() as timer:
            graph_context = query_graph_ablation(question, conn, mode)
            prompt = build_graph_prompt(question, graph_context)
            response = generate(
                prompt=prompt,
                system_instruction=(
                    "You are a helpful financial research assistant specializing "
                    "in SEC filings. Answer the question based ONLY on the "
                    "provided graph context below. If the context doesn't "
                    "contain enough information, say so. Be concise and specific."
                ),
            )

        prompt_tokens = len(encoder.encode(prompt))
        question_tokens = len(encoder.encode(question))
        context_tokens = prompt_tokens - question_tokens

        metrics = create_metrics(
            pipeline_name=f"GraphRAG-{mode}",
            question=question,
            llm_response=response,
            latency=timer.elapsed,
            context_chunks=len(graph_context["details"]),
            context_tokens=context_tokens,
        )
        results.append(metrics)
        time.sleep(2)

    return results


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "data")
    results_dir = os.path.join(project_root, "results")
    os.makedirs(results_dir, exist_ok=True)

    questions_file = os.path.join(data_dir, "benchmark_questions3.json")
    with open(questions_file, "r") as f:
        data = json.load(f)
        questions = [q["question"] for q in data]

    print("=" * 60)
    print("Ablation Analysis: GraphRAG components")
    print("=" * 60)
    print(f"Loaded {len(questions)} questions")

    conn = get_connection(TG_GRAPH)

    modes = ["vector_only", "graph_only"]
    all_summaries = {}

    for mode in modes:
        results = run_ablation_mode(questions, conn, mode)
        output = [m.to_dict() for m in results]

        output_path = os.path.join(results_dir, f"pipeline3_graphrag_ablation_{mode}.json")
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nSaved: {output_path}")

        total_tokens = sum(m.total_tokens for m in results)
        avg_context = sum(m.context_tokens for m in results) / len(results)
        avg_latency = sum(m.latency_seconds for m in results) / len(results)
        total_cost = sum(m.cost_usd for m in results)

        all_summaries[mode] = {
            "total_tokens": total_tokens,
            "avg_context_tokens": round(avg_context, 1),
            "avg_latency": round(avg_latency, 2),
            "total_cost": round(total_cost, 6),
        }

    print(f"\n{'=' * 60}")
    print("Ablation Summary")
    print(f"{'=' * 60}")
    print("(Compare against your FULL pipeline3_graphrag.json results)")
    for mode, summary in all_summaries.items():
        print(f"\n  {mode.upper()}:")
        print(f"    Total tokens:     {summary['total_tokens']:,}")
        print(f"    Avg context:      {summary['avg_context_tokens']} tokens")
        print(f"    Avg latency:      {summary['avg_latency']}s")
        print(f"    Total cost:       ${summary['total_cost']}")
    print(f"{'=' * 60}")
    print("\nNext step: run 'python -m evaluation.accuracy' logic manually")
    print("against each ablation results file to compare accuracy too.")