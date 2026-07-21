"""
reproducibility.py - Run the held-out evaluation 3 times per pipeline and
report mean + variance (round 3 requirement: "Run the complete held-out
evaluation three times using the same frozen pipeline configurations.
Report the mean and variation for nondeterministic metrics... Retain the
raw per-question outputs from every run rather than reporting only the
best run.")

Usage:
    python -m evaluation.reproducibility --pipeline graph_rag
    python -m evaluation.reproducibility --pipeline basic_rag
    python -m evaluation.reproducibility --pipeline llm_only
    python -m evaluation.reproducibility --pipeline all
"""

import json
import sys
import os
import argparse
import statistics

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_questions():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    questions_file = os.path.join(project_root, "data", "benchmark_questions3.json")
    with open(questions_file, "r") as f:
        data = json.load(f)
    return [q["question"] for q in data]


def run_n_times(pipeline_name: str, run_benchmark_fn, questions: list, n: int = 3) -> dict:
    """
    Run a pipeline's benchmark N times, save each run's raw output
    separately, and compute mean/variance of the nondeterministic metrics
    (total_tokens, latency, cost) across runs.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(project_root, "results", "reproducibility")
    os.makedirs(results_dir, exist_ok=True)

    run_summaries = []

    for run_idx in range(1, n + 1):
        print(f"\n{'=' * 60}")
        print(f"{pipeline_name} - Run {run_idx}/{n}")
        print(f"{'=' * 60}")

        results = run_benchmark_fn(questions)
        output = [m.to_dict() for m in results]

        run_path = os.path.join(results_dir, f"{pipeline_name}_run{run_idx}.json")
        with open(run_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Saved raw run to {run_path}")

        total_tokens = sum(m.total_tokens for m in results)
        avg_latency = sum(m.latency_seconds for m in results) / len(results)
        total_cost = sum(m.cost_usd for m in results)

        run_summaries.append({
            "run": run_idx,
            "total_tokens": total_tokens,
            "avg_latency": avg_latency,
            "total_cost": total_cost,
        })

    # Compute mean and variance (stdev) across the 3 runs
    tokens_list = [r["total_tokens"] for r in run_summaries]
    latency_list = [r["avg_latency"] for r in run_summaries]
    cost_list = [r["total_cost"] for r in run_summaries]

    summary = {
        "pipeline": pipeline_name,
        "num_runs": n,
        "runs": run_summaries,
        "total_tokens_mean": statistics.mean(tokens_list),
        "total_tokens_stdev": statistics.stdev(tokens_list) if n > 1 else 0,
        "avg_latency_mean": round(statistics.mean(latency_list), 3),
        "avg_latency_stdev": round(statistics.stdev(latency_list), 3) if n > 1 else 0,
        "total_cost_mean": round(statistics.mean(cost_list), 6),
        "total_cost_stdev": round(statistics.stdev(cost_list), 6) if n > 1 else 0,
    }

    summary_path = os.path.join(results_dir, f"{pipeline_name}_reproducibility_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved reproducibility summary to {summary_path}")

    return summary


def print_summary(summary: dict):
    print(f"\n{'=' * 60}")
    print(f"Reproducibility Summary: {summary['pipeline']}")
    print(f"{'=' * 60}")
    print(f"  Runs: {summary['num_runs']}")
    print(f"  Total tokens:  mean={summary['total_tokens_mean']:.0f}, "
          f"stdev={summary['total_tokens_stdev']:.1f}")
    print(f"  Avg latency:   mean={summary['avg_latency_mean']}s, "
          f"stdev={summary['avg_latency_stdev']}s")
    print(f"  Total cost:    mean=${summary['total_cost_mean']}, "
          f"stdev=${summary['total_cost_stdev']}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a pipeline's benchmark 3x for reproducibility")
    parser.add_argument("--pipeline", choices=["llm_only", "basic_rag", "graph_rag", "all"],
                         default="all", help="Which pipeline to run 3x")
    parser.add_argument("--n", type=int, default=3, help="Number of runs (default 3)")
    args = parser.parse_args()

    questions = load_questions()

    pipelines_to_run = []
    if args.pipeline in ("llm_only", "all"):
        pipelines_to_run.append("llm_only")
    if args.pipeline in ("basic_rag", "all"):
        pipelines_to_run.append("basic_rag")
    if args.pipeline in ("graph_rag", "all"):
        pipelines_to_run.append("graph_rag")

    for name in pipelines_to_run:
        if name == "llm_only":
            from pipelines.llm_only import run_benchmark
        elif name == "basic_rag":
            from pipelines.basic_rag import run_benchmark
        elif name == "graph_rag":
            from pipelines.graph_rag import run_benchmark

        summary = run_n_times(name, run_benchmark, questions, n=args.n)
        print_summary(summary)