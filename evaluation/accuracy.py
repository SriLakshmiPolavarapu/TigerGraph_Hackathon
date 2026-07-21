"""
accuracy.py - Answer quality evaluation for all three pipelines.

Evaluation methods:

1. Lenient LLM-as-a-Judge: keyword-coverage-based PASS/FAIL (original, kept
   for reference / comparison with round 1 methodology).

2. Strict pass/fail (round 3 requirement): requires a complete, factually
   correct, and adequately supported answer.

3. Graded quality score (round 3 requirement): fully_correct / mostly_correct /
   partially_correct / incorrect.

4. BERTScore: semantic similarity between answer and expected answer.
   Config pinned per round 3 requirements: model_type="roberta-large",
   rescale_with_baseline=True, idf=False.

5. Tier breakdown (round 3 requirement: "correct handling of factual,
   numerical, policy, and multi-document questions" + "consistency of
   token reduction across question and document types"): all of the
   above, broken out by question tier (A/B/C/D) using the "tier" field
   in benchmark_questions3.json.

Bonus points thresholds:
  - LLM-as-a-Judge pass rate >= 90%
  - BERTScore F1 rescaled >= 0.55 or raw >= 0.88

No Gemini API calls needed. Everything runs locally or via free HF API.
"""

import json
import sys
import os
import time
import re
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_results(filepath: str) -> list:
    """Load pipeline results from JSON."""
    with open(filepath, "r") as f:
        return json.load(f)


def load_expected_answers(filepath: str) -> list:
    """Load expected answers from benchmark questions."""
    with open(filepath, "r") as f:
        return json.load(f)


NON_ANSWER_SIGNALS = [
    "has not yet", "have not yet", "not yet reported", "not yet available",
    "not been reported", "no official", "i don't know", "i do not know",
    "cannot determine", "unable to determine", "not enough information",
    "does not contain enough information", "insufficient information",
    "would be an estimate", "analyst estimates rather than",
]


# ══════════════════════════════════════════════════════════════
# Lenient LLM-as-a-Judge (Local, no API calls) - kept from round 1
# ══════════════════════════════════════════════════════════════

def keyword_judge(question: str, answer: str, expected: str) -> str:
    """Lenient PASS/FAIL: >= 40% keyword coverage."""
    expected_lower = expected.lower()
    answer_lower = answer.lower()

    words = re.findall(r'\b[a-z]{3,}\b', expected_lower)
    stop_words = {"the", "and", "for", "are", "but", "not", "you", "all",
                  "can", "had", "her", "was", "one", "our", "out", "has",
                  "that", "this", "from", "they", "been", "have", "its",
                  "will", "with", "each", "make", "like", "which", "their",
                  "such", "into", "than", "more", "also", "based", "using",
                  "between", "through", "specific", "example", "including",
                  "different", "other", "these", "those", "about"}
    key_words = [w for w in words if w not in stop_words and len(w) > 3]

    if not key_words:
        return "PASS"

    matches = sum(1 for w in key_words if w in answer_lower)
    coverage = matches / len(key_words)

    return "PASS" if coverage >= 0.4 else "FAIL"


def _keyword_coverage(answer: str, expected: str) -> float:
    """Shared helper: fraction of expected-answer key words found in answer."""
    expected_lower = expected.lower()
    answer_lower = answer.lower()

    words = re.findall(r'\b[a-z0-9]{2,}\b', expected_lower)
    stop_words = {"the", "and", "for", "are", "but", "not", "you", "all",
                  "can", "had", "her", "was", "one", "our", "out", "has",
                  "that", "this", "from", "they", "been", "have", "its",
                  "will", "with", "each", "make", "like", "which", "their",
                  "such", "into", "than", "more", "also", "based", "using",
                  "between", "through", "specific", "example", "including",
                  "different", "other", "these", "those", "about"}
    key_words = [w for w in words if w not in stop_words and len(w) > 1]

    if not key_words:
        return 1.0

    matches = sum(1 for w in key_words if w in answer_lower)
    return matches / len(key_words)


def hf_judge(question: str, answer: str, expected: str) -> str:
    """HF API judge with keyword fallback."""
    try:
        from huggingface_hub import InferenceClient

        client = InferenceClient()

        prompt = f"""Grade the following answer as PASS or FAIL.
PASS: The answer correctly addresses the question and covers key points.
FAIL: The answer is incorrect, misses critical points, or is irrelevant.

Question: {question}
Expected: {expected[:500]}
Answer: {answer[:500]}

Grade (PASS or FAIL):"""

        response = client.text_generation(
            prompt,
            model="microsoft/Phi-3-mini-4k-instruct",
            max_new_tokens=10,
        )

        result = response.strip().upper()
        if "PASS" in result:
            return "PASS"
        elif "FAIL" in result:
            return "FAIL"
        else:
            return keyword_judge(question, answer, expected)

    except Exception:
        return keyword_judge(question, answer, expected)


def run_llm_judge(results: list, expected_answers: list, pipeline_name: str) -> dict:
    """Lenient PASS/FAIL judge (kept from round 1 for comparison)."""
    print(f"\n  Judging {pipeline_name} (lenient)...")
    grades = []

    for i, (result, expected) in enumerate(zip(results, expected_answers)):
        question = result["question"]
        answer = result["answer"]
        expected_answer = expected.get("answer_key", "")

        grade = hf_judge(question, answer, expected_answer)
        grades.append(grade)
        print(f"    Q{i+1}: {grade}")

    pass_count = sum(1 for g in grades if g == "PASS")
    total = len(grades)
    pass_rate = (pass_count / total) * 100 if total > 0 else 0

    return {
        "pipeline": pipeline_name,
        "pass_count": pass_count,
        "total": total,
        "pass_rate": pass_rate,
        "grades": grades,
        "bonus": pass_rate >= 90,
    }


# ══════════════════════════════════════════════════════════════
# Strict pass/fail (round 3 requirement)
# ══════════════════════════════════════════════════════════════

def strict_judge(question: str, answer: str, expected: str) -> str:
    """Strict PASS/FAIL: rejects hedges/non-answers, requires >=75% coverage."""
    answer_lower = answer.lower()

    if any(signal in answer_lower for signal in NON_ANSWER_SIGNALS):
        return "FAIL"

    coverage = _keyword_coverage(answer, expected)
    return "PASS" if coverage >= 0.75 else "FAIL"


def run_strict_judge(results: list, expected_answers: list, pipeline_name: str) -> dict:
    """Strict PASS/FAIL judge required by round 3."""
    print(f"\n  Judging {pipeline_name} (strict)...")
    grades = []

    for i, (result, expected) in enumerate(zip(results, expected_answers)):
        answer = result["answer"]
        expected_answer = expected.get("answer_key", "")

        grade = strict_judge(result["question"], answer, expected_answer)
        grades.append(grade)
        print(f"    Q{i+1}: {grade}")

    pass_count = sum(1 for g in grades if g == "PASS")
    total = len(grades)
    pass_rate = (pass_count / total) * 100 if total > 0 else 0

    return {
        "pipeline": pipeline_name,
        "pass_count": pass_count,
        "total": total,
        "pass_rate": pass_rate,
        "grades": grades,
    }


# ══════════════════════════════════════════════════════════════
# Graded quality score (round 3 requirement)
# ══════════════════════════════════════════════════════════════

def graded_score(question: str, answer: str, expected: str) -> str:
    """Graded quality: fully_correct / mostly_correct / partially_correct / incorrect."""
    answer_lower = answer.lower()

    if any(signal in answer_lower for signal in NON_ANSWER_SIGNALS):
        return "incorrect"

    coverage = _keyword_coverage(answer, expected)

    if coverage >= 0.85:
        return "fully_correct"
    elif coverage >= 0.6:
        return "mostly_correct"
    elif coverage >= 0.3:
        return "partially_correct"
    else:
        return "incorrect"


def run_graded_scoring(results: list, expected_answers: list, pipeline_name: str) -> dict:
    """Graded quality scoring required by round 3."""
    print(f"\n  Grading {pipeline_name} (graded)...")
    grades = []

    for i, (result, expected) in enumerate(zip(results, expected_answers)):
        answer = result["answer"]
        expected_answer = expected.get("answer_key", "")

        grade = graded_score(result["question"], answer, expected_answer)
        grades.append(grade)
        print(f"    Q{i+1}: {grade}")

    total = len(grades)
    counts = {
        "fully_correct": grades.count("fully_correct"),
        "mostly_correct": grades.count("mostly_correct"),
        "partially_correct": grades.count("partially_correct"),
        "incorrect": grades.count("incorrect"),
    }
    distribution = {k: round((v / total) * 100, 1) if total else 0 for k, v in counts.items()}

    return {
        "pipeline": pipeline_name,
        "total": total,
        "counts": counts,
        "distribution_pct": distribution,
        "grades": grades,
    }


# ══════════════════════════════════════════════════════════════
# BERTScore (runs fully locally)
# ══════════════════════════════════════════════════════════════

def run_bertscore(results: list, expected_answers: list, pipeline_name: str) -> dict:
    """BERTScore with round 3 pinned config."""
    from bert_score import score as bert_score

    print(f"\n  Computing BERTScore for {pipeline_name}...")

    candidates = [r["answer"][:1000] for r in results]
    references = [e.get("answer_key", "") for e in expected_answers]

    P, R, F1 = bert_score(
        candidates,
        references,
        model_type="roberta-large",
        lang="en",
        rescale_with_baseline=True,
        idf=False,
        verbose=False,
    )

    f1_scores = F1.tolist()
    precision_scores = P.tolist()
    recall_scores = R.tolist()

    avg_f1 = sum(f1_scores) / len(f1_scores)
    avg_precision = sum(precision_scores) / len(precision_scores)
    avg_recall = sum(recall_scores) / len(recall_scores)

    return {
        "pipeline": pipeline_name,
        "avg_f1": round(avg_f1, 4),
        "avg_precision": round(avg_precision, 4),
        "avg_recall": round(avg_recall, 4),
        "f1_scores": [round(f, 4) for f in f1_scores],
        "bonus_rescaled": avg_f1 >= 0.55,
    }


# ══════════════════════════════════════════════════════════════
# Tier breakdown (round 3 requirement)
# ══════════════════════════════════════════════════════════════

def run_tier_breakdown(results: list, expected_answers: list, bert_result: dict, pipeline_name: str) -> dict:
    """
    Break down strict pass rate, graded distribution, BERTScore F1, AND
    token consumption by question tier (A/B/C/D), using the "tier" field
    in benchmark_questions3.json.

    Token breakdown addresses round 3's "consistency of token reduction
    across question and document types" requirement under Token Efficiency.

    Tier A: single-hop controls (low GraphRAG benefit expected)
    Tier B: two-hop bridges (medium)
    Tier C: multi-hop aggregation / corpus-scan (high)
    Tier D: cross-document & temporal (medium-high)
    """
    by_tier = defaultdict(lambda: {
        "indices": [], "strict_pass": 0, "graded": [], "f1_scores": [],
        "total_tokens": [], "context_tokens": [],
    })

    for i, expected in enumerate(expected_answers):
        tier = expected.get("tier", "UNKNOWN")
        answer = results[i]["answer"]
        expected_answer = expected.get("answer_key", "")

        by_tier[tier]["indices"].append(i)

        if strict_judge(results[i]["question"], answer, expected_answer) == "PASS":
            by_tier[tier]["strict_pass"] += 1

        by_tier[tier]["graded"].append(graded_score(results[i]["question"], answer, expected_answer))
        by_tier[tier]["f1_scores"].append(bert_result["f1_scores"][i])
        by_tier[tier]["total_tokens"].append(results[i].get("total_tokens", 0))
        by_tier[tier]["context_tokens"].append(results[i].get("context_tokens", 0))

    summary = {}
    for tier, data in sorted(by_tier.items()):
        n = len(data["indices"])
        avg_f1 = sum(data["f1_scores"]) / n if n else 0
        fully_correct_pct = round((data["graded"].count("fully_correct") / n) * 100, 1) if n else 0
        avg_total_tokens = round(sum(data["total_tokens"]) / n, 1) if n else 0
        avg_context_tokens = round(sum(data["context_tokens"]) / n, 1) if n else 0
        summary[tier] = {
            "count": n,
            "strict_pass_rate": round((data["strict_pass"] / n) * 100, 1) if n else 0,
            "fully_correct_pct": fully_correct_pct,
            "avg_bertscore_f1": round(avg_f1, 4),
            "avg_total_tokens": avg_total_tokens,
            "avg_context_tokens": avg_context_tokens,
        }

    return {"pipeline": pipeline_name, "by_tier": summary}


# ══════════════════════════════════════════════════════════════
# Citation and evidence quality (round 3 requirement, 15% of score)
# ══════════════════════════════════════════════════════════════

def _text_overlap(a: str, b: str, window: int = 200) -> float:
    """Rough overlap check: shared word fraction between two text snippets."""
    words_a = set(re.findall(r'\b[a-z0-9]{3,}\b', a[:window].lower()))
    words_b = set(re.findall(r'\b[a-z0-9]{3,}\b', b[:window].lower()))
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)


def run_citation_quality(results: list, pipeline_name: str) -> dict:
    """
    Score citation/evidence quality using the evidence + citations fields
    now saved per question (requires the updated metrics.py / pipeline
    files that pass evidence into create_metrics). LLM-Only has no
    evidence by design, so it's reported as not applicable.

    Metrics:
    - citation_rate: fraction of retrieved evidence sources actually
      cited in the answer (completeness)
    - citation_support_rate: of the citations claimed, how many actually
      overlap textually with the answer (correctness / whether cited
      evidence supports the answer)
    - duplicate_evidence_rate: fraction of retrieved evidence items that
      are near-duplicates of another item in the same question's
      evidence set (should be low; high = redundant context not
      being filtered out)
    """
    has_evidence = any(r.get("evidence") for r in results)
    if not has_evidence:
        return {
            "pipeline": pipeline_name,
            "applicable": False,
            "note": "No evidence field present in results (e.g. LLM-Only has no retrieval)",
        }

    citation_rates = []
    support_rates = []
    duplicate_rates = []

    for r in results:
        evidence = r.get("evidence", [])
        citations = r.get("citations", [])
        answer = r.get("answer", "")

        if evidence:
            citation_rates.append(len(citations) / len(evidence))
        else:
            citation_rates.append(0.0)

        if citations:
            supported = sum(
                1 for source_id in citations
                if any(
                    e.get("source_id") == source_id and _text_overlap(e.get("text", ""), answer) > 0.05
                    for e in evidence
                )
            )
            support_rates.append(supported / len(citations))
        else:
            support_rates.append(0.0)

        if len(evidence) > 1:
            dup_count = 0
            texts = [e.get("text", "") for e in evidence]
            for i in range(len(texts)):
                for j in range(i + 1, len(texts)):
                    if _text_overlap(texts[i], texts[j]) > 0.6:
                        dup_count += 1
            max_pairs = (len(texts) * (len(texts) - 1)) / 2
            duplicate_rates.append(dup_count / max_pairs if max_pairs else 0.0)
        else:
            duplicate_rates.append(0.0)

    n = len(results)
    return {
        "pipeline": pipeline_name,
        "applicable": True,
        "avg_citation_rate_pct": round((sum(citation_rates) / n) * 100, 1),
        "avg_citation_support_rate_pct": round((sum(support_rates) / n) * 100, 1),
        "avg_duplicate_evidence_rate_pct": round((sum(duplicate_rates) / n) * 100, 1),
    }


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(project_root, "results")
    data_dir = os.path.join(project_root, "data")

    print("=" * 60)
    print("Accuracy Evaluation (No Gemini API needed)")
    print("=" * 60)

    questions_file = os.path.join(data_dir, "benchmark_questions3.json")
    if not os.path.exists(questions_file):
        print(f"ERROR: {questions_file} not found.")
        sys.exit(1)

    expected = load_expected_answers(questions_file)

    pipelines = {
        "LLM-Only": os.path.join(results_dir, "pipeline1_llm_only.json"),
        "Basic RAG": os.path.join(results_dir, "pipeline2_basic_rag.json"),
        "GraphRAG": os.path.join(results_dir, "pipeline3_graphrag.json"),
    }

    all_results = {}
    for name, filepath in pipelines.items():
        if os.path.exists(filepath):
            all_results[name] = load_results(filepath)
        else:
            print(f"WARNING: {filepath} not found. Skipping {name}.")

    # ── Lenient LLM-as-a-Judge ────────────────────────────────
    print("\n" + "=" * 60)
    print("Part 1: LLM-as-a-Judge (lenient, round 1 style)")
    print("=" * 60)

    judge_results = {}
    for name, results in all_results.items():
        judge_results[name] = run_llm_judge(results, expected, name)

    print(f"\n{'=' * 60}")
    print("Lenient Judge Summary")
    print(f"{'=' * 60}")
    for name, jr in judge_results.items():
        bonus = " (BONUS)" if jr["bonus"] else ""
        print(f"  {name}: {jr['pass_count']}/{jr['total']} PASS "
              f"({jr['pass_rate']:.0f}%){bonus}")
    print(f"{'=' * 60}")

    # ── Strict pass/fail ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("Part 2: Strict Pass/Fail (round 3 requirement)")
    print("=" * 60)

    strict_results = {}
    for name, results in all_results.items():
        strict_results[name] = run_strict_judge(results, expected, name)

    print(f"\n{'=' * 60}")
    print("Strict Pass/Fail Summary")
    print(f"{'=' * 60}")
    for name, sr in strict_results.items():
        print(f"  {name}: {sr['pass_count']}/{sr['total']} PASS ({sr['pass_rate']:.0f}%)")
    print(f"{'=' * 60}")

    # ── Graded scoring ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Part 3: Graded Quality Score (round 3 requirement)")
    print("=" * 60)

    graded_results = {}
    for name, results in all_results.items():
        graded_results[name] = run_graded_scoring(results, expected, name)

    print(f"\n{'=' * 60}")
    print("Graded Quality Summary")
    print(f"{'=' * 60}")
    for name, gr in graded_results.items():
        dist = gr["distribution_pct"]
        print(f"  {name}:")
        print(f"    Fully correct:     {gr['counts']['fully_correct']} ({dist['fully_correct']}%)")
        print(f"    Mostly correct:    {gr['counts']['mostly_correct']} ({dist['mostly_correct']}%)")
        print(f"    Partially correct: {gr['counts']['partially_correct']} ({dist['partially_correct']}%)")
        print(f"    Incorrect:         {gr['counts']['incorrect']} ({dist['incorrect']}%)")
    print(f"{'=' * 60}")

    # ── BERTScore ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Part 4: BERTScore")
    print("=" * 60)

    bert_results = {}
    for name, results in all_results.items():
        bert_results[name] = run_bertscore(results, expected, name)

    print(f"\n{'=' * 60}")
    print("BERTScore Summary")
    print(f"{'=' * 60}")
    for name, br in bert_results.items():
        rescaled_bonus = " (RESCALED BONUS)" if br["bonus_rescaled"] else ""
        print(f"  {name}:")
        print(f"    F1 (rescaled): {br['avg_f1']:.4f}{rescaled_bonus}")
        print(f"    Precision:     {br['avg_precision']:.4f}")
        print(f"    Recall:        {br['avg_recall']:.4f}")
    print(f"{'=' * 60}")

    # ── Tier breakdown ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Part 5: Tier Breakdown (A=single-hop, B=two-hop, C=multi-hop, D=cross-doc/temporal)")
    print("=" * 60)

    tier_results = {}
    for name, results in all_results.items():
        tier_results[name] = run_tier_breakdown(results, expected, bert_results[name], name)

    print(f"\n{'=' * 60}")
    print("Tier Breakdown Summary")
    print(f"{'=' * 60}")
    for name, tr in tier_results.items():
        print(f"  {name}:")
        for tier, stats in tr["by_tier"].items():
            print(f"    Tier {tier} (n={stats['count']}): "
                  f"strict pass {stats['strict_pass_rate']}%, "
                  f"fully correct {stats['fully_correct_pct']}%, "
                  f"BERTScore F1 {stats['avg_bertscore_f1']:.4f}, "
                  f"avg tokens {stats['avg_total_tokens']}, "
                  f"avg context tokens {stats['avg_context_tokens']}")
    print(f"{'=' * 60}")

    # ── Citation and evidence quality ────────────────────────
    print("\n" + "=" * 60)
    print("Part 6: Citation and Evidence Quality (round 3 requirement)")
    print("=" * 60)

    citation_results = {}
    for name, results in all_results.items():
        citation_results[name] = run_citation_quality(results, name)

    print(f"\n{'=' * 60}")
    print("Citation and Evidence Quality Summary")
    print(f"{'=' * 60}")
    for name, cr in citation_results.items():
        if not cr["applicable"]:
            print(f"  {name}: N/A ({cr['note']})")
            continue
        print(f"  {name}:")
        print(f"    Citation rate (completeness):     {cr['avg_citation_rate_pct']}%")
        print(f"    Citation support rate (correctness): {cr['avg_citation_support_rate_pct']}%")
        print(f"    Duplicate evidence rate:           {cr['avg_duplicate_evidence_rate_pct']}%")
    print(f"{'=' * 60}")

    # ── Save Results ──────────────────────────────────────────
    eval_output = {
        "lenient_judge": judge_results,
        "strict_judge": strict_results,
        "graded_score": graded_results,
        "bertscore": bert_results,
        "tier_breakdown": tier_results,
        "citation_quality": citation_results,
    }

    output_path = os.path.join(results_dir, "accuracy_evaluation.json")
    with open(output_path, "w") as f:
        json.dump(eval_output, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # ── Final Summary ─────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("FINAL ACCURACY SUMMARY")
    print(f"{'=' * 60}")
    for name in all_results.keys():
        jr = judge_results[name]
        sr = strict_results[name]
        gr = graded_results[name]
        br = bert_results[name]
        print(f"\n  {name}:")
        print(f"    Lenient judge: {jr['pass_rate']:.0f}% pass rate")
        print(f"    Strict judge:  {sr['pass_rate']:.0f}% pass rate")
        print(f"    Fully correct: {gr['distribution_pct']['fully_correct']}%")
        print(f"    BERTScore F1 (rescaled): {br['avg_f1']:.4f}")
    print(f"{'=' * 60}")