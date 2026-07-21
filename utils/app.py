import streamlit as st
import json
import os
import pandas as pd
import plotly.graph_objects as go


st.set_page_config(
    page_title="GraphRAG Benchmark Dashboard - Round 3",
    layout="wide",
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        text-align: center;
        color: #888;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_results(filepath):
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return json.load(f)
    return None


def get_results_dir():
    possible = ["results", "../results", "results/"]
    for p in possible:
        if os.path.isdir(p):
            return p
    return "results"


results_dir = get_results_dir()
p1_results = load_results(os.path.join(results_dir, "pipeline1_llm_only.json"))
p2_results = load_results(os.path.join(results_dir, "pipeline2_basic_rag.json"))
p3_results = load_results(os.path.join(results_dir, "pipeline3_graphrag.json"))
accuracy_data = load_results(os.path.join(results_dir, "accuracy_evaluation.json"))
p2_ingestion = load_results(os.path.join(results_dir, "basic_rag_ingestion_cost.json"))
p3_ingestion = load_results(os.path.join(results_dir, "graph_rag_ingestion_cost.json"))

# ── Round 1 reference numbers (hardcoded from the live round 1 dashboard,
# 10 questions on 885 arXiv papers - raw per-question JSON not carried
# forward into this repo, so this is a static summary snapshot) ──
ROUND1 = {
    "LLM-Only":  {"total_tokens": 7842,  "avg_input": 53,   "avg_output": 731, "avg_context": 0,    "avg_latency": 6.80,  "total_cost": 0.00298},
    "Basic RAG": {"total_tokens": 25136, "avg_input": 2384, "avg_output": 130, "avg_context": 2239, "avg_latency": 11.89, "total_cost": 0.00290},
    "GraphRAG":  {"total_tokens": 6132,  "avg_input": 512,  "avg_output": 101, "avg_context": 427,  "avg_latency": 12.53, "total_cost": 0.00092},
    "num_questions": 10,
    "dataset": "885 arXiv papers",
}


st.markdown('<div class="main-header">🐯 GraphRAG Inference Benchmark</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Round 1 (arXiv) vs Round 3 (sp100 SEC filings) - '
            'generalizing GraphRAG to a heterogeneous, organizer-provided dataset</div>', unsafe_allow_html=True)

st.divider()

if not (p1_results and p2_results and p3_results):
    st.warning("Round 3 results not found. Run all three pipelines first:")
    st.code("""
python -m pipelines.llm_only
python -m pipelines.basic_rag
python -m pipelines.graph_rag --all
python -m evaluation.accuracy
    """)
    st.stop()


def calc_metrics(results):
    total_tokens = sum(r["total_tokens"] for r in results)
    avg_input = sum(r["input_tokens"] for r in results) / len(results)
    avg_output = sum(r["output_tokens"] for r in results) / len(results)
    avg_latency = sum(r["latency_seconds"] for r in results) / len(results)
    total_cost = sum(r["cost_usd"] for r in results)
    avg_context = sum(r.get("context_tokens", 0) for r in results) / len(results)
    return {
        "total_tokens": total_tokens, "avg_input": avg_input, "avg_output": avg_output,
        "avg_latency": avg_latency, "total_cost": total_cost, "avg_context": avg_context,
    }


m1 = calc_metrics(p1_results)
m2 = calc_metrics(p2_results)
m3 = calc_metrics(p3_results)

token_reduction = ((m2["total_tokens"] - m3["total_tokens"]) / m2["total_tokens"]) * 100
context_reduction = ((m2["avg_context"] - m3["avg_context"]) / m2["avg_context"]) * 100 if m2["avg_context"] > 0 else 0
cost_reduction = ((m2["total_cost"] - m3["total_cost"]) / m2["total_cost"]) * 100 if m2["total_cost"] > 0 else 0

r1 = ROUND1
r1_token_reduction = ((r1["Basic RAG"]["total_tokens"] - r1["GraphRAG"]["total_tokens"]) / r1["Basic RAG"]["total_tokens"]) * 100
r1_context_reduction = ((r1["Basic RAG"]["avg_context"] - r1["GraphRAG"]["avg_context"]) / r1["Basic RAG"]["avg_context"]) * 100
r1_cost_reduction = ((r1["Basic RAG"]["total_cost"] - r1["GraphRAG"]["total_cost"]) / r1["Basic RAG"]["total_cost"]) * 100

tab1, tab2, tab3, tab4 = st.tabs([
    "Round 1 vs Round 3", "Round 3 Accuracy", "Round 3 Efficiency", "Side-by-Side Answers"
])

# ══════════════════════════════════════════════════════════════
# TAB 1: Round 1 vs Round 3 comparison
# ══════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Did GraphRAG's efficiency edge generalize to a harder, unfamiliar dataset?")
    st.caption(f"Round 1: {r1['dataset']}, {r1['num_questions']} questions, self-selected data | "
               f"Round 3: sp100 SEC filings (400 filings, ~18.7M tokens), 50 organizer-provided questions")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Context Token Reduction (vs Basic RAG)",
                   f"{context_reduction:.0f}%",
                   delta=f"{context_reduction - r1_context_reduction:+.0f}pp vs Round 1 ({r1_context_reduction:.0f}%)")
    with col2:
        st.metric("Total Token Reduction (vs Basic RAG)",
                   f"{token_reduction:.0f}%",
                   delta=f"{token_reduction - r1_token_reduction:+.0f}pp vs Round 1 ({r1_token_reduction:.0f}%)")
    with col3:
        st.metric("Cost Reduction (vs Basic RAG)",
                   f"{cost_reduction:.0f}%",
                   delta=f"{cost_reduction - r1_cost_reduction:+.0f}pp vs Round 1 ({r1_cost_reduction:.0f}%)")

    st.markdown("**GraphRAG's efficiency edge over Basic RAG widened on the harder round 3 dataset**, "
                 "despite documents being ~90x longer on average and the domain shifting from research "
                 "papers to SEC filings.")

    st.divider()
    st.subheader("Full Summary: Round 1 vs Round 3, all pipelines, all metrics")
    st.caption("Same metrics tracked in both rounds. Change is Round 3 vs Round 1, "
               "negative = decreased, positive = increased.")

    metric_specs = [
        ("Total Tokens", "total_tokens", "{:,}", False),
        ("Avg Input Tokens", "avg_input", "{:.0f}", False),
        ("Avg Output Tokens", "avg_output", "{:.0f}", False),
        ("Avg Context Tokens", "avg_context", "{:.0f}", False),
        ("Avg Latency (s)", "avg_latency", "{:.2f}", False),
        ("Total Cost (USD)", "total_cost", "${:.5f}", True),
    ]

    round3_metrics = {"LLM-Only": m1, "Basic RAG": m2, "GraphRAG": m3}

    for pipeline in ["LLM-Only", "Basic RAG", "GraphRAG"]:
        st.markdown(f"**{pipeline}**")
        rows = []
        for label, key, fmt, is_cost in metric_specs:
            r1_val = ROUND1[pipeline][key]
            r3_val = round3_metrics[pipeline][key]
            pct_change = ((r3_val - r1_val) / r1_val * 100) if r1_val else 0
            r1_str = fmt.format(r1_val)
            r3_str = fmt.format(r3_val)
            rows.append({
                "Metric": label,
                "Round 1 (10Q)": r1_str,
                "Round 3 (50Q)": r3_str,
                "Change": f"{pct_change:+.0f}%",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("**Latency flipped**: GraphRAG went from the *slowest* pipeline in round 1 "
                 f"({r1['GraphRAG']['avg_latency']:.2f}s, slower than Basic RAG's {r1['Basic RAG']['avg_latency']:.2f}s) "
                 f"to the *fastest* in round 3 ({m3['avg_latency']:.2f}s), since financial topic keywords "
                 "match filing text directly far more often than round 1's broader AI/ML concept fallback search.")

    if accuracy_data:
        st.divider()
        st.subheader("The honest part round 1 never measured: accuracy")
        st.markdown("Round 1's dashboard only tracked tokens/cost/latency, it never scored whether "
                     "answers were actually correct. Round 3 required it, and the story is more "
                     "complicated than the efficiency numbers alone suggest:")

        strict = accuracy_data.get("strict_judge", {})
        bert = accuracy_data.get("bertscore", {})
        graded = accuracy_data.get("graded_score", {})

        acc_df = pd.DataFrame({
            "Pipeline": ["LLM-Only", "Basic RAG", "GraphRAG"],
            "Strict Pass Rate": [f"{strict.get(p, {}).get('pass_rate', 0):.0f}%" for p in ["LLM-Only", "Basic RAG", "GraphRAG"]],
            "Fully Correct": [f"{graded.get(p, {}).get('distribution_pct', {}).get('fully_correct', 0):.1f}%" for p in ["LLM-Only", "Basic RAG", "GraphRAG"]],
            "BERTScore F1": [f"{bert.get(p, {}).get('avg_f1', 0):.4f}" for p in ["LLM-Only", "Basic RAG", "GraphRAG"]],
        })
        st.dataframe(acc_df, use_container_width=True, hide_index=True)

        st.markdown("**GraphRAG trades some accuracy for its efficiency gains.** This was true from the "
                     "start, but it narrowed substantially after two fixes: adding native TigerGraph "
                     "vector search (a round 3 architecture requirement), and correcting an embedding "
                     "bug where filing vectors were built from SEC cover-page boilerplate instead of "
                     "actual financial content.")

# ══════════════════════════════════════════════════════════════
# TAB 2: Round 3 Accuracy (full detail)
# ══════════════════════════════════════════════════════════════
with tab2:
    if not accuracy_data:
        st.warning("Accuracy results not found. Run: `python -m evaluation.accuracy`")
    else:
        st.subheader("Answer Quality and Accuracy")

        lenient = accuracy_data.get("lenient_judge", {})
        strict = accuracy_data.get("strict_judge", {})
        graded = accuracy_data.get("graded_score", {})
        bert = accuracy_data.get("bertscore", {})
        tiers = accuracy_data.get("tier_breakdown", {})
        citation = accuracy_data.get("citation_quality", {})

        pipelines = ["LLM-Only", "Basic RAG", "GraphRAG"]
        colors = {"LLM-Only": "#ff6b6b", "Basic RAG": "#ffa726", "GraphRAG": "#66bb6a"}

        fig_acc = go.Figure()
        fig_acc.add_trace(go.Bar(name="Lenient Pass %", x=pipelines,
                                   y=[lenient.get(p, {}).get("pass_rate", 0) for p in pipelines]))
        fig_acc.add_trace(go.Bar(name="Strict Pass %", x=pipelines,
                                   y=[strict.get(p, {}).get("pass_rate", 0) for p in pipelines]))
        fig_acc.add_trace(go.Bar(name="Fully Correct %", x=pipelines,
                                   y=[graded.get(p, {}).get("distribution_pct", {}).get("fully_correct", 0) for p in pipelines]))
        fig_acc.update_layout(title="Accuracy Metrics by Pipeline", barmode="group",
                                yaxis_title="%", height=420, template="plotly_dark")
        st.plotly_chart(fig_acc, use_container_width=True)

        fig_bert = go.Figure(data=[go.Bar(
            x=pipelines, y=[bert.get(p, {}).get("avg_f1", 0) for p in pipelines],
            marker_color=[colors[p] for p in pipelines],
            text=[f"{bert.get(p, {}).get('avg_f1', 0):.4f}" for p in pipelines],
            textposition="auto",
        )])
        fig_bert.update_layout(title="BERTScore F1 (rescaled, roberta-large)",
                                 yaxis_title="F1", height=350, template="plotly_dark")
        st.plotly_chart(fig_bert, use_container_width=True)

        st.divider()
        st.subheader("Accuracy by Question Tier")
        st.caption("A: single-hop controls | B: two-hop bridges | C: multi-hop aggregation | D: cross-document/temporal")

        if tiers:
            tier_rows = []
            for p in pipelines:
                by_tier = tiers.get(p, {}).get("by_tier", {})
                for tier, stats in by_tier.items():
                    tier_rows.append({
                        "Pipeline": p, "Tier": tier, "N": stats["count"],
                        "Strict Pass %": stats["strict_pass_rate"],
                        "Fully Correct %": stats["fully_correct_pct"],
                        "BERTScore F1": stats["avg_bertscore_f1"],
                        "Avg Tokens": stats["avg_total_tokens"],
                    })
            tier_df = pd.DataFrame(tier_rows)
            st.dataframe(tier_df, use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("Evidence and Citation Quality")
        if citation:
            cite_rows = []
            for p in pipelines:
                c = citation.get(p, {})
                if c.get("applicable"):
                    cite_rows.append({
                        "Pipeline": p,
                        "Citation Rate (completeness)": f"{c['avg_citation_rate_pct']}%",
                        "Citation Support Rate (correctness)": f"{c['avg_citation_support_rate_pct']}%",
                        "Duplicate Evidence Rate": f"{c['avg_duplicate_evidence_rate_pct']}%",
                    })
                else:
                    cite_rows.append({
                        "Pipeline": p,
                        "Citation Rate (completeness)": "N/A",
                        "Citation Support Rate (correctness)": "N/A",
                        "Duplicate Evidence Rate": "N/A",
                    })
            st.dataframe(pd.DataFrame(cite_rows), use_container_width=True, hide_index=True)
            st.caption("LLM-Only is N/A by design, it has no retrieval, so there's no evidence to cite.")

# ══════════════════════════════════════════════════════════════
# TAB 3: Round 3 Efficiency (tokens, cost, latency, ingestion)
# ══════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Token Efficiency (Round 3, 50 questions)")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Token Reduction vs Basic RAG", f"{token_reduction:.0f}%")
    with col2:
        st.metric("Context Token Reduction vs Basic RAG", f"{context_reduction:.0f}%")
    with col3:
        st.metric("Cost Reduction vs Basic RAG", f"{cost_reduction:.0f}%")

    fig_tokens = go.Figure(data=[go.Bar(
        x=["LLM-Only", "Basic RAG", "GraphRAG"],
        y=[m1["total_tokens"], m2["total_tokens"], m3["total_tokens"]],
        marker_color=["#ff6b6b", "#ffa726", "#66bb6a"],
        text=[f"{m1['total_tokens']:,}", f"{m2['total_tokens']:,}", f"{m3['total_tokens']:,}"],
        textposition="auto",
    )])
    fig_tokens.update_layout(title="Total Tokens (50 questions)", yaxis_title="Tokens",
                               height=400, template="plotly_dark")
    st.plotly_chart(fig_tokens, use_container_width=True)

    st.divider()
    st.subheader("One-Time Ingestion Costs (reported separately from per-question inference costs)")
    ing_col1, ing_col2 = st.columns(2)
    with ing_col1:
        st.markdown("**Basic RAG (ChromaDB)**")
        if p2_ingestion:
            st.json(p2_ingestion)
        else:
            st.caption("Not found. Run `python -m pipelines.basic_rag` to generate.")
    with ing_col2:
        st.markdown("**GraphRAG (TigerGraph vector attribute)**")
        if p3_ingestion:
            st.json(p3_ingestion)
        else:
            st.caption("Not found. Run `python -m pipelines.graph_rag --setup` to generate.")

    st.divider()
    st.subheader("Summary Table")
    summary_df = pd.DataFrame({
        "Metric": ["Total Tokens", "Avg Input Tokens", "Avg Output Tokens",
                   "Avg Context Tokens", "Avg Latency (s)", "Total Cost (USD)"],
        "LLM-Only": [f"{m1['total_tokens']:,}", f"{m1['avg_input']:.0f}", f"{m1['avg_output']:.0f}",
                     "0", f"{m1['avg_latency']:.2f}", f"${m1['total_cost']:.5f}"],
        "Basic RAG": [f"{m2['total_tokens']:,}", f"{m2['avg_input']:.0f}", f"{m2['avg_output']:.0f}",
                      f"{m2['avg_context']:.0f}", f"{m2['avg_latency']:.2f}", f"${m2['total_cost']:.5f}"],
        "GraphRAG": [f"{m3['total_tokens']:,}", f"{m3['avg_input']:.0f}", f"{m3['avg_output']:.0f}",
                     f"{m3['avg_context']:.0f}", f"{m3['avg_latency']:.2f}", f"${m3['total_cost']:.5f}"],
    })
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════
# TAB 4: Side-by-side answers
# ══════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Side-by-Side Answers (Round 3)")

    question_list = [r["question"] for r in p1_results]
    selected_q = st.selectbox("Select a question:", question_list)

    if selected_q:
        idx = question_list.index(selected_q)
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Pipeline 1: LLM-Only**")
            st.caption(f"Tokens: {p1_results[idx]['total_tokens']} | "
                       f"Latency: {p1_results[idx]['latency_seconds']:.1f}s | "
                       f"Cost: ${p1_results[idx]['cost_usd']:.5f}")
            st.markdown(p1_results[idx]["answer"][:1000])

        with col2:
            st.markdown("**Pipeline 2: Basic RAG**")
            st.caption(f"Tokens: {p2_results[idx]['total_tokens']} | "
                       f"Latency: {p2_results[idx]['latency_seconds']:.1f}s | "
                       f"Cost: ${p2_results[idx]['cost_usd']:.5f}")
            st.markdown(p2_results[idx]["answer"][:1000])
            if p2_results[idx].get("citations"):
                st.caption(f"Cited: {', '.join(p2_results[idx]['citations'][:3])}")

        with col3:
            st.markdown("**Pipeline 3: GraphRAG**")
            st.caption(f"Tokens: {p3_results[idx]['total_tokens']} | "
                       f"Latency: {p3_results[idx]['latency_seconds']:.1f}s | "
                       f"Cost: ${p3_results[idx]['cost_usd']:.5f}")
            st.markdown(p3_results[idx]["answer"][:1000])
            if p3_results[idx].get("citations"):
                st.caption(f"Cited: {', '.join(p3_results[idx]['citations'][:3])}")

st.divider()
st.caption("Built by Sri for the TigerGraph GraphRAG Hackathon | "
           "Round 1: 885 arXiv papers | Round 3: 400 sp100 SEC filings | LLM: Gemini 2.5 Flash")