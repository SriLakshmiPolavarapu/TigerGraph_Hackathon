import streamlit as st
import json
import os
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


st.set_page_config(
    page_title="GraphRAG Benchmark Dashboard",
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
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        border: 1px solid #333;
    }
    .winner-badge {
        background: #00c853;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .pipeline-header {
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_results(filepath):
    """Load pipeline results from JSON file."""
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return json.load(f)
    return None


def get_results_dir():
    """Find the results directory."""
    possible = ["results", "../results", "results/"]
    for p in possible:
        if os.path.isdir(p):
            return p
    return "results"


results_dir = get_results_dir()
p1_results = load_results(os.path.join(results_dir, "pipeline1_llm_only.json"))
p2_results = load_results(os.path.join(results_dir, "pipeline2_basic_rag.json"))
p3_results = load_results(os.path.join(results_dir, "pipeline3_graphrag.json"))



st.markdown('<div class="main-header">🐯 GraphRAG Inference Benchmark</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Proving that GraphRAG cuts token costs without losing answer quality</div>', unsafe_allow_html=True)

st.divider()

# Overall Metrics
if p1_results and p2_results and p3_results:

    # Calculate aggregate metrics
    def calc_metrics(results):
        total_tokens = sum(r["total_tokens"] for r in results)
        avg_input = sum(r["input_tokens"] for r in results) / len(results)
        avg_output = sum(r["output_tokens"] for r in results) / len(results)
        avg_latency = sum(r["latency_seconds"] for r in results) / len(results)
        total_cost = sum(r["cost_usd"] for r in results)
        avg_context = sum(r.get("context_tokens", 0) for r in results) / len(results)
        return {
            "total_tokens": total_tokens,
            "avg_input": avg_input,
            "avg_output": avg_output,
            "avg_latency": avg_latency,
            "total_cost": total_cost,
            "avg_context": avg_context,
        }

    m1 = calc_metrics(p1_results)
    m2 = calc_metrics(p2_results)
    m3 = calc_metrics(p3_results)

    # Token reduction percentage
    token_reduction = ((m2["total_tokens"] - m3["total_tokens"]) / m2["total_tokens"]) * 100
    context_reduction = ((m2["avg_context"] - m3["avg_context"]) / m2["avg_context"]) * 100 if m2["avg_context"] > 0 else 0
    cost_reduction = ((m2["total_cost"] - m3["total_cost"]) / m2["total_cost"]) * 100 if m2["total_cost"] > 0 else 0

    # Key metrics row
    st.subheader("Key Results: GraphRAG vs Basic RAG")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Token Reduction",
            value=f"{token_reduction:.0f}%",
            delta="fewer tokens",
            delta_color="normal",
        )

    with col2:
        st.metric(
            label="Context Token Reduction",
            value=f"{context_reduction:.0f}%",
            delta="smaller context",
            delta_color="normal",
        )

    with col3:
        st.metric(
            label="Cost Reduction",
            value=f"{cost_reduction:.0f}%",
            delta="cheaper",
            delta_color="normal",
        )

    with col4:
        st.metric(
            label="GraphRAG Total Cost",
            value=f"${m3['total_cost']:.4f}",
            delta=f"vs ${m2['total_cost']:.4f} Basic RAG",
            delta_color="normal",
        )

    st.divider()

    # Comparison Chart
    st.subheader("Pipeline Comparison")

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        # Total tokens bar chart
        fig_tokens = go.Figure(data=[
            go.Bar(
                x=["LLM-Only", "Basic RAG", "GraphRAG"],
                y=[m1["total_tokens"], m2["total_tokens"], m3["total_tokens"]],
                marker_color=["#ff6b6b", "#ffa726", "#66bb6a"],
                text=[f"{m1['total_tokens']:,}", f"{m2['total_tokens']:,}", f"{m3['total_tokens']:,}"],
                textposition="auto",
            )
        ])
        fig_tokens.update_layout(
            title="Total Tokens (10 questions)",
            yaxis_title="Tokens",
            height=400,
            template="plotly_dark",
        )
        st.plotly_chart(fig_tokens, use_container_width=True)

    with chart_col2:
        # Average context tokens
        fig_context = go.Figure(data=[
            go.Bar(
                x=["LLM-Only", "Basic RAG", "GraphRAG"],
                y=[0, m2["avg_context"], m3["avg_context"]],
                marker_color=["#ff6b6b", "#ffa726", "#66bb6a"],
                text=["0", f"{m2['avg_context']:.0f}", f"{m3['avg_context']:.0f}"],
                textposition="auto",
            )
        ])
        fig_context.update_layout(
            title="Avg Context Tokens per Query",
            yaxis_title="Tokens",
            height=400,
            template="plotly_dark",
        )
        st.plotly_chart(fig_context, use_container_width=True)

    chart_col3, chart_col4 = st.columns(2)

    with chart_col3:
        # Cost comparison
        fig_cost = go.Figure(data=[
            go.Bar(
                x=["LLM-Only", "Basic RAG", "GraphRAG"],
                y=[m1["total_cost"], m2["total_cost"], m3["total_cost"]],
                marker_color=["#ff6b6b", "#ffa726", "#66bb6a"],
                text=[f"${m1['total_cost']:.4f}", f"${m2['total_cost']:.4f}", f"${m3['total_cost']:.4f}"],
                textposition="auto",
            )
        ])
        fig_cost.update_layout(
            title="Total Cost (10 questions)",
            yaxis_title="USD",
            height=400,
            template="plotly_dark",
        )
        st.plotly_chart(fig_cost, use_container_width=True)

    with chart_col4:
        # Latency comparison
        fig_latency = go.Figure(data=[
            go.Bar(
                x=["LLM-Only", "Basic RAG", "GraphRAG"],
                y=[m1["avg_latency"], m2["avg_latency"], m3["avg_latency"]],
                marker_color=["#ff6b6b", "#ffa726", "#66bb6a"],
                text=[f"{m1['avg_latency']:.1f}s", f"{m2['avg_latency']:.1f}s", f"{m3['avg_latency']:.1f}s"],
                textposition="auto",
            )
        ])
        fig_latency.update_layout(
            title="Avg Latency per Query",
            yaxis_title="Seconds",
            height=400,
            template="plotly_dark",
        )
        st.plotly_chart(fig_latency, use_container_width=True)

    st.divider()

    # ── Per-Question Breakdown ────────────────────────────────
    st.subheader("Per-Question Breakdown")

    # Build comparison dataframe
    questions = [r["question"][:60] + "..." for r in p1_results]

    df = pd.DataFrame({
        "Question": questions,
        "P1 Tokens": [r["total_tokens"] for r in p1_results],
        "P2 Tokens": [r["total_tokens"] for r in p2_results],
        "P3 Tokens": [r["total_tokens"] for r in p3_results],
        "P1 Cost": [f"${r['cost_usd']:.5f}" for r in p1_results],
        "P2 Cost": [f"${r['cost_usd']:.5f}" for r in p2_results],
        "P3 Cost": [f"${r['cost_usd']:.5f}" for r in p3_results],
    })

    st.dataframe(df, use_container_width=True, hide_index=True)

    # Per-question token chart
    fig_per_q = go.Figure()
    fig_per_q.add_trace(go.Bar(
        name="LLM-Only",
        x=[f"Q{i+1}" for i in range(10)],
        y=[r["total_tokens"] for r in p1_results],
        marker_color="#ff6b6b",
    ))
    fig_per_q.add_trace(go.Bar(
        name="Basic RAG",
        x=[f"Q{i+1}" for i in range(10)],
        y=[r["total_tokens"] for r in p2_results],
        marker_color="#ffa726",
    ))
    fig_per_q.add_trace(go.Bar(
        name="GraphRAG",
        x=[f"Q{i+1}" for i in range(10)],
        y=[r["total_tokens"] for r in p3_results],
        marker_color="#66bb6a",
    ))
    fig_per_q.update_layout(
        title="Tokens per Question - All Pipelines",
        barmode="group",
        yaxis_title="Tokens",
        height=450,
        template="plotly_dark",
    )
    st.plotly_chart(fig_per_q, use_container_width=True)

    st.divider()

    # ── Side-by-Side Answers ──────────────────────────────────
    st.subheader("Side-by-Side Answers")

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

        with col3:
            st.markdown("**Pipeline 3: GraphRAG**")
            st.caption(f"Tokens: {p3_results[idx]['total_tokens']} | "
                      f"Latency: {p3_results[idx]['latency_seconds']:.1f}s | "
                      f"Cost: ${p3_results[idx]['cost_usd']:.5f}")
            st.markdown(p3_results[idx]["answer"][:1000])

    st.divider()

    # Summary Table 
    st.subheader("Summary")

    summary_df = pd.DataFrame({
        "Metric": ["Total Tokens", "Avg Input Tokens", "Avg Output Tokens",
                   "Avg Context Tokens", "Avg Latency (s)", "Total Cost (USD)"],
        "LLM-Only": [
            f"{m1['total_tokens']:,}",
            f"{m1['avg_input']:.0f}",
            f"{m1['avg_output']:.0f}",
            "0",
            f"{m1['avg_latency']:.2f}",
            f"${m1['total_cost']:.5f}",
        ],
        "Basic RAG": [
            f"{m2['total_tokens']:,}",
            f"{m2['avg_input']:.0f}",
            f"{m2['avg_output']:.0f}",
            f"{m2['avg_context']:.0f}",
            f"{m2['avg_latency']:.2f}",
            f"${m2['total_cost']:.5f}",
        ],
        "GraphRAG": [
            f"{m3['total_tokens']:,}",
            f"{m3['avg_input']:.0f}",
            f"{m3['avg_output']:.0f}",
            f"{m3['avg_context']:.0f}",
            f"{m3['avg_latency']:.2f}",
            f"${m3['total_cost']:.5f}",
        ],
    })

    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # Key takeaways
    st.subheader("Key Takeaways")
    st.markdown(f"""
    - **GraphRAG uses {token_reduction:.0f}% fewer total tokens** than Basic RAG across all 10 questions
    - **Context is {context_reduction:.0f}% smaller**: GraphRAG sends ~{m3['avg_context']:.0f} tokens of structured context vs ~{m2['avg_context']:.0f} tokens of raw chunks
    - **Cost dropped {cost_reduction:.0f}%**: from ${m2['total_cost']:.4f} to ${m3['total_cost']:.4f}
    - GraphRAG achieves this by using **structured graph traversal** instead of brute-force vector similarity
    - The knowledge graph has **885 papers, 3,563 authors, and 102 concepts** connected by relationship edges
    """)

else:
    st.warning("Results not found. Run all three pipelines first:")
    st.code("""
python -m pipelines.llm_only
python -m pipelines.basic_rag
python -m pipelines.graph_rag --query
    """)

st.divider()
st.caption("Built by Sri for the GraphRAG Inference Hackathon by TigerGraph | "
           "Dataset: 885 arXiv papers | LLM: Gemini 2.5 Flash")
