"""
Pipeline 3: GraphRAG (TigerGraph + Gemini) - sp100 SEC filings version

FAST-PATH VERSION: keeps the existing filing-level embeddings (already
ingested), adds only Auditor entities and exact graph traversal on top.
This targets aggregation-style questions ("list all companies audited by
KPMG") that no similarity search, chunk-level or filing-level, can answer
correctly, without needing the slow full chunk-level re-ingestion.
"""

import json
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyTigerGraph import TigerGraphConnection
from sentence_transformers import SentenceTransformer
import config
from utils.gemini_client import generate
from utils.metrics import Timer, create_metrics, QueryMetrics
from preprocessing.data_loader import load_sp100_documents
import tiktoken


# ── TigerGraph Savanna Connection ─────────────────────────────
TG_HOST = config.TG_HOST
TG_USERNAME = config.TG_USERNAME
TG_PASSWORD = config.TG_PASSWORD
TG_JWT = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ3b3Jrc3BhY2UxIiwiaWF0IjoxNzg0NTkzNjcwLCJleHAiOjE3ODUxOTg0NzUsImlzcyI6IlRpZ2VyR3JhcGgifQ.B7iWEf7JDBCeDF6TpSDAP14TczNJQydQem8uF9w3qQk"
TG_GRAPH = "SP100GraphRAG"

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_DIM = 384

SYSTEM_PROMPT = """You are a helpful financial research assistant specializing in SEC filings.
Answer the question based ONLY on the provided graph context below.
The context contains entities and relationships extracted from a knowledge graph
of SEC filings (8-K, 10-K, DEF14A) for S&P 100 companies. Use these connections
to provide accurate, structured answers.
If the context doesn't contain enough information, say so.
Be concise and specific."""

COMMON_TOPICS = [
    "revenue", "net income", "earnings", "quarterly results", "annual report",
    "acquisition", "merger", "divestiture", "dividend", "stock buyback",
    "share repurchase", "litigation", "risk factors", "executive compensation",
    "board of directors", "shareholder", "proxy statement", "financial condition",
    "cash flow", "debt", "restructuring", "guidance", "forecast", "press release",
    "results of operations", "financial statements", "capital expenditure",
    "credit facility", "impairment", "goodwill", "segment", "outlook",
]

KNOWN_AUDITORS = ["Ernst & Young", "PricewaterhouseCoopers", "PwC", "Deloitte", "KPMG"]
AUDITOR_CANONICAL = {
    "ernst & young": "Ernst & Young",
    "pricewaterhousecoopers": "PricewaterhouseCoopers",
    "pwc": "PricewaterhouseCoopers",
    "deloitte": "Deloitte",
    "kpmg": "KPMG",
}

EXCERPT_CHARS = 1500
MAX_TOPIC_RESULTS = 5
MAX_VECTOR_RESULTS = 5
MAX_TOTAL_RESULTS = 8

_embed_model = None


def get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    return _embed_model


def get_connection(graph_name: str = "") -> TigerGraphConnection:
    conn = TigerGraphConnection(
        host=TG_HOST, username=TG_USERNAME, password=TG_PASSWORD,
        restppPort="443", gsPort="443", jwtToken=TG_JWT, graphname=graph_name,
    )
    return conn


# ══════════════════════════════════════════════════════════════
# SETUP: only add Auditor schema + fast extraction pass
# (Filing vertices, topics, and embeddings are already ingested -
#  this does NOT touch or re-ingest them)
# ══════════════════════════════════════════════════════════════

def add_auditor_schema():
    """Add Auditor vertex and AUDITED_BY edge to the existing graph."""
    conn = get_connection()
    schema_gsql = f"""
    USE GRAPH {TG_GRAPH}

    CREATE SCHEMA_CHANGE JOB add_auditor_job FOR GRAPH {TG_GRAPH} {{
        ADD VERTEX Auditor(PRIMARY_ID name STRING) WITH primary_id_as_attribute="true";
        ADD UNDIRECTED EDGE AUDITED_BY(FROM Company, TO Auditor);
    }}

    RUN SCHEMA_CHANGE JOB add_auditor_job
    DROP JOB add_auditor_job
    """
    print("Adding Auditor vertex/edge type...")
    try:
        result = conn.gsql(schema_gsql)
        print(result)
        print("Auditor schema added successfully!")
    except Exception as e:
        if "already exists" in str(e).lower():
            print("Auditor type already exists. Skipping.")
        else:
            print(f"Auditor schema error: {e}")


def extract_auditor_from_content(content: str) -> str:
    """Find a known Big Four / common auditor name in filing content."""
    content_lower = content.lower()
    for auditor in KNOWN_AUDITORS:
        if auditor.lower() in content_lower:
            return AUDITOR_CANONICAL[auditor.lower()]
    return None


def ingest_auditors():
    """
    Fast pass: scan all 400 filings for auditor names and add
    Company -AUDITED_BY-> Auditor edges. No embedding, no chunking,
    just text search - this is the fast alternative to a full re-ingest.
    """
    conn = get_connection(TG_GRAPH)

    print("\nLoading documents...")
    documents = load_sp100_documents()
    if not documents:
        print("ERROR: No documents found.")
        return

    print(f"\nScanning {len(documents)} filings for auditor names...")
    auditors_added = set()
    companies_tagged = set()

    for i, doc in enumerate(documents):
        content = doc["content"]
        ticker = doc["metadata"].get("ticker", "UNKNOWN")

        auditor = extract_auditor_from_content(content)
        if auditor and ticker not in companies_tagged:
            try:
                # Company vertex should already exist from the original
                # ingestion, but upsert is safe/idempotent either way.
                conn.upsertVertex("Company", ticker, {"ticker": ticker})
                conn.upsertVertex("Auditor", auditor, {"name": auditor})
                conn.upsertEdge("Company", ticker, "AUDITED_BY", "Auditor", auditor)
                auditors_added.add(auditor)
                companies_tagged.add(ticker)
            except Exception as e:
                print(f"  Error tagging auditor for {ticker}: {e}")

        if (i + 1) % 50 == 0:
            print(f"  Scanned {i + 1}/{len(documents)} filings, "
                  f"{len(companies_tagged)} companies tagged so far...")

    print(f"\nAuditor ingestion complete!")
    print(f"  Companies tagged: {len(companies_tagged)}")
    print(f"  Auditors found: {len(auditors_added)} ({', '.join(sorted(auditors_added))})")


# ══════════════════════════════════════════════════════════════
# QUERY FUNCTIONS
# ══════════════════════════════════════════════════════════════

def extract_entities_from_question(question: str, known_tickers: list = None) -> list:
    question_lower = question.lower()
    matched = [t for t in COMMON_TOPICS if t in question_lower]
    if known_tickers:
        for ticker in known_tickers:
            if ticker.lower() in question_lower or ticker in question:
                matched.append(ticker)
    if len(matched) < 2:
        stop_words = {"what", "how", "does", "the", "and", "are", "for", "with",
                      "that", "this", "from", "have", "been", "their", "between",
                      "main", "key", "role", "work", "explain", "compare",
                      "differences", "challenges", "concerns", "mitigated",
                      "improve", "accuracy", "can", "they", "its", "about"}
        words = [w.strip("?.,!") for w in question_lower.split()
                 if len(w) > 3 and w.strip("?.,!") not in stop_words]
        matched.extend(words[:3])
    return matched[:4]


def extract_auditor_from_question(question: str) -> list:
    """Detect which known auditor(s) a question is asking about."""
    question_lower = question.lower()
    found = []
    for auditor_key, canonical in AUDITOR_CANONICAL.items():
        if auditor_key in question_lower and canonical not in found:
            found.append(canonical)
    if "big four" in question_lower:
        found = list(dict.fromkeys(found + list(dict.fromkeys(AUDITOR_CANONICAL.values()))))
    return found


def get_companies_by_auditor(auditor_name: str, conn: TigerGraphConnection) -> list:
    """Exact graph traversal: every company audited by this auditor."""
    try:
        edges = conn.getEdges("Auditor", auditor_name, "AUDITED_BY")
        companies = []
        for edge in edges:
            ticker = edge.get("to_id", edge.get("from_id", ""))
            if ticker and ticker != auditor_name:
                companies.append(ticker)
        return sorted(set(companies))
    except Exception:
        return []


def query_graph(question: str, conn: TigerGraphConnection) -> dict:
    """
    Hybrid retrieval: Auditor traversal (exact, new) + existing filing-level
    vector search + topic/ticker graph traversal (unchanged from before).
    """
    context_parts = []
    found_filings = set()
    found_companies = set()
    found_topics = set()
    found_auditors = []

    # ── Auditor traversal (exact, complete - new) ──
    auditors_mentioned = extract_auditor_from_question(question)
    for auditor in auditors_mentioned:
        companies = get_companies_by_auditor(auditor, conn)
        if companies:
            found_auditors.append(auditor)
            context_parts.append(
                f"Auditor {auditor}: audits these {len(companies)} companies "
                f"in the dataset: {', '.join(companies)}"
            )

    # ── Filing-level vector search (existing, unchanged) ──
    embed_model = get_embed_model()
    query_vector = embed_model.encode(question).tolist()
    try:
        result = conn.runInstalledQuery(
            "vectorSearchFilings", params={"query_vector": query_vector, "k": MAX_VECTOR_RESULTS}
        )
        vertices = result[0].get("v", []) if len(result) > 0 else []
        for vtx in vertices:
            v_id = vtx.get("v_id", "")
            attrs = vtx.get("attributes", {})
            if v_id and v_id not in found_filings:
                found_filings.add(v_id)
                context_parts.append(
                    f"Filing: {v_id} ({attrs.get('filing_type','')}) [vector match]\n"
                    f"Excerpt: {attrs.get('content','')[:EXCERPT_CHARS]}"
                )
    except Exception as e:
        print(f"    Vector search error: {e}")

    # ── Topic/ticker graph traversal (existing, unchanged) ──
    entities = extract_entities_from_question(question)
    for entity in entities:
        if len(context_parts) >= MAX_TOTAL_RESULTS + len(found_auditors):
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
                                found_filings.add(filing_id)
                                context_parts.append(
                                    f"Filing: {filing_id} ({attrs.get('filing_type','')}) [topic match]\n"
                                    f"Excerpt: {attrs.get('content','')[:EXCERPT_CHARS]}"
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
                                found_filings.add(filing_id)
                                context_parts.append(
                                    f"Filing: {filing_id} ({attrs.get('filing_type','')}) [company match]\n"
                                    f"Excerpt: {attrs.get('content','')[:EXCERPT_CHARS]}"
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
                content = attrs.get("content", "")[:EXCERPT_CHARS]
                if content:
                    context_parts.append(f"Filing: ({attrs.get('filing_type','')})\nExcerpt: {content}")
        except Exception:
            pass

    return {
        "filings": list(found_filings), "companies": list(found_companies),
        "topics": list(found_topics), "auditors": found_auditors,
        "details": context_parts[:MAX_TOTAL_RESULTS + len(found_auditors)],
    }


def build_graph_prompt(question: str, graph_context: dict) -> str:
    parts = []
    if graph_context.get("topics"):
        parts.append(f"Related Topics: {', '.join(graph_context['topics'][:10])}")
    if graph_context.get("companies"):
        parts.append(f"Companies: {', '.join(graph_context['companies'][:10])}")
    if graph_context["details"]:
        parts.append("Relevant Information:\n" + "\n---\n".join(graph_context["details"]))
    context = "\n\n".join(parts) if parts else "No specific graph context found."
    return f"""Graph Context:
{context}

Question: {question}

Answer based on the graph context above. Use the relationships between
companies, filings, topics, and auditors to provide a well-connected answer."""


def run_query(question: str, conn: TigerGraphConnection = None) -> QueryMetrics:
    if conn is None:
        conn = get_connection(TG_GRAPH)
    encoder = tiktoken.get_encoding("cl100k_base")
    error = None
    try:
        with Timer() as timer:
            graph_context = query_graph(question, conn)
            prompt = build_graph_prompt(question, graph_context)
            response = generate(prompt=prompt, system_instruction=SYSTEM_PROMPT)
    except Exception as e:
        error = str(e)
        graph_context = {"filings": [], "companies": [], "topics": [], "auditors": [], "details": []}
        prompt = ""
        response = {"answer": "", "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        timer = Timer()
        timer.elapsed = 0.0

    prompt_tokens = len(encoder.encode(prompt)) if prompt else 0
    question_tokens = len(encoder.encode(question))
    context_tokens = max(prompt_tokens - question_tokens, 0)

    evidence = []
    for detail in graph_context["details"]:
        lines = detail.split("\n", 1)
        header = lines[0] if lines else ""
        excerpt = lines[1].replace("Excerpt: ", "", 1) if len(lines) > 1 else header
        source_id = header.split(" [")[0].strip()
        evidence.append({"source_id": source_id, "text": excerpt})

    metrics = create_metrics(
        pipeline_name="GraphRAG", question=question, llm_response=response,
        latency=timer.elapsed, context_chunks=len(graph_context["details"]),
        context_tokens=context_tokens, system_instruction=SYSTEM_PROMPT,
        evidence=evidence, error=error,
    )
    return metrics


def run_benchmark(questions: list) -> list:
    conn = get_connection(TG_GRAPH)
    results = []
    print(f"\nRunning Pipeline 3 (GraphRAG) on {len(questions)} questions...")
    print("-" * 60)
    for i, question in enumerate(questions, 1):
        print(f"\n[{i}/{len(questions)}] {question[:80]}...")
        metrics = run_query(question, conn)
        print(f"  Tokens: {metrics.total_tokens} (in: {metrics.input_tokens}, out: {metrics.output_tokens})")
        print(f"  Context: {metrics.context_chunks} graph results, ~{metrics.context_tokens} tokens")
        print(f"  Latency: {metrics.latency_seconds:.2f}s")
        print(f"  Cost: ${metrics.cost_usd:.6f}")
        results.append(metrics)
        time.sleep(2)

    total_tokens = sum(m.total_tokens for m in results)
    avg_latency = sum(m.latency_seconds for m in results) / len(results)
    total_cost = sum(m.cost_usd for m in results)
    avg_context = sum(m.context_tokens for m in results) / len(results)
    print(f"\n{'=' * 60}\nPipeline 3 (GraphRAG) Summary\n{'=' * 60}")
    print(f"  Questions:      {len(results)}")
    print(f"  Total tokens:   {total_tokens:,}")
    print(f"  Avg context:    {avg_context:.0f} tokens")
    print(f"  Avg latency:    {avg_latency:.2f}s")
    print(f"  Total cost:     ${total_cost:.6f}")
    print(f"{'=' * 60}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Pipeline 3: GraphRAG (sp100, fast auditor-only path)")
    parser.add_argument("--setup", action="store_true", help="Add Auditor schema + scan filings (fast, no re-embedding)")
    parser.add_argument("--query", action="store_true", help="Run benchmark queries")
    parser.add_argument("--all", action="store_true", help="Setup + query")
    args = parser.parse_args()

    if args.setup or args.all:
        print("=" * 60)
        print("Pipeline 3: GraphRAG Fast Setup (Auditor only)")
        print("=" * 60)
        print("\n--- Adding Auditor Schema ---")
        add_auditor_schema()
        print("\n--- Scanning Filings for Auditors ---")
        ingest_auditors()

    if args.query or args.all:
        print("\n" + "=" * 60)
        print("Pipeline 3: GraphRAG Benchmark (sp100)")
        print("=" * 60)
        questions_file = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "benchmark_questions3.json"
        )
        if os.path.exists(questions_file):
            with open(questions_file, "r") as f:
                data = json.load(f)
                questions = [q["question"] for q in data]
            print(f"Loaded {len(questions)} questions")
        else:
            print(f"ERROR: {questions_file} not found.")
            sys.exit(1)

        results = run_benchmark(questions)
        results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
        os.makedirs(results_dir, exist_ok=True)
        output = [m.to_dict() for m in results]
        output_path = os.path.join(results_dir, "pipeline3_graphrag.json")
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {output_path}")

    if not (args.setup or args.query or args.all):
        print("Usage:")
        print("  python -m pipelines.graph_rag --setup   # Add Auditor schema + scan filings (fast)")
        print("  python -m pipelines.graph_rag --query   # Run benchmark queries")
        print("  python -m pipelines.graph_rag --all     # Both")