"""
Pipeline 3: GraphRAG (TigerGraph + Gemini) - sp100 SEC filings version

HOW IT WORKS:
1. SETUP (one-time):
   - Connect to TigerGraph Savanna
   - Create a graph schema (Company, Filing, Topic, relationships)
   - Add a vector attribute to Filing for embeddings (native TigerGraph vector search)
   - Ingest sp100 SEC filings: extract entities, relationships, and embeddings
   - Store them as vertices, edges, and vectors in the graph
   - Install a GSQL query that performs vector similarity search

2. QUERY TIME (per question):
   - Embed the question using the same model used for Filing content (all-MiniLM-L6-v2)
   - Run vector search inside TigerGraph to find the most semantically similar filings
   - Also extract topics/tickers from the question and traverse graph relationships
   - Merge both result sets into one structured context (true hybrid retrieval)
   - Send structured context + question to Gemini
   - Gemini answers using the combined graph + vector context

WHY THIS IS BETTER THAN PIPELINE 2 (Basic RAG):
Basic RAG retrieves similar text chunks from a separate vector store (ChromaDB).
GraphRAG here does vector similarity search AND graph relationship traversal,
both natively inside TigerGraph Savanna, so the comparison isolates the
contribution of graph structure on top of the same embedding-based retrieval.
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
# NOTE: This JWT expires July 28, 2026. If it stops working after that,
# regenerate one (createSecret() + getToken()) and swap it in here.
TG_HOST = config.TG_HOST
TG_USERNAME = config.TG_USERNAME
TG_PASSWORD = config.TG_PASSWORD
TG_JWT = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ3b3Jrc3BhY2UxIiwiaWF0IjoxNzg0NTkzNjcwLCJleHAiOjE3ODUxOTg0NzUsImlzcyI6IlRpZ2VyR3JhcGgifQ.B7iWEf7JDBCeDF6TpSDAP14TczNJQydQem8uF9w3qQk"
TG_GRAPH = "SP100GraphRAG"

# Embedding model - MUST match Basic RAG's embedding model (config.EMBEDDING_MODEL)
# so the round 3 comparison isolates the contribution of graph retrieval.
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_DIM = 384

SYSTEM_PROMPT = """You are a helpful financial research assistant specializing in SEC filings.
Answer the question based ONLY on the provided graph context below.
The context contains entities and relationships extracted from a knowledge graph
of SEC filings (8-K, 10-K, DEF14A) for S&P 100 companies. Use these connections
to provide accurate, structured answers.
If the context doesn't contain enough information, say so.
Be concise and specific."""

# Financial topics to extract from filings and questions.
COMMON_TOPICS = [
    "revenue", "net income", "earnings", "quarterly results", "annual report",
    "acquisition", "merger", "divestiture", "dividend", "stock buyback",
    "share repurchase", "litigation", "risk factors", "executive compensation",
    "board of directors", "shareholder", "proxy statement", "financial condition",
    "cash flow", "debt", "restructuring", "guidance", "forecast", "press release",
    "results of operations", "financial statements", "capital expenditure",
    "credit facility", "impairment", "goodwill", "segment", "outlook",
]

# Context sizing.
EXCERPT_CHARS = 1500
MAX_TOPIC_RESULTS = 5
MAX_VECTOR_RESULTS = 5
MAX_TOTAL_RESULTS = 8

_embed_model = None


def get_embed_model() -> SentenceTransformer:
    """Load (once) the embedding model used for Filing content and questions."""
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    return _embed_model


def get_connection(graph_name: str = "") -> TigerGraphConnection:
    """Create a connection to TigerGraph Savanna."""
    conn = TigerGraphConnection(
        host=TG_HOST,
        username=TG_USERNAME,
        password=TG_PASSWORD,
        restppPort="443",
        gsPort="443",
        jwtToken=TG_JWT,
        graphname=graph_name,
    )
    return conn


# ══════════════════════════════════════════════════════════════
# SETUP FUNCTIONS (run once to build the knowledge graph)
# ══════════════════════════════════════════════════════════════

def create_schema():
    """
    Create the graph schema in TigerGraph.

    Schema:
    - Company (vertex): ticker
    - Filing (vertex): filing_id, filing_type, filing_date, content (truncated),
      content_embedding (vector attribute, 384-dim, for native TigerGraph vector search)
    - Topic (vertex): name (financial topic extracted from filing text)
    - FILED (edge): Company -> Filing
    - COVERS_TOPIC (edge): Filing -> Topic
    - RELATED_TO (edge): Topic -> Topic
    """
    conn = get_connection()

    schema_gsql = f"""
    CREATE GRAPH {TG_GRAPH}()

    USE GRAPH {TG_GRAPH}

    CREATE SCHEMA_CHANGE JOB schema_job {{
        ADD VERTEX Company(PRIMARY_ID ticker STRING) WITH primary_id_as_attribute="true";
        ADD VERTEX Filing(PRIMARY_ID filing_id STRING, filing_type STRING, filing_date STRING, content STRING) WITH primary_id_as_attribute="true";
        ADD VERTEX Topic(PRIMARY_ID name STRING) WITH primary_id_as_attribute="true";
        ADD UNDIRECTED EDGE FILED(FROM Company, TO Filing);
        ADD UNDIRECTED EDGE COVERS_TOPIC(FROM Filing, TO Topic);
        ADD UNDIRECTED EDGE RELATED_TO(FROM Topic, TO Topic);
    }}

    RUN SCHEMA_CHANGE JOB schema_job
    DROP JOB schema_job
    """

    print("Creating graph schema...")
    try:
        result = conn.gsql(schema_gsql)
        print(result)
        print("Schema created successfully!")
    except Exception as e:
        if "already exists" in str(e).lower():
            print(f"Graph '{TG_GRAPH}' already exists. Skipping schema creation.")
        else:
            print(f"Schema creation error: {e}")
            print("If graph already exists, this is fine.")


def add_vector_schema():
    """
    Add a vector attribute to Filing vertices, so TigerGraph Savanna acts
    as both graph database and vector database (required for round 3).
    Must run after create_schema(), since Filing must already exist.
    """
    conn = get_connection()

    vector_gsql = f"""
    USE GRAPH {TG_GRAPH}

    CREATE SCHEMA_CHANGE JOB add_vector_job FOR GRAPH {TG_GRAPH} {{
        ALTER VERTEX Filing ADD VECTOR ATTRIBUTE content_embedding(DIMENSION={EMBED_DIM}, METRIC="COSINE");
    }}

    RUN SCHEMA_CHANGE JOB add_vector_job
    DROP JOB add_vector_job
    """

    print("Adding vector attribute to Filing...")
    try:
        result = conn.gsql(vector_gsql)
        print(result)
        print("Vector attribute added successfully!")
    except Exception as e:
        if "already exists" in str(e).lower():
            print("Vector attribute already exists. Skipping.")
        else:
            print(f"Vector schema error: {e}")


def install_vector_search_query():
    """
    Install a GSQL query that performs native vector similarity search
    over Filing.content_embedding using TigerGraph's built-in vectorSearch().
    """
    conn = get_connection(TG_GRAPH)

    query_gsql = f"""
    USE GRAPH {TG_GRAPH}

    CREATE OR REPLACE QUERY vectorSearchFilings(LIST<float> query_vector, INT k) SYNTAX v3 {{
        MapAccum<VERTEX, FLOAT> @@distances;
        v = vectorSearch({{Filing.content_embedding}}, query_vector, k, {{distance_map: @@distances}});
        PRINT v;
        PRINT @@distances;
    }}

    INSTALL QUERY vectorSearchFilings
    """

    print("Installing vector search query...")
    try:
        result = conn.gsql(query_gsql)
        print(result)
        print("Vector search query installed successfully!")
    except Exception as e:
        print(f"Query install error: {e}")


def ingest_filings():
    """
    Load sp100 SEC filings and ingest them into TigerGraph as a knowledge graph.
    Extracts: Companies, Filings (with content embeddings), Topics, and their
    relationships. Reports one-time ingestion cost (embedding time/count)
    separately from per-question inference costs, per round 3 requirements.
    """
    conn = get_connection(TG_GRAPH)
    embed_model = get_embed_model()

    ingestion_start = time.perf_counter()

    # Load documents
    print("\nLoading documents...")
    documents = load_sp100_documents()

    if not documents:
        print("ERROR: No documents found. Check data/sp100_dataset/ exists.")
        return

    print(f"\nIngesting {len(documents)} filings into TigerGraph (with embeddings)...")

    filings_added = 0
    companies_added = set()
    topics_added = set()

    for i, doc in enumerate(documents):
        content = doc["content"]
        metadata = doc["metadata"]

        ticker = metadata.get("ticker", "UNKNOWN")
        filing_type = metadata.get("filing_type", "UNKNOWN")
        filing_date = metadata.get("filing_date", "UNKNOWN")
        # Unique filing id across tickers (filenames repeat per ticker, e.g. "8-K_2026-04-30")
        filing_id = f"{ticker}_{doc['filename'].replace('.txt', '')}"

        # Upsert Company vertex
        try:
            conn.upsertVertex("Company", ticker, {"ticker": ticker})
            companies_added.add(ticker)
        except Exception as e:
            print(f"  Error upserting company {ticker}: {e}")
            continue

        # Compute embedding for a window of actual financial content.
        # IMPORTANT: all-MiniLM-L6-v2 has a ~256 token (~1,200 char) max
        # sequence length and silently truncates anything beyond that, so
        # simply widening the slice (e.g. content[2000:15000]) doesn't
        # help, the model only ever "sees" the first ~1,200 characters of
        # whatever slice is passed in. What matters is WHICH ~1,200
        # characters we pick. Search for a marker that signals actual
        # financial content (MD&A / results of operations / net sales)
        # and embed starting there instead of a blind offset.
        FINANCIAL_MARKERS = [
            "results of operations", "management's discussion",
            "net sales", "total revenue", "item 7", "item 2.02",
        ]
        content_lower_search = content.lower()
        marker_pos = -1
        for marker in FINANCIAL_MARKERS:
            pos = content_lower_search.find(marker, 500)  # skip past the very start
            if pos != -1 and (marker_pos == -1 or pos < marker_pos):
                marker_pos = pos

        if marker_pos != -1:
            embed_source = content[marker_pos:marker_pos + 1200]
        elif len(content) > 2000:
            embed_source = content[2000:3200]
        else:
            embed_source = content

        # What's stored/displayed to the LLM can be a bit wider than the
        # embedding window itself, since the LLM doesn't have the same
        # truncation limit.
        if marker_pos != -1:
            stored_content = content[marker_pos:marker_pos + 3000]
        elif len(content) > 2000:
            stored_content = content[2000:5000]
        else:
            stored_content = content

        content_embedding = embed_model.encode(embed_source).tolist()

        # Upsert Filing vertex (content truncated for storage, plus embedding)
        try:
            conn.upsertVertex("Filing", filing_id, {
                "filing_type": filing_type,
                "filing_date": filing_date,
                "content": stored_content,
                "content_embedding": content_embedding,
            })
            conn.upsertEdge("Company", ticker, "FILED", "Filing", filing_id)
        except Exception as e:
            print(f"  Error upserting filing {filing_id}: {e}")
            continue

        # Extract financial topics present in the filing text
        # (scan a bounded window of the content for speed on large filings)
        content_lower = content[:20000].lower()
        local_topics = [t for t in COMMON_TOPICS if t in content_lower]

        # Also tag the filing type itself as a topic (8-K, 10-K, DEF14A)
        local_topics.append(filing_type.lower())

        for topic in local_topics[:10]:
            try:
                conn.upsertVertex("Topic", topic, {"name": topic})
                conn.upsertEdge("Filing", filing_id, "COVERS_TOPIC", "Topic", topic)
                topics_added.add(topic)
            except Exception:
                pass

        filings_added += 1
        if filings_added % 50 == 0:
            print(f"  Ingested {filings_added}/{len(documents)} filings...")

    print(f"\nIngestion complete!")
    print(f"  Filings: {filings_added}")
    print(f"  Companies: {len(companies_added)}")
    print(f"  Topics: {len(topics_added)}")

    # ── One-time ingestion cost disclosure (round 3 requirement) ──
    # Embedding runs locally via sentence-transformers, so there's no
    # per-token API cost, but the time/compute cost is real and reported
    # here separately from per-question inference costs.
    ingestion_elapsed = time.perf_counter() - ingestion_start
    ingestion_report = {
        "filings_ingested": filings_added,
        "companies": len(companies_added),
        "topics": len(topics_added),
        "embedding_time_seconds": round(ingestion_elapsed, 2),
        "embedding_model": EMBED_MODEL_NAME,
        "embedding_api_cost_usd": 0.0,  # local model, no API cost
    }
    results_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"
    )
    os.makedirs(results_dir, exist_ok=True)
    ingestion_report_path = os.path.join(results_dir, "graph_rag_ingestion_cost.json")
    with open(ingestion_report_path, "w") as f:
        json.dump(ingestion_report, f, indent=2)
    print(f"Ingestion cost report saved to {ingestion_report_path}")


# ══════════════════════════════════════════════════════════════
# QUERY FUNCTIONS (run per question)
# ══════════════════════════════════════════════════════════════

def extract_entities_from_question(question: str, known_tickers: list = None) -> list:
    """Extract searchable entities (tickers, financial topics) from a question."""
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


def vector_search_filings(question: str, conn: TigerGraphConnection, k: int = MAX_VECTOR_RESULTS) -> list:
    """
    Run native vector similarity search inside TigerGraph to find the
    Filing vertices most semantically similar to the question.
    Returns a list of dicts: {filing_id, filing_type, content, distance}.
    """
    embed_model = get_embed_model()
    query_vector = embed_model.encode(question).tolist()

    try:
        result = conn.runInstalledQuery(
            "vectorSearchFilings",
            params={"query_vector": query_vector, "k": k},
        )
    except Exception as e:
        print(f"    Vector search error: {e}")
        return []

    matches = []
    try:
        # result is a list of dicts, one per PRINT statement in the query.
        # First PRINT is the vertex set "v", second is the distance map.
        vertices = result[0].get("v", []) if len(result) > 0 else []
        distances = result[1].get("@@distances", {}) if len(result) > 1 else {}

        for vtx in vertices:
            v_id = vtx.get("v_id", "")
            attrs = vtx.get("attributes", {})
            distance = distances.get(v_id, None)
            matches.append({
                "filing_id": v_id,
                "filing_type": attrs.get("filing_type", ""),
                "content": attrs.get("content", "")[:EXCERPT_CHARS],
                "distance": distance,
            })
    except Exception as e:
        print(f"    Vector search result parsing error: {e}")

    return matches


def query_graph(question: str, conn: TigerGraphConnection) -> dict:
    """
    Hybrid retrieval: combine native vector similarity search (semantic match
    on filing content) with graph relationship traversal (topic/ticker based).

    1. Vector search: embed the question, find semantically similar Filings
       directly via TigerGraph's built-in vectorSearch().
    2. Graph traversal: extract topics/tickers from the question, walk
       Company -FILED-> Filing -COVERS_TOPIC-> Topic relationships.
    3. Merge both result sets, deduplicated by filing_id.
    """
    context_parts = []
    found_filings = set()
    found_companies = set()
    found_topics = set()

    # ── 1. Vector search (semantic similarity, native to TigerGraph) ──
    vector_matches = vector_search_filings(question, conn, k=MAX_VECTOR_RESULTS)
    for match in vector_matches:
        filing_id = match["filing_id"]
        if filing_id and filing_id not in found_filings:
            found_filings.add(filing_id)
            context_parts.append(
                f"Filing: {filing_id} ({match['filing_type']}) [vector match]\n"
                f"Excerpt: {match['content']}"
            )

    # ── 2. Graph relationship traversal (topics/tickers) ──
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
                                try:
                                    company_edges = conn.getEdges("Filing", filing_id, "FILED")
                                    for ce in company_edges[:2]:
                                        ticker = ce.get("to_id", ce.get("from_id", ""))
                                        if ticker:
                                            found_companies.add(ticker)
                                except Exception:
                                    pass
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

    # ── 3. Fallback if nothing found at all ──
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

    context = {
        "filings": list(found_filings),
        "companies": list(found_companies),
        "topics": list(found_topics),
        "details": context_parts[:MAX_TOTAL_RESULTS],
    }

    return context


def build_graph_prompt(question: str, graph_context: dict) -> str:
    """
    Build a prompt from graph-derived context (vector matches + graph traversal).
    """
    parts = []

    if graph_context["topics"]:
        parts.append(f"Related Topics: {', '.join(graph_context['topics'][:10])}")

    if graph_context["companies"]:
        parts.append(f"Companies: {', '.join(graph_context['companies'][:10])}")

    if graph_context["details"]:
        parts.append("Relevant Filings:\n" + "\n---\n".join(graph_context["details"]))

    context = "\n\n".join(parts) if parts else "No specific graph context found."

    prompt = f"""Graph Context:
{context}

Question: {question}

Answer based on the graph context above. Use the relationships between
companies, filings, and topics to provide a well-connected answer."""

    return prompt


def run_query(question: str, conn: TigerGraphConnection = None) -> QueryMetrics:
    """Run a single question through the GraphRAG pipeline."""
    if conn is None:
        conn = get_connection(TG_GRAPH)

    encoder = tiktoken.get_encoding("cl100k_base")
    error = None

    try:
        with Timer() as timer:
            graph_context = query_graph(question, conn)
            prompt = build_graph_prompt(question, graph_context)
            response = generate(
                prompt=prompt,
                system_instruction=SYSTEM_PROMPT,
            )
    except Exception as e:
        error = str(e)
        graph_context = {"filings": [], "companies": [], "topics": [], "details": []}
        prompt = ""
        response = {"answer": "", "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        timer = Timer()
        timer.elapsed = 0.0

    prompt_tokens = len(encoder.encode(prompt)) if prompt else 0
    question_tokens = len(encoder.encode(question))
    context_tokens = max(prompt_tokens - question_tokens, 0)

    # Retrieved evidence, required by round 3 (retrieved document/chunk
    # identifiers + evidence actually supplied to the LLM). Parse the
    # "Filing: <id> (<type>) [<match kind>]\nExcerpt: <text>" entries
    # built in query_graph()/build_graph_prompt().
    evidence = []
    for detail in graph_context["details"]:
        lines = detail.split("\n", 1)
        header = lines[0] if lines else ""
        excerpt = lines[1].replace("Excerpt: ", "", 1) if len(lines) > 1 else ""
        # header looks like: "Filing: AAPL_8-K_2026-04-30 (8-K) [vector match]"
        source_id = header.replace("Filing: ", "", 1).split(" (")[0].strip()
        evidence.append({"source_id": source_id, "text": excerpt})

    metrics = create_metrics(
        pipeline_name="GraphRAG",
        question=question,
        llm_response=response,
        latency=timer.elapsed,
        context_chunks=len(graph_context["details"]),
        context_tokens=context_tokens,
        system_instruction=SYSTEM_PROMPT,
        evidence=evidence,
        error=error,
    )

    return metrics


def run_benchmark(questions: list) -> list:
    """Run all questions through Pipeline 3 and collect metrics."""
    conn = get_connection(TG_GRAPH)

    results = []
    print(f"\nRunning Pipeline 3 (GraphRAG) on {len(questions)} questions...")
    print("-" * 60)

    for i, question in enumerate(questions, 1):
        print(f"\n[{i}/{len(questions)}] {question[:80]}...")
        metrics = run_query(question, conn)

        print(f"  Tokens: {metrics.total_tokens} "
              f"(in: {metrics.input_tokens}, out: {metrics.output_tokens})")
        print(f"  Context: {metrics.context_chunks} graph results, "
              f"~{metrics.context_tokens} tokens")
        print(f"  Latency: {metrics.latency_seconds:.2f}s")
        print(f"  Cost: ${metrics.cost_usd:.6f}")

        results.append(metrics)
        time.sleep(2)  # Rate limit

    total_tokens = sum(m.total_tokens for m in results)
    avg_latency = sum(m.latency_seconds for m in results) / len(results)
    total_cost = sum(m.cost_usd for m in results)
    avg_context = sum(m.context_tokens for m in results) / len(results)

    print(f"\n{'=' * 60}")
    print(f"Pipeline 3 (GraphRAG) Summary")
    print(f"{'=' * 60}")
    print(f"  Questions:      {len(results)}")
    print(f"  Total tokens:   {total_tokens:,}")
    print(f"  Avg context:    {avg_context:.0f} tokens")
    print(f"  Avg latency:    {avg_latency:.2f}s")
    print(f"  Total cost:     ${total_cost:.6f}")
    print(f"{'=' * 60}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pipeline 3: GraphRAG (sp100)")
    parser.add_argument("--setup", action="store_true", help="Create schema, add vector attribute, ingest data, install vector query")
    parser.add_argument("--query", action="store_true", help="Run benchmark queries")
    parser.add_argument("--all", action="store_true", help="Setup + query")
    args = parser.parse_args()

    if args.setup or args.all:
        print("=" * 60)
        print("Pipeline 3: GraphRAG Setup (sp100)")
        print("=" * 60)

        print("\n--- Creating Schema ---")
        create_schema()

        print("\n--- Adding Vector Attribute ---")
        add_vector_schema()

        print("\n--- Installing Vector Search Query ---")
        install_vector_search_query()

        print("\n--- Ingesting Filings (with embeddings) ---")
        ingest_filings()

    if args.query or args.all:
        print("\n" + "=" * 60)
        print("Pipeline 3: GraphRAG Benchmark (sp100)")
        print("=" * 60)

        questions_file = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "benchmark_questions3.json"
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

        results_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "results"
        )
        os.makedirs(results_dir, exist_ok=True)
        output = [m.to_dict() for m in results]
        output_path = os.path.join(results_dir, "pipeline3_graphrag.json")
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {output_path}")

    if not (args.setup or args.query or args.all):
        print("Usage:")
        print("  python -m pipelines.graph_rag --setup   # Schema + vector attribute + ingest + install query")
        print("  python -m pipelines.graph_rag --query   # Run benchmark queries")
        print("  python -m pipelines.graph_rag --all     # Both setup and query")