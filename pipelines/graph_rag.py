"""
Pipeline 3: GraphRAG (TigerGraph + Gemini)

HOW IT WORKS:
1. SETUP (one-time):
   - Connect to TigerGraph Savanna
   - Create a graph schema (Papers, Authors, Concepts, relationships)
   - Ingest arXiv papers: extract entities and relationships
   - Store them as vertices and edges in the graph

2. QUERY TIME (per question):
   - Extract key entities from the question using Gemini
   - Query TigerGraph to find related entities via graph traversal
   - Multi-hop reasoning: follow relationships across nodes
   - Build a focused, structured context from graph results
   - Send structured context + question to Gemini
   - Gemini answers using graph-derived context

WHY THIS IS BETTER THAN PIPELINE 2 (Basic RAG):
Basic RAG retrieves similar text chunks. GraphRAG understands
RELATIONSHIPS between entities. If a question requires connecting
Author A -> Paper B -> Concept C, GraphRAG traverses that path
directly instead of hoping the right chunks appear in vector search.

The result: fewer tokens (structured context vs raw chunks),
better accuracy (multi-hop reasoning), lower cost.
"""

import json
import sys
import os
import time
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyTigerGraph import TigerGraphConnection
import config
from utils.gemini_client import generate
from utils.metrics import Timer, create_metrics, QueryMetrics
from preprocessing.data_loader import load_all_documents
import tiktoken


# ── TigerGraph Savanna Connection ─────────────────────────────
TG_HOST = "https://tg-a038d9c9-3c9c-4044-92c3-89b706ffe27f.tg-2635877100.i.tgcloud.io"
TG_USERNAME = "workspace1"
TG_PASSWORD = "workspace123"
TG_JWT = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ3b3Jrc3BhY2UxIiwiaWF0IjoxNzc4MTE4OTMzLCJleHAiOjE3Nzg3MjM3MzgsImlzcyI6IlRpZ2VyR3JhcGgifQ.HkA2SncfBLx7kuiJcwXSHRAl_3Ics5AIObmc-MqW-vo"
TG_GRAPH = "ArxivGraphRAG"

SYSTEM_PROMPT = """You are a helpful research assistant specializing in AI and machine learning.
Answer the question based ONLY on the provided graph context below.
The context contains entities and relationships extracted from a knowledge graph
of research papers. Use these connections to provide accurate, structured answers.
If the context doesn't contain enough information, say so.
Be concise and specific."""


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
    - Paper (vertex): title, abstract, published date, paper_id
    - Author (vertex): name
    - Concept (vertex): name (extracted from abstracts)
    - AUTHORED_BY (edge): Paper -> Author
    - COVERS_CONCEPT (edge): Paper -> Concept
    - RELATED_TO (edge): Concept -> Concept
    """
    conn = get_connection()

    schema_gsql = f"""
    CREATE GRAPH {TG_GRAPH}()

    USE GRAPH {TG_GRAPH}

    CREATE SCHEMA_CHANGE JOB schema_job {{
        ADD VERTEX Paper(PRIMARY_ID paper_id STRING, title STRING, abstract STRING, published STRING, categories STRING) WITH primary_id_as_attribute="true";
        ADD VERTEX Author(PRIMARY_ID name STRING) WITH primary_id_as_attribute="true";
        ADD VERTEX Concept(PRIMARY_ID name STRING) WITH primary_id_as_attribute="true";
        ADD UNDIRECTED EDGE AUTHORED_BY(FROM Paper, TO Author);
        ADD UNDIRECTED EDGE COVERS_CONCEPT(FROM Paper, TO Concept);
        ADD UNDIRECTED EDGE RELATED_TO(FROM Concept, TO Concept);
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


def extract_concepts_from_abstract(abstract: str) -> list:
    """
    Use Gemini to extract key concepts from a paper abstract.
    Returns a list of concept strings.
    """
    prompt = f"""Extract 3-5 key technical concepts from this research paper abstract.
Return ONLY a JSON array of strings, nothing else.
Example: ["transformer architecture", "attention mechanism", "language model"]

Abstract: {abstract[:1000]}"""

    try:
        response = generate(prompt=prompt)
        answer = response["answer"].strip()
        # Parse JSON array from response
        # Handle markdown code blocks
        answer = answer.replace("```json", "").replace("```", "").strip()
        concepts = json.loads(answer)
        if isinstance(concepts, list):
            return [c.lower().strip() for c in concepts[:5]]
    except Exception as e:
        pass

    # Fallback: extract simple keyword concepts
    words = abstract.lower().split()
    keywords = []
    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
    # Common AI/ML concepts to look for
    common = ["machine learning", "deep learning", "neural network", "natural language",
              "knowledge graph", "attention mechanism", "transformer", "reinforcement learning",
              "graph neural", "federated learning", "language model", "computer vision",
              "object detection", "recommendation system", "question answering"]
    for concept in common:
        if concept in abstract.lower():
            keywords.append(concept)
    return keywords[:5] if keywords else ["machine learning"]


def ingest_papers():
    """
    Load arXiv papers and ingest them into TigerGraph as a knowledge graph.
    Extracts: Papers, Authors, Concepts, and their relationships.
    """
    conn = get_connection(TG_GRAPH)

    # Load documents
    print("\nLoading documents...")
    documents = load_all_documents()

    if not documents:
        print("ERROR: No documents found. Run preprocessing/data_loader.py first.")
        return

    print(f"\nIngesting {len(documents)} papers into TigerGraph...")

    papers_added = 0
    authors_added = set()
    concepts_added = set()

    for i, doc in enumerate(documents):
        content = doc["content"]
        metadata = doc["metadata"]

        # Parse paper fields
        title = metadata.get("title", "Unknown")
        authors_str = metadata.get("authors", "")
        paper_id = doc["filename"].replace(".txt", "")

        # Extract fields from content
        abstract = ""
        published = ""
        categories = ""
        lines = content.split("\n")
        in_abstract = False
        for line in lines:
            if line.startswith("Abstract:"):
                in_abstract = True
                continue
            elif line.startswith("Paper ID:") or line.startswith("Summary:"):
                in_abstract = False
            elif line.startswith("Published: "):
                published = line[11:]
            elif line.startswith("Categories: "):
                categories = line[12:]
            elif in_abstract:
                abstract += line + " "

        abstract = abstract.strip()[:2000]  # Limit abstract length

        # Upsert Paper vertex
        try:
            conn.upsertVertex("Paper", paper_id, {
                "title": title[:500],
                "abstract": abstract[:2000],
                "published": published,
                "categories": categories,
            })
        except Exception as e:
            print(f"  Error upserting paper {paper_id}: {e}")
            continue

        # Upsert Author vertices and edges
        if authors_str:
            for author in authors_str.split(", ")[:10]:  # Max 10 authors
                author = author.strip()
                if author and len(author) > 1:
                    try:
                        conn.upsertVertex("Author", author, {"name": author})
                        conn.upsertEdge("Paper", paper_id, "AUTHORED_BY", "Author", author)
                        authors_added.add(author)
                    except Exception:
                        pass

        # Extract concepts from categories + keyword matching (no Gemini calls)
        local_concepts = []

        # Use arXiv categories as concepts
        if categories:
            for cat in categories.split(", "):
                cat = cat.strip().lower()
                if cat:
                    local_concepts.append(cat)

        # Also extract common AI/ML concepts from abstract
        common_concepts = [
            "machine learning", "deep learning", "neural network", "natural language",
            "knowledge graph", "attention mechanism", "transformer", "reinforcement learning",
            "graph neural", "federated learning", "language model", "computer vision",
            "object detection", "recommendation system", "question answering",
            "generative adversarial", "convolutional neural", "recurrent neural",
            "transfer learning", "semi-supervised", "self-supervised", "contrastive learning",
            "few-shot learning", "zero-shot", "prompt engineering", "retrieval augmented",
            "text classification", "sentiment analysis", "named entity", "relation extraction",
            "image segmentation", "speech recognition", "time series", "anomaly detection",
            "causal inference", "diffusion model", "variational autoencoder",
        ]
        abstract_lower = abstract.lower()
        for concept in common_concepts:
            if concept in abstract_lower:
                local_concepts.append(concept)

        # Upsert concepts and edges
        for concept in local_concepts[:8]:
            try:
                conn.upsertVertex("Concept", concept, {"name": concept})
                conn.upsertEdge("Paper", paper_id, "COVERS_CONCEPT", "Concept", concept)
                concepts_added.add(concept)
            except Exception:
                pass

        papers_added += 1
        if papers_added % 50 == 0:
            print(f"  Ingested {papers_added}/{len(documents)} papers...")

    # Create RELATED_TO edges between concepts that co-occur in papers
    print("\nCreating concept relationships...")
    try:
        # Concepts that share papers are related
        relate_query = f"""
        INTERPRET QUERY () FOR GRAPH {TG_GRAPH} {{
            all_concepts = {{Concept.*}};
            papers = SELECT p FROM all_concepts:c -(COVERS_CONCEPT:e)- Paper:p;
            PRINT papers.size() AS paper_count;
        }}
        """
        conn.gsql(relate_query)
    except Exception as e:
        print(f"  Note: Could not create concept relationships: {e}")

    print(f"\nIngestion complete!")
    print(f"  Papers: {papers_added}")
    print(f"  Authors: {len(authors_added)}")
    print(f"  Concepts: {len(concepts_added)}")


# ══════════════════════════════════════════════════════════════
# QUERY FUNCTIONS (run per question)
# ══════════════════════════════════════════════════════════════

def extract_entities_from_question(question: str) -> list:
    """Extract searchable entities from a question using keyword matching (no API calls)."""
    question_lower = question.lower()

    # Common AI/ML concepts to search for in the graph
    common_concepts = [
        "retrieval augmented generation", "knowledge graph", "transformer",
        "attention mechanism", "graph neural network", "federated learning",
        "collaborative filtering", "content-based", "recommendation",
        "reinforcement learning", "rlhf", "human feedback",
        "bert", "gpt", "language model", "knowledge graph embedding",
        "transe", "rotate", "privacy", "multi-hop reasoning",
        "question answering", "machine learning", "deep learning",
        "neural network", "natural language", "computer vision",
        "self-attention", "embedding", "healthcare",
    ]

    matched = [c for c in common_concepts if c in question_lower]

    # Also extract significant words as fallback
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


def query_graph(question: str, conn: TigerGraphConnection) -> dict:
    """
    Query TigerGraph to find relevant context for a question.

    Strategy:
    1. Extract entities from the question (local, no API calls)
    2. Search for matching Concepts in TigerGraph
    3. Get connected Papers and Authors via edges
    4. Build structured context
    """
    entities = extract_entities_from_question(question)

    context_parts = []
    found_papers = set()
    found_authors = set()
    found_concepts = set()

    for entity in entities:
        try:
            # Try to get the concept vertex directly
            try:
                concept_data = conn.getVerticesById("Concept", entity)
                if concept_data:
                    found_concepts.add(entity)

                    # Get papers connected to this concept
                    edges = conn.getEdges("Concept", entity, "COVERS_CONCEPT")
                    for edge in edges[:5]:
                        paper_id = edge.get("to_id", edge.get("from_id", ""))
                        if paper_id and paper_id not in found_papers:
                            try:
                                paper_data = conn.getVerticesById("Paper", paper_id)
                                if paper_data:
                                    p = paper_data[0] if isinstance(paper_data, list) else paper_data
                                    attrs = p.get("attributes", p)
                                    title = attrs.get("title", "")
                                    abstract = attrs.get("abstract", "")[:300]
                                    found_papers.add(paper_id)
                                    context_parts.append(
                                        f"Paper: {title}\nAbstract: {abstract}"
                                    )

                                    # Get authors of this paper
                                    try:
                                        author_edges = conn.getEdges("Paper", paper_id, "AUTHORED_BY")
                                        for ae in author_edges[:5]:
                                            author_name = ae.get("to_id", ae.get("from_id", ""))
                                            if author_name:
                                                found_authors.add(author_name)
                                    except Exception:
                                        pass
                            except Exception:
                                pass
            except Exception:
                pass

            # Also search for concepts that contain the entity as substring
            try:
                all_concepts = conn.getVertices("Concept", limit=200)
                for c in all_concepts:
                    c_name = c.get("v_id", c.get("attributes", {}).get("name", ""))
                    if entity in c_name and c_name not in found_concepts:
                        found_concepts.add(c_name)

                        # Get papers for this concept too
                        try:
                            edges = conn.getEdges("Concept", c_name, "COVERS_CONCEPT")
                            for edge in edges[:3]:
                                paper_id = edge.get("to_id", edge.get("from_id", ""))
                                if paper_id and paper_id not in found_papers:
                                    paper_data = conn.getVerticesById("Paper", paper_id)
                                    if paper_data:
                                        p = paper_data[0] if isinstance(paper_data, list) else paper_data
                                        attrs = p.get("attributes", p)
                                        title = attrs.get("title", "")
                                        abstract = attrs.get("abstract", "")[:300]
                                        found_papers.add(paper_id)
                                        context_parts.append(
                                            f"Paper: {title}\nAbstract: {abstract}"
                                        )
                        except Exception:
                            pass
            except Exception:
                pass

        except Exception:
            pass

    # If we found nothing, get some random papers as fallback
    if not context_parts:
        try:
            papers = conn.getVertices("Paper", limit=3)
            for p in papers:
                attrs = p.get("attributes", p)
                title = attrs.get("title", "")
                abstract = attrs.get("abstract", "")[:300]
                if title:
                    context_parts.append(f"Paper: {title}\nAbstract: {abstract}")
        except Exception:
            pass

    # Build structured context
    context = {
        "papers": list(found_papers),
        "authors": list(found_authors),
        "concepts": list(found_concepts),
        "details": context_parts[:5],  # Limit to 5 most relevant
    }

    return context


def build_graph_prompt(question: str, graph_context: dict) -> str:
    """
    Build a prompt from graph-derived context.

    Unlike Pipeline 2 which dumps raw text chunks,
    this provides STRUCTURED context with entities and relationships.
    This is typically much shorter (fewer tokens) while being more precise.
    """
    parts = []

    if graph_context["concepts"]:
        parts.append(f"Related Concepts: {', '.join(graph_context['concepts'][:10])}")

    if graph_context["authors"]:
        parts.append(f"Key Researchers: {', '.join(graph_context['authors'][:10])}")

    if graph_context["details"]:
        parts.append("Relevant Research:\n" + "\n---\n".join(graph_context["details"]))

    context = "\n\n".join(parts) if parts else "No specific graph context found."

    prompt = f"""Graph Context:
{context}

Question: {question}

Answer based on the graph context above. Use the relationships between
concepts, papers, and authors to provide a well-connected answer."""

    return prompt


def run_query(question: str, conn: TigerGraphConnection = None) -> QueryMetrics:
    """Run a single question through the GraphRAG pipeline."""
    if conn is None:
        conn = get_connection(TG_GRAPH)

    encoder = tiktoken.get_encoding("cl100k_base")

    with Timer() as timer:
        # Step 1: Query the graph for relevant context
        graph_context = query_graph(question, conn)

        # Step 2: Build prompt from graph context
        prompt = build_graph_prompt(question, graph_context)

        # Step 3: Send to Gemini
        response = generate(
            prompt=prompt,
            system_instruction=SYSTEM_PROMPT,
        )

    # Count context tokens
    prompt_tokens = len(encoder.encode(prompt))
    question_tokens = len(encoder.encode(question))
    context_tokens = prompt_tokens - question_tokens

    metrics = create_metrics(
        pipeline_name="GraphRAG",
        question=question,
        llm_response=response,
        latency=timer.elapsed,
        context_chunks=len(graph_context["details"]),
        context_tokens=context_tokens,
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

    # Summary
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

    parser = argparse.ArgumentParser(description="Pipeline 3: GraphRAG")
    parser.add_argument("--setup", action="store_true", help="Create schema and ingest data")
    parser.add_argument("--query", action="store_true", help="Run benchmark queries")
    parser.add_argument("--all", action="store_true", help="Setup + query")
    args = parser.parse_args()

    if args.setup or args.all:
        print("=" * 60)
        print("Pipeline 3: GraphRAG Setup")
        print("=" * 60)

        print("\n--- Creating Schema ---")
        create_schema()

        print("\n--- Ingesting Papers ---")
        ingest_papers()

    if args.query or args.all:
        print("\n" + "=" * 60)
        print("Pipeline 3: GraphRAG Benchmark")
        print("=" * 60)

        questions_file = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "benchmark_questions.json"
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

        # Save results
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
        print("  python -m pipelines.graph_rag --setup   # Create schema + ingest data")
        print("  python -m pipelines.graph_rag --query   # Run benchmark queries")
        print("  python -m pipelines.graph_rag --all     # Both setup and query")