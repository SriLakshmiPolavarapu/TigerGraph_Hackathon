"""
Pipeline 2: Basic RAG (Retrieval Augmented Generation)

HOW IT WORKS:
1. SETUP (one-time):
   - Load all arXiv papers
   - Split them into small chunks (500 tokens each)
   - Convert each chunk into a vector (embedding) using sentence-transformers
   - Store all vectors in ChromaDB

2. QUERY TIME (per question):
   - Convert the question into a vector
   - Find the 5 most similar chunks in ChromaDB (cosine similarity)
   - Stuff those chunks into the prompt as "context"
   - Send context + question to Gemini
   - Gemini answers using the provided context

WHY THIS IS BETTER THAN PIPELINE 1:
Pipeline 1 relies entirely on Gemini's training data. If you ask about a
specific paper or recent research, it might hallucinate. Pipeline 2 gives
Gemini actual source text to reference, improving accuracy.

WHY THIS IS WORSE THAN PIPELINE 3 (GraphRAG):
Vector search finds "similar text chunks" but can't reason across
relationships. If the answer requires connecting Author A -> Paper B ->
Concept C -> Paper D, vector search won't find that chain. It just
returns the 5 most textually similar chunks, which might miss the
connection entirely. That's what GraphRAG solves.
"""

import json
import sys
import os
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chromadb
from sentence_transformers import SentenceTransformer
import tiktoken

import config
from utils.gemini_client import generate
from utils.metrics import Timer, create_metrics, QueryMetrics
from preprocessing.data_loader import load_all_documents
from preprocessing.chunker import chunk_documents


# ── System prompt for RAG ─────────────────────────────────────
# Notice this is different from Pipeline 1's prompt.
# We tell the model to ONLY use the provided context.
SYSTEM_PROMPT = """You are a helpful research assistant specializing in AI and machine learning.
Answer the question based ONLY on the provided context below.
If the context doesn't contain enough information to answer, say so.
Be concise and specific. Cite relevant details from the context."""


# ══════════════════════════════════════════════════════════════
# SETUP FUNCTIONS (run once to build the vector database)
# ══════════════════════════════════════════════════════════════

def get_chroma_client():
    """
    Create a ChromaDB client with persistent storage.

    PersistentClient means the data is saved to disk. If you restart
    your script, you don't need to re-embed everything.
    """
    persist_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        config.CHROMA_PERSIST_DIR,
    )
    os.makedirs(persist_dir, exist_ok=True)
    return chromadb.PersistentClient(path=persist_dir)


def get_embedding_model():
    """
    Load the sentence-transformers model for converting text to vectors.

    'all-MiniLM-L6-v2' is a lightweight model that:
    - Runs locally (no API calls, no cost)
    - Produces 384-dimensional vectors
    - Is fast enough for hackathon-scale data
    """
    return SentenceTransformer(config.EMBEDDING_MODEL)


def build_vector_store(force_rebuild: bool = False):
    """
    Load documents, chunk them, embed them, and store in ChromaDB.

    This is the one-time setup step. After running this, ChromaDB
    has all your chunks stored as vectors on disk.

    Parameters:
    -----------
    force_rebuild : bool
        If True, delete existing collection and rebuild from scratch.
        If False, skip if collection already exists.

    Returns:
    --------
    chromadb.Collection - the vector store collection
    """
    client = get_chroma_client()

    # Check if collection already exists
    existing = [c.name for c in client.list_collections()]
    if config.CHROMA_COLLECTION_NAME in existing and not force_rebuild:
        collection = client.get_collection(config.CHROMA_COLLECTION_NAME)
        count = collection.count()
        print(f"Collection '{config.CHROMA_COLLECTION_NAME}' already exists "
              f"with {count} chunks. Skipping rebuild.")
        print(f"(Use force_rebuild=True to rebuild)")
        return collection

    # Delete existing collection if rebuilding
    if config.CHROMA_COLLECTION_NAME in existing:
        client.delete_collection(config.CHROMA_COLLECTION_NAME)
        print(f"Deleted existing collection '{config.CHROMA_COLLECTION_NAME}'")

    # Create new collection
    collection = client.create_collection(
        name=config.CHROMA_COLLECTION_NAME,
        metadata={"description": "arXiv papers for GraphRAG hackathon"},
    )

    # Step 1: Load all documents
    print("\nStep 1: Loading documents...")
    documents = load_all_documents()

    if not documents:
        print("ERROR: No documents found. Run preprocessing/data_loader.py first.")
        return collection

    # Step 2: Chunk documents
    print("\nStep 2: Chunking documents...")
    chunks = chunk_documents(documents)

    # Step 3: Generate embeddings and store in ChromaDB
    print(f"\nStep 3: Embedding {len(chunks)} chunks...")
    model = get_embedding_model()

    # Process in batches (ChromaDB works better with batches)
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]

        # Extract text and metadata for this batch
        ids = [c["chunk_id"] for c in batch]
        texts = [c["text"] for c in batch]
        metadatas = [c["metadata"] for c in batch]

        # Generate embeddings for all texts in the batch
        # model.encode() converts text strings into numerical vectors
        embeddings = model.encode(texts).tolist()

        # Add to ChromaDB
        collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        if (i + batch_size) % 500 == 0 or (i + batch_size) >= len(chunks):
            print(f"  Embedded {min(i + batch_size, len(chunks))}/{len(chunks)} chunks")

    print(f"\nVector store built: {collection.count()} chunks stored")
    return collection


# ══════════════════════════════════════════════════════════════
# QUERY FUNCTIONS (run per question)
# ══════════════════════════════════════════════════════════════

def retrieve_chunks(question: str, collection, model, top_k: int = None) -> list:
    """
    Find the most relevant chunks for a question using vector similarity.

    HOW VECTOR SEARCH WORKS:
    1. Convert the question into a vector (same model used for chunks)
    2. ChromaDB compares this vector against all stored chunk vectors
    3. Returns the top_k most similar chunks (cosine similarity)

    Parameters:
    -----------
    question : str
        The user's question
    collection : chromadb.Collection
        The ChromaDB collection with all chunks
    model : SentenceTransformer
        The embedding model (same one used to embed chunks)
    top_k : int
        Number of chunks to retrieve. Default from config (5).

    Returns:
    --------
    list of dict, each with 'text', 'metadata', 'distance'
    """
    top_k = top_k or config.TOP_K_RESULTS

    # Convert question to a vector
    question_embedding = model.encode([question]).tolist()

    # Query ChromaDB for similar chunks
    results = collection.query(
        query_embeddings=question_embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    # Package results into a clean list
    retrieved = []
    for i in range(len(results["ids"][0])):
        retrieved.append({
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
        })

    return retrieved


def build_rag_prompt(question: str, retrieved_chunks: list) -> str:
    """
    Build the prompt that gets sent to Gemini.

    This combines the retrieved context with the question.
    The prompt structure matters a lot for answer quality.

    The format is:
        Context:
        [chunk 1 text]
        ---
        [chunk 2 text]
        ---
        ...

        Question: [the actual question]
    """
    # Build context string from retrieved chunks
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        source = chunk["metadata"].get("title", chunk["metadata"].get("source", "Unknown"))
        context_parts.append(f"[Source {i}: {source}]\n{chunk['text']}")

    context = "\n---\n".join(context_parts)

    # Combine context and question into final prompt
    prompt = f"""Context:
{context}

Question: {question}

Answer based on the context above. Be concise and specific."""

    return prompt


def count_context_tokens(prompt: str, question: str) -> int:
    """
    Count how many tokens are in the context portion of the prompt.
    This helps measure how much data we're sending to the LLM.
    """
    encoder = tiktoken.get_encoding("cl100k_base")
    # The context is everything in the prompt except the question
    prompt_tokens = len(encoder.encode(prompt))
    question_tokens = len(encoder.encode(question))
    return prompt_tokens - question_tokens


def run_query(question: str, collection=None, model=None) -> QueryMetrics:
    """
    Run a single question through the Basic RAG pipeline.

    Steps:
    1. Retrieve relevant chunks from ChromaDB
    2. Build a prompt with context + question
    3. Send to Gemini
    4. Track all metrics

    Parameters:
    -----------
    question : str
        The question to answer
    collection : chromadb.Collection, optional
        Pre-loaded ChromaDB collection (avoids reloading each query)
    model : SentenceTransformer, optional
        Pre-loaded embedding model (avoids reloading each query)
    """
    # Load collection and model if not provided
    if collection is None:
        client = get_chroma_client()
        collection = client.get_collection(config.CHROMA_COLLECTION_NAME)
    if model is None:
        model = get_embedding_model()

    with Timer() as timer:
        # Step 1: Retrieve relevant chunks
        retrieved = retrieve_chunks(question, collection, model)

        # Step 2: Build prompt with context
        prompt = build_rag_prompt(question, retrieved)

        # Step 3: Send to Gemini
        response = generate(
            prompt=prompt,
            system_instruction=SYSTEM_PROMPT,
        )

    # Count context tokens
    ctx_tokens = count_context_tokens(prompt, question)

    metrics = create_metrics(
        pipeline_name="Basic RAG",
        question=question,
        llm_response=response,
        latency=timer.elapsed,
        context_chunks=len(retrieved),
        context_tokens=ctx_tokens,
    )

    return metrics


def run_benchmark(questions: list) -> list:
    """
    Run all questions through Pipeline 2 and collect metrics.
    """
    # Pre-load collection and model once (faster than loading per query)
    print("Loading ChromaDB collection and embedding model...")
    client = get_chroma_client()

    # Check if collection exists
    existing = [c.name for c in client.list_collections()]
    if config.CHROMA_COLLECTION_NAME not in existing:
        print("ERROR: Vector store not built yet. Building now...")
        collection = build_vector_store()
    else:
        collection = client.get_collection(config.CHROMA_COLLECTION_NAME)
        print(f"Loaded collection with {collection.count()} chunks")

    model = get_embedding_model()

    results = []
    print(f"\nRunning Pipeline 2 (Basic RAG) on {len(questions)} questions...")
    print("-" * 60)

    for i, question in enumerate(questions, 1):
        print(f"\n[{i}/{len(questions)}] {question[:80]}...")
        metrics = run_query(question, collection, model)

        print(f"  Tokens: {metrics.total_tokens} "
              f"(in: {metrics.input_tokens}, out: {metrics.output_tokens})")
        print(f"  Context: {metrics.context_chunks} chunks, "
              f"~{metrics.context_tokens} tokens")
        print(f"  Latency: {metrics.latency_seconds:.2f}s")
        print(f"  Cost: ${metrics.cost_usd:.6f}")

        results.append(metrics)

        # Rate limit delay
        time.sleep(2)

    # Print summary
    total_tokens = sum(m.total_tokens for m in results)
    avg_latency = sum(m.latency_seconds for m in results) / len(results)
    total_cost = sum(m.cost_usd for m in results)
    avg_context = sum(m.context_tokens for m in results) / len(results)

    print(f"\n{'=' * 60}")
    print(f"Pipeline 2 (Basic RAG) Summary")
    print(f"{'=' * 60}")
    print(f"  Questions:      {len(results)}")
    print(f"  Total tokens:   {total_tokens:,}")
    print(f"  Avg context:    {avg_context:.0f} tokens")
    print(f"  Avg latency:    {avg_latency:.2f}s")
    print(f"  Total cost:     ${total_cost:.6f}")
    print(f"{'=' * 60}")

    return results


if __name__ == "__main__":
    print("=" * 60)
    print("Pipeline 2: Basic RAG (ChromaDB + Gemini)")
    print("=" * 60)

    # Step 1: Build vector store if needed
    print("\n--- Building Vector Store ---")
    build_vector_store()

    # Step 2: Load benchmark questions
    questions_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "benchmark_questions.json"
    )

    if os.path.exists(questions_file):
        with open(questions_file, "r") as f:
            data = json.load(f)
            questions = [q["question"] for q in data]
        print(f"\nLoaded {len(questions)} questions from {questions_file}")
    else:
        print(f"\nERROR: {questions_file} not found.")
        print("Create benchmark_questions.json in the data/ folder.")
        sys.exit(1)

    # Step 3: Run benchmark
    results = run_benchmark(questions)

    # Step 4: Save results
    results_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results"
    )
    os.makedirs(results_dir, exist_ok=True)
    output = [m.to_dict() for m in results]
    output_path = os.path.join(results_dir, "pipeline2_basic_rag.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")
