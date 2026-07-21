"""
Data loader for the GraphRAG hackathon.

Round 1: arXiv papers (download_papers, load_all_documents)
Round 3: sp100 SEC filings, already provided under data/sp100_dataset/<TICKER>/*.txt
         (load_sp100_documents)

Usage (run from project root):
    python preprocessing/data_loader.py
"""

import os
import sys
import json
import arxiv
import tiktoken

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import config


SEARCH_QUERIES = [
    "large language models retrieval augmented generation",
    "knowledge graphs natural language processing",
    "graph neural networks applications",
    "transformer architecture attention mechanism",
    "machine learning healthcare clinical NLP",
    "reinforcement learning robotics multi-agent",
    "computer vision object detection autonomous driving",
    "federated learning privacy preserving machine learning",
    "recommendation systems collaborative filtering",
    "natural language understanding question answering",
    "prompt engineering in-context learning LLM",
    "diffusion models image generation text-to-image",
    "multimodal learning vision language models",
    "neural architecture search AutoML",
    "continual learning catastrophic forgetting",
    "causal inference machine learning",
    "AI safety alignment large language models",
    "code generation large language models programming",
    "speech recognition natural language processing",
    "time series forecasting deep learning",
]

PAPERS_PER_QUERY = 50  # ~1000 papers total to hit 2M+ tokens


# ---------------------------------------------------------------------------
# Round 1: arXiv papers
# ---------------------------------------------------------------------------

def download_papers(output_dir: str = None, max_papers_per_query: int = PAPERS_PER_QUERY):
    """
    Download arXiv paper abstracts and metadata.
    Saves each paper as a text file with title, authors, abstract, and categories.
    """
    output_dir = output_dir or os.path.join(PROJECT_ROOT, config.DATA_DIR)
    os.makedirs(output_dir, exist_ok=True)

    paper_count = 0
    seen_ids = set()

    for query in SEARCH_QUERIES:
        print(f"\nSearching: '{query}'")
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_papers_per_query,
            sort_by=arxiv.SortCriterion.Relevance,
        )

        for result in client.results(search):
            paper_id = result.entry_id.split("/")[-1]

            if paper_id in seen_ids:
                continue
            seen_ids.add(paper_id)

            authors = ", ".join([a.name for a in result.authors])
            categories = ", ".join(result.categories)
            published = result.published.strftime("%Y-%m-%d") if result.published else "Unknown"

            content = f"""Title: {result.title}

Authors: {authors}

Published: {published}

Categories: {categories}

Abstract:
{result.summary}

Paper ID: {paper_id}

Summary:
This paper titled "{result.title}" was authored by {authors} and published on {published}.
It falls under the categories: {categories}.
The research explores the following topic: {result.summary}
"""
            if result.links:
                links = "\n".join([f"  - {link.href}" for link in result.links])
                content += f"\nLinks:\n{links}\n"

            safe_id = paper_id.replace("/", "_").replace(".", "_")
            filepath = os.path.join(output_dir, f"{safe_id}.txt")
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)

            paper_count += 1

            if paper_count % 100 == 0:
                print(f"  Downloaded {paper_count} papers so far...")

    print(f"\nTotal papers downloaded: {paper_count}")
    return paper_count


def count_tokens(directory: str = None) -> dict:
    """Count total tokens across all downloaded papers."""
    directory = directory or os.path.join(PROJECT_ROOT, config.DATA_DIR)
    encoder = tiktoken.get_encoding("cl100k_base")

    total_tokens = 0
    file_count = 0

    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
            tokens = len(encoder.encode(text))
            total_tokens += tokens
            file_count += 1

    stats = {
        "total_files": file_count,
        "total_tokens": total_tokens,
        "avg_tokens_per_file": total_tokens // max(file_count, 1),
    }

    print(f"\nDataset Stats:")
    print(f"  Files: {stats['total_files']}")
    print(f"  Total tokens: {stats['total_tokens']:,}")
    print(f"  Avg tokens/file: {stats['avg_tokens_per_file']:,}")
    print(f"  Target: 2,000,000 tokens")
    print(f"  {'REACHED' if total_tokens >= 2_000_000 else 'NEED MORE DATA'}")

    return stats


def load_all_documents(directory: str = None) -> list:
    """
    Load all arXiv text files and return as a list of documents.
    """
    directory = directory or os.path.join(PROJECT_ROOT, config.DATA_DIR)
    documents = []

    for filename in sorted(os.listdir(directory)):
        if not filename.endswith(".txt"):
            continue

        filepath = os.path.join(directory, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        lines = content.strip().split("\n")
        title = ""
        authors = ""
        for line in lines:
            if line.startswith("Title: "):
                title = line[7:]
            elif line.startswith("Authors: "):
                authors = line[9:]

        documents.append({
            "filename": filename,
            "content": content,
            "metadata": {
                "title": title,
                "authors": authors,
                "source": filename,
            },
        })

    print(f"Loaded {len(documents)} documents")
    return documents


# ---------------------------------------------------------------------------
# Round 3: sp100 SEC filings
# ---------------------------------------------------------------------------

def load_sp100_documents(directory: str = None) -> list:
    """
    Walk data/sp100_dataset/<TICKER>/*.txt and load all SEC filings.
    Returns the same document shape as load_all_documents() so downstream
    pipelines (llm_only, basic_rag, graph_rag) don't need to change how
    they consume documents.

    Expected filename pattern: "<FILING_TYPE>_<DATE>.txt"
      e.g. "8-K_2026-04-30.txt", "10-K_2025-10-31.txt", "DEF14A_2026-01-08.txt"
    """
    directory = directory or os.path.join(PROJECT_ROOT, "data", "sp100_dataset")
    documents = []

    if not os.path.isdir(directory):
        raise FileNotFoundError(
            f"sp100 dataset directory not found: {directory}\n"
            "Make sure the sp100_dataset folder is placed under data/."
        )

    for ticker in sorted(os.listdir(directory)):
        ticker_path = os.path.join(directory, ticker)
        if not os.path.isdir(ticker_path):
            continue

        for filename in sorted(os.listdir(ticker_path)):
            if not filename.endswith(".txt"):
                continue

            filepath = os.path.join(ticker_path, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            name_no_ext = filename.replace(".txt", "")
            parts = name_no_ext.split("_", 1)
            filing_type = parts[0] if len(parts) > 0 else "UNKNOWN"
            filing_date = parts[1] if len(parts) > 1 else "UNKNOWN"

            documents.append({
                "filename": filename,
                "content": content,
                "metadata": {
                    "title": f"{ticker} {filing_type} ({filing_date})",
                    "authors": ticker,  # kept for shape compatibility with round 1
                    "source": filename,
                    "ticker": ticker,
                    "filing_type": filing_type,
                    "filing_date": filing_date,
                },
            })

    num_tickers = len(set(d["metadata"]["ticker"] for d in documents)) if documents else 0
    print(f"Loaded {len(documents)} sp100 filings across {num_tickers} tickers")
    return documents


def count_sp100_tokens(directory: str = None) -> dict:
    """Count total tokens across all sp100 filings (mirrors count_tokens())."""
    directory = directory or os.path.join(PROJECT_ROOT, "data", "sp100_dataset")
    encoder = tiktoken.get_encoding("cl100k_base")

    total_tokens = 0
    file_count = 0

    for ticker in sorted(os.listdir(directory)):
        ticker_path = os.path.join(directory, ticker)
        if not os.path.isdir(ticker_path):
            continue
        for filename in os.listdir(ticker_path):
            if not filename.endswith(".txt"):
                continue
            filepath = os.path.join(ticker_path, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
            total_tokens += len(encoder.encode(text))
            file_count += 1

    stats = {
        "total_files": file_count,
        "total_tokens": total_tokens,
        "avg_tokens_per_file": total_tokens // max(file_count, 1),
    }

    print(f"\nsp100 Dataset Stats:")
    print(f"  Files: {stats['total_files']}")
    print(f"  Total tokens: {stats['total_tokens']:,}")
    print(f"  Avg tokens/file: {stats['avg_tokens_per_file']:,}")

    return stats


if __name__ == "__main__":
    print("=" * 60)
    print("sp100 SEC Filing Loader for GraphRAG Hackathon Round 3")
    print("=" * 60)

    print("\nLoading sp100 documents...")
    docs = load_sp100_documents()

    print("\nCounting tokens...")
    count_sp100_tokens()