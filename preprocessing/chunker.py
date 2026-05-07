"""
chunker.py - Splits documents into smaller chunks for vector search.

WHY CHUNKING?
ChromaDB (and any vector DB) works best with small, focused text chunks.
If you embed an entire 2000-token paper as one vector, the embedding
captures a vague average of the whole paper. If you split it into
400-token chunks, each embedding captures a specific concept.

When a user asks a question, we find the chunks most similar to the
question and pass ONLY those chunks to the LLM as context.

HOW IT WORKS:
1. Load a document (e.g., an arXiv paper text file)
2. Split it into chunks of CHUNK_SIZE tokens
3. Each chunk overlaps with the next by CHUNK_OVERLAP tokens
   (so we don't lose context at the boundaries)

Example with CHUNK_SIZE=10, CHUNK_OVERLAP=3:
    Document: "The quick brown fox jumps over the lazy dog today"
    Chunk 1:  "The quick brown fox jumps over the lazy dog today"
    Chunk 2:  "the lazy dog today and then goes home to sleep"
    The overlap ("the lazy dog today") appears in both chunks.
"""

import os
import sys
import tiktoken

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def chunk_text(text: str, chunk_size: int = None, chunk_overlap: int = None) -> list:
    """
    Split text into overlapping chunks based on token count.

    Parameters:
    -----------
    text : str
        The full document text to split.
    chunk_size : int
        Max tokens per chunk. Defaults to config.CHUNK_SIZE (500).
    chunk_overlap : int
        Tokens of overlap between chunks. Defaults to config.CHUNK_OVERLAP (50).

    Returns:
    --------
    list of str - each string is one chunk
    """
    chunk_size = chunk_size or config.CHUNK_SIZE
    chunk_overlap = chunk_overlap or config.CHUNK_OVERLAP

    # tiktoken encodes text into tokens (same tokenizer OpenAI/Gemini-compatible)
    encoder = tiktoken.get_encoding("cl100k_base")
    tokens = encoder.encode(text)

    # If the whole document fits in one chunk, return it as-is
    if len(tokens) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(tokens):
        # Take a slice of tokens from start to start + chunk_size
        end = start + chunk_size
        chunk_tokens = tokens[start:end]

        # Decode tokens back to text
        chunk_text = encoder.decode(chunk_tokens)
        chunks.append(chunk_text)

        # Move forward by (chunk_size - overlap)
        # This creates the overlap between consecutive chunks
        start += chunk_size - chunk_overlap

    return chunks


def chunk_documents(documents: list) -> list:
    """
    Take a list of documents and return a list of chunks with metadata.

    Parameters:
    -----------
    documents : list of dict
        Each dict has 'filename', 'content', and 'metadata' keys.
        (This is what data_loader.load_all_documents() returns)

    Returns:
    --------
    list of dict, where each dict has:
        'chunk_id'  - unique ID like "paper123_chunk_0"
        'text'      - the chunk text
        'metadata'  - source filename, title, chunk index
    """
    all_chunks = []
    total_chunks = 0

    for doc in documents:
        text_chunks = chunk_text(doc["content"])

        for i, chunk in enumerate(text_chunks):
            all_chunks.append({
                "chunk_id": f"{doc['filename'].replace('.txt', '')}_chunk_{i}",
                "text": chunk,
                "metadata": {
                    "source": doc["filename"],
                    "title": doc["metadata"].get("title", ""),
                    "authors": doc["metadata"].get("authors", ""),
                    "chunk_index": i,
                    "total_chunks": len(text_chunks),
                },
            })

        total_chunks += len(text_chunks)

    print(f"Chunked {len(documents)} documents into {total_chunks} chunks")
    print(f"Avg chunks per document: {total_chunks / max(len(documents), 1):.1f}")

    return all_chunks


if __name__ == "__main__":
    """Quick test: chunk a sample text and print results."""
    sample = "This is a test. " * 200  # ~800 tokens

    chunks = chunk_text(sample, chunk_size=100, chunk_overlap=20)
    print(f"Input length: ~800 tokens")
    print(f"Chunk size: 100, Overlap: 20")
    print(f"Number of chunks: {len(chunks)}")
    for i, c in enumerate(chunks):
        encoder = tiktoken.get_encoding("cl100k_base")
        print(f"  Chunk {i}: {len(encoder.encode(c))} tokens")
