# GraphRAG Inference Hackathon - TigerGraph

## Overview
Three-pipeline benchmark comparing LLM-Only, Basic RAG (ChromaDB), and GraphRAG (TigerGraph) on arXiv research papers. Measures token reduction, answer accuracy, latency, and cost.

## Tech Stack
- **LLM Provider:** Google Gemini (free tier)
- **Vector DB:** ChromaDB (Pipeline 2)
- **Graph DB:** TigerGraph Savanna + GraphRAG repo (Pipeline 3)
- **Frontend:** Streamlit
- **Evaluation:** LLM-as-a-Judge + BERTScore

## Project Structure
```
graphrag-hackathon/
├── config.py              # API keys, model settings
├── data/
│   └── papers/            # Raw arXiv papers (text files)
├── pipelines/
│   ├── __init__.py
│   ├── llm_only.py        # Pipeline 1: Direct LLM
│   ├── basic_rag.py       # Pipeline 2: ChromaDB + LLM
│   └── graph_rag.py       # Pipeline 3: TigerGraph GraphRAG + LLM
├── preprocessing/
│   ├── __init__.py
│   └── data_loader.py     # Download and preprocess arXiv papers
├── evaluation/
│   ├── __init__.py
│   └── accuracy.py        # LLM-as-a-Judge + BERTScore
├── utils/
│   ├── __init__.py
│   ├── gemini_client.py   # Shared Gemini API wrapper
│   └── metrics.py         # Token counting, latency, cost tracking
├── app.py                 # Streamlit dashboard
├── requirements.txt
└── README.md
```

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set your Gemini API key
Create a `.env` file:
```
GEMINI_API_KEY=your_api_key_here
```

### 3. Download dataset
```bash
python preprocessing/data_loader.py
```

### 4. Run Pipeline 1 (LLM-Only)
```bash
python -m pipelines.llm_only
```

### 5. Launch Dashboard
```bash
streamlit run app.py
```
