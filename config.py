import os
from dotenv import load_dotenv

load_dotenv()


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.5-flash"  


COST_PER_1M_INPUT_TOKENS = 0.10
COST_PER_1M_OUTPUT_TOKENS = 0.40


DATA_DIR = "data/papers"
CHUNK_SIZE = 500        
CHUNK_OVERLAP = 50      


CHROMA_PERSIST_DIR = "data/chroma_db"
CHROMA_COLLECTION_NAME = "arxiv_papers"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  
TOP_K_RESULTS = 5       


TG_HOST = os.getenv("TG_HOST", "https://your-instance.i.tgcloud.io")
TG_GRAPHRAG_API = os.getenv("TG_GRAPHRAG_API", "http://localhost:8000")
TG_USERNAME = os.getenv("TG_USERNAME", "tigergraph")
TG_PASSWORD = os.getenv("TG_PASSWORD", "")


BENCHMARK_QUESTIONS_FILE = "data/benchmark_questions.json"
