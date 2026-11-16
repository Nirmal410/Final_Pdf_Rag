
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Directories
UPLOAD_DIR = BASE_DIR / os.getenv(
    "UPLOAD_DIR", "data/uploads"
)

PROCESSED_DIR = BASE_DIR / os.getenv(
    "PROCESSED_DIR", "data/processed"
)

CHROMA_DIR = BASE_DIR / os.getenv(
    "CHROMA_DIR", "chroma_db"
)

# Create directories if they don't exist
for directory in [UPLOAD_DIR, PROCESSED_DIR, CHROMA_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Models
LLM_MODEL = os.getenv(
    "OLLAMA_MODEL", "qwen2.5:3b"
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"
)

RERANKER_MODEL = os.getenv(
    "RERANKER_MODEL",
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

# Chunking
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 700))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))

# Retrieval
TOP_K_DENSE = int(os.getenv("TOP_K_DENSE", 5))
TOP_K_BM25 = int(os.getenv("TOP_K_BM25", 5))
TOP_K_RERANK = int(os.getenv("TOP_K_RERANK", 5))
FINAL_TOP_K = int(os.getenv("FINAL_TOP_K", 3))

# Ollama
OLLAMA_BASE_URL = "http://localhost:11434"

# Chroma collection
COLLECTION_NAME = "pdf_rag_collection"
