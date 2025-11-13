"""Konfigurasi untuk knowledge-base sync."""
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "garut_knowledge_base"
RAW_API_PATH = DATA_DIR / "chroma_db" / "raw_api_data.json"
KNOWLEDGE_PATH = DATA_DIR / "chroma_db" / "knowledge_base.json"


# API endpoint
API_BASE = os.getenv("GARUT_API_BASE", "https://satudata-api.garutkab.go.id/api/data")


# Chroma / embeddings
CHROMA_DB_PATH = str(ROOT / "chatbot_db")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "dataset_embeddings")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")


# Refresh settings (seconds)
REFRESH_INTERVAL_SECONDS = int(os.getenv("REFRESH_INTERVAL_SECONDS", str(60 * 60 * 6))) # default 6 jam


# Safety: max candidates
MAX_CANDIDATES = 10