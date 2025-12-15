import os
from pathlib import Path

# ... (Kode path tetap sama) ...
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "garut_knowledge_base"
RAW_API_PATH = DATA_DIR / "chroma_db" / "raw_api_data.json"
KNOWLEDGE_PATH = DATA_DIR / "chroma_db" / "knowledge_base.json"

# API endpoint
API_BASE = os.getenv("GARUT_API_BASE", "https://satudata-api.garutkab.go.id/api/data")

# Chroma / embeddings
CHROMA_DB_PATH = str(ROOT / "chatbot_db")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "dataset_embeddings")

# --- PERUBAHAN PENTING DI SINI ---
# Ganti ke model Multilingual agar sinkron dengan chatbot.py
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "paraphrase-multilingual-MiniLM-L12-v2") 

# Refresh settings (seconds)
REFRESH_INTERVAL_SECONDS = int(os.getenv("REFRESH_INTERVAL_SECONDS", str(60 * 60 * 6))) 

# Safety: max candidates
MAX_CANDIDATES = 10