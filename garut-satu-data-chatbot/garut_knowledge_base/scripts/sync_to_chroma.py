import json
import logging
from bs4 import BeautifulSoup
from garut_knowledge_base.config import KNOWLEDGE_PATH, CHROMA_DB_PATH, CHROMA_COLLECTION_NAME, EMBEDDING_MODEL_NAME
from garut_knowledge_base.scripts.build_knowledge_base import build_embedding_text
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

log = logging.getLogger("sync_to_chroma")

def sync_to_chroma():
    """Sync knowledge_base.json ke database vektor Chroma."""
    log.info("Mulai sinkronisasi knowledge_base.json ke ChromaDB...")

    if not KNOWLEDGE_PATH.exists():
        raise FileNotFoundError(f"File {KNOWLEDGE_PATH} tidak ditemukan. Jalankan build_knowledge_base dulu.")

    with open(KNOWLEDGE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs = []
    metadatas = []

    for item in data.get("kumpulan_dataset", []):
        
        title = item.get("title") or "Tidak ada judul"
        publisher = item.get("publisher") or ""
        tahun = item.get("tahun") or ""
        desc = item.get("description") or ""
        download_url = item.get("download_url") or item.get("downloadURL") or ""
        landing_page = item.get("landing_page") or item.get("landingPage") or download_url

        # 🔥 PAKAI PEMBANGUN EMBEDDING BARU
        text = build_embedding_text(item)

        meta = {
            "title": title,
            "publisher": publisher,
            "tahun": tahun,
            "description": desc,
            "download_url": download_url,
            "landing_page": landing_page,
        }

        docs.append(text)
        metadatas.append(meta)

    if not docs:
        log.warning("Tidak ada dokumen untuk disimpan di Chroma.")
        return

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    db = Chroma(
        persist_directory=str(CHROMA_DB_PATH),
        embedding_function=embeddings,
        collection_name=CHROMA_COLLECTION_NAME
    )

    db.add_texts(texts=docs, metadatas=metadatas)

    log.info("Sinkronisasi selesai. Total dataset disimpan: %d", len(docs))
