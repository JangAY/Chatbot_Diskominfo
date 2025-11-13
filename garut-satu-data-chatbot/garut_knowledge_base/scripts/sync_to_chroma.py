import json
import logging
from garut_knowledge_base.config import KNOWLEDGE_PATH, CHROMA_DB_PATH, CHROMA_COLLECTION_NAME, EMBEDDING_MODEL_NAME
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

log = logging.getLogger("sync_to_chroma")


def sync_to_chroma():
    """Sync knowledge_base.json ke database vektor Chroma."""
    log.info("Mulai sinkronisasi knowledge_base.json ke ChromaDB...")

    # Pastikan file knowledge_base.json ada
    if not KNOWLEDGE_PATH.exists():
        raise FileNotFoundError(f"File {KNOWLEDGE_PATH} tidak ditemukan. Jalankan build_knowledge_base dulu.")

    # Load data knowledge base
    with open(KNOWLEDGE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs = []
    for item in data.get("kumpulan_dataset", []):
        text = item.get("knowledge") or item.get("description") or ""
        meta = {
            "title": item.get("title"),
            "publisher": item.get("publisher"),
            "tahun": item.get("tahun"),
            "download_url": item.get("download_url") or item.get("downloadURL"),
            "landing_page": item.get("landing_page") or item.get("landingPage"),
        }

        if text.strip():
            docs.append((text, meta))

    if not docs:
        log.warning("Tidak ada dokumen untuk disimpan di Chroma.")
        return

    # Embedding model
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    # Buat atau muat ChromaDB collection
    db = Chroma(
        persist_directory=str(CHROMA_DB_PATH),
        embedding_function=embeddings,
        collection_name=CHROMA_COLLECTION_NAME
    )

    # Tambahkan dokumen
    texts, metadatas = zip(*docs)
    db.add_texts(texts=texts, metadatas=metadatas)

    # db.persist()
    log.info("Sinkronisasi selesai. Disimpan di %s", CHROMA_DB_PATH)
