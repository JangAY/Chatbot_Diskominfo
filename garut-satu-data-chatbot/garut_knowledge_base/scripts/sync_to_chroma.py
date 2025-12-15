import json
import logging
import os
import shutil
import chromadb
from sentence_transformers import SentenceTransformer
from garut_knowledge_base.config import (
    KNOWLEDGE_PATH, 
    CHROMA_DB_PATH, 
    CHROMA_COLLECTION_NAME, 
    EMBEDDING_MODEL_NAME
)
# Pastikan fungsi ini bisa diimport
from garut_knowledge_base.scripts.build_knowledge_base import build_embedding_text

log = logging.getLogger("sync_to_chroma")

def sync_to_chroma():
    """Sync knowledge_base.json ke database vektor Chroma (Native Client)."""
    log.info(f"Mulai sinkronisasi ke ChromaDB di: {CHROMA_DB_PATH}")
    log.info(f"Menggunakan Model: {EMBEDDING_MODEL_NAME}")

    if not KNOWLEDGE_PATH.exists():
        raise FileNotFoundError(f"File {KNOWLEDGE_PATH} tidak ditemukan. Jalankan build_knowledge_base dulu.")

    # 1. LOAD DATA JSON
    with open(KNOWLEDGE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    datasets = data.get("kumpulan_dataset", [])
    if not datasets:
        log.warning("Tidak ada dataset untuk diproses.")
        return

    # 2. LOAD MODEL (SentenceTransformer Langsung)
    # Ini memastikan vektor yang dihasilkan identik dengan chatbot.py
    log.info("Memuat model embedding...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # 3. RESET DATABASE (Hapus folder lama agar bersih)
    if os.path.exists(CHROMA_DB_PATH):
        log.warning("Menghapus database lama untuk full refresh...")
        try:
            shutil.rmtree(CHROMA_DB_PATH)
        except Exception as e:
            log.error(f"Gagal menghapus folder DB lama: {e}")

    # 4. INIT CHROMA CLIENT
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    
    # Hapus koleksi jika masih ada sisa (double check)
    try:
        client.delete_collection(CHROMA_COLLECTION_NAME)
    except:
        pass
        
    collection = client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)

    # 5. PREPARE BATCH DATA
    ids = []
    documents = []
    metadatas = []
    embeddings = []

    total = len(datasets)
    log.info(f"Memproses {total} dataset...")

    for i, item in enumerate(datasets):
        # Build Text yang akan divektorisasi
        text_content = build_embedding_text(item)
        
        # Build Metadata (pastikan flat dictionary)
        meta = {
            "title": str(item.get("title") or ""),
            "publisher": str(item.get("publisher") or ""),
            "tahun": str(item.get("tahun") or ""), 
            "landing_page": str(item.get("landing_page") or item.get("download_url") or ""),
            "download_url": str(item.get("download_url") or "")
        }

        # Generate Vector
        vector = model.encode(text_content).tolist()

        ids.append(f"doc_{i}")
        documents.append(text_content)
        metadatas.append(meta)
        embeddings.append(vector)

        if (i + 1) % 10 == 0:
            print(f"   Processed {i + 1}/{total}...")

    # 6. INSERT TO CHROMA
    if ids:
        collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )
        log.info(f"✅ Sukses menyimpan {len(ids)} dokumen ke ChromaDB.")
    else:
        log.warning("Tidak ada data valid yang bisa disimpan.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sync_to_chroma()