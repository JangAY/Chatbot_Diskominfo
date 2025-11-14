import json
import logging
from bs4 import BeautifulSoup
from garut_knowledge_base.config import KNOWLEDGE_PATH, CHROMA_DB_PATH, CHROMA_COLLECTION_NAME, EMBEDDING_MODEL_NAME
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

log = logging.getLogger("sync_to_chroma")


# ============================================================
#  FUNGSI BARU — Wajib dipakai agar embedding jadi akurat 🔥
# ============================================================
def build_embedding_text(item):
    title = item.get("title", "")
    tahun = str(item.get("tahun", ""))
    publisher = item.get("publisher", "")

    # Bersihkan HTML
    desc = item.get("description") or item.get("deskripsi") or ""
    desc_plain = BeautifulSoup(desc, "html.parser").get_text(" ", strip=True)

    # Sinyal kuat utk reranker dan vector search
    enriched = (
        f"Ini adalah dataset berjudul {title}. "
        f"Dataset ini membahas tentang {title}. "
        f"Topik utama dataset ini: {title}. "
        f"Dataset terkait tahun {tahun}. "
        f"Kata kunci penting: {title} tahun {tahun}, {desc_plain}. "
    )

    text = f"""
Judul Dataset: {title}
Tahun: {tahun}
Penerbit: {publisher}
Deskripsi: {desc_plain}

{enriched}
"""
    return text.strip()


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
