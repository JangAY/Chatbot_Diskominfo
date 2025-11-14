"""
main.py – Entry point untuk membangun knowledge base otomatis
"""

import os
import sys

if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # root project
    from garut_knowledge_base.scripts.fetch_api_data import fetch_and_save
    from garut_knowledge_base.scripts.build_knowledge_base import build_knowledge_base
    from garut_knowledge_base.scripts.sync_to_chroma import sync_to_chroma
    from garut_knowledge_base.config import CHROMA_DB_PATH
else:
    from .scripts.fetch_api_data import fetch_and_save
    from .scripts.build_knowledge_base import build_knowledge_base
    from .scripts.sync_to_chroma import sync_to_chroma
    from .config import CHROMA_DB_PATH as CHROMA_PATH

def main():
    print("🚀 Memulai proses pembangunan knowledge base Garut...")
    
    # 1️⃣ Ambil data dari API
    print("📥 Mengambil data dari API...")
    fetch_and_save()

    # 2️⃣ Bangun knowledge base (gabungkan data mentah)
    print("🧠 Membangun knowledge base...")
    build_knowledge_base()

    # 3️⃣ Sinkronkan ke ChromaDB
    print("🗃️ Sinkronisasi ke ChromaDB...")
    sync_to_chroma()

    print("✅ Semua proses selesai! Knowledge base telah diperbarui.")
    print(f"📂 Lokasi ChromaDB: {CHROMA_DB_PATH}")


if __name__ == "__main__":
    main()
