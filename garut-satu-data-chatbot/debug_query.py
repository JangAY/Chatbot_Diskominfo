import json
from chromadb import PersistentClient
from garut_knowledge_base.config import KNOWLEDGE_PATH

# Load knowledge_base.json
kb = json.loads(KNOWLEDGE_PATH.read_text(encoding="utf-8"))

# Init chroma client (NEW ARCHITECTURE)
client = PersistentClient(path="./chatbot_db")

# Load collection
coll = client.get_collection("dataset_embeddings")


def debug_query(q: str, top_k: int = 5):
    print("========================================")
    print("🔎 Query:", q)
    print("========================================\n")

    result = coll.query(
        query_texts=[q],
        n_results=top_k
    )

    ids = result["ids"][0]
    dists = result["distances"][0]
    metas = result["metadatas"][0]
    docs = result["documents"][0]

    print("🔍 Hasil semantic search (top", top_k, "):\n")

    for i, (id_, dist, meta, doc) in enumerate(zip(ids, dists, metas, docs)):
        print(f"#{i+1}")
        print(f"ID: {id_}")
        print(f"Distance: {dist:.4f}")
        print(f"Dataset Title : {meta.get('title')}")
        print(f"Tahun         : {meta.get('tahun')}")
        print(f"Download URL  : {meta.get('download_url')}")
        print("-----------------------------")
        print("Cuplikan Embedding Text:")
        print(doc[:600], "...")  # preview
        print("\n")

    print("========================================")


if __name__ == "__main__":
    print("DEBUG TOOL — Cek hasil semantic search")
    print("Masukkan pertanyaan contoh:")
    print("  > jumlah penduduk miskin 2022\n")

    try:
        while True:
            q = input("\nMasukkan query: ").strip()
            if not q:
                continue
            debug_query(q)
    except KeyboardInterrupt:
        print("\nKeluar.")
