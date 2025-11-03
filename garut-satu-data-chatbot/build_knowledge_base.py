import json
import chromadb
from sentence_transformers import SentenceTransformer

def build_vector_db():
    """
    Membaca data dari knowledge_base.json, membuat embedding,
    dan menyimpannya ke dalam ChromaDB.
    """
    # 1. Inisialisasi Model Embedding dan ChromaDB
    print("Inisialisasi model embedding...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    client = chromadb.PersistentClient(path="./chatbot_db")

    # Buat collection
    site_guide_collection = client.get_or_create_collection(name="panduan_situs")
    dataset_collection = client.get_or_create_collection(name="kumpulan_dataset")

    # 2. Muat file knowledge_base.json
    print("Memuat knowledge_base.json...")
    with open('knowledge_base.json', 'r', encoding='utf-8') as f:
        knowledge_base = json.load(f)

    # ========== Panduan Situs ==========
    print("Memproses data panduan situs...")
    site_guide_docs, site_guide_ids = [], []

    for key, value in knowledge_base['panduan_situs'].items():
        if isinstance(value, dict) and 'description' in value:
            site_guide_docs.append(value['description'])
            site_guide_ids.append(f"panduan_{key}")

    for fitur in knowledge_base['panduan_situs']['detail_fitur']:
        doc_content = f"Fitur {fitur['name']}: {fitur['description']}"
        site_guide_docs.append(doc_content)
        site_guide_ids.append(f"fitur_{fitur['name'].lower().replace(' ', '_')}")

    if site_guide_docs:
        site_guide_embeddings = model.encode(site_guide_docs)
        site_guide_collection.add(
            embeddings=site_guide_embeddings.tolist(),
            documents=site_guide_docs,
            ids=site_guide_ids
        )
        print(f"{len(site_guide_docs)} dokumen panduan situs berhasil disimpan.")

    # ========== Kumpulan Dataset ==========
    print("Memproses data kumpulan dataset...")
    dataset_docs, dataset_metadatas, dataset_ids = [], [], []

    for i, dataset in enumerate(knowledge_base['kumpulan_dataset']):
        # Gabungkan judul dan deskripsi untuk pencarian yang kaya konteks
        combined_text = dataset.get('knowledge', '')
        dataset_docs.append(combined_text)

        # Ambil informasi metadata (dengan nilai default jika None)
        publisher = "Tidak diketahui"
        if "Diterbitkan oleh:" in dataset.get("knowledge", ""):
            pub = dataset["knowledge"].split("Diterbitkan oleh: ")[-1].strip('.')
            if pub:
                publisher = pub

        dataset_metadatas.append({
            "title": dataset.get("original_title", "Tanpa Judul") or "Tanpa Judul",
            "url": dataset.get("landing_page", "-") or "-",
            "download_url": dataset.get("download_url", "-") or "-",
            "publisher": publisher
        })
        dataset_ids.append(f"dataset_{i}")

    # Buat embedding dan simpan ke ChromaDB
    if dataset_docs:
        dataset_embeddings = model.encode(dataset_docs)
        dataset_collection.add(
            embeddings=dataset_embeddings.tolist(),
            documents=dataset_docs,
            metadatas=dataset_metadatas,
            ids=dataset_ids
        )
        print(f"{len(dataset_docs)} dokumen dataset berhasil disimpan.")

    print("\n✅ Proses pembuatan Vector Database selesai!")

# --- Jalankan fungsi ---
if __name__ == '__main__':
    build_vector_db()
