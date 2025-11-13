import os
import json
import requests
import chromadb
from sentence_transformers import SentenceTransformer

# --- KONFIGURASI ---
API_URL = "https://satudata-api.garutkab.go.id/api/data"
STATIC_KNOWLEDGE_FILE = "knowledge_base.json"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, "chatbot_db")

def main():
    print(f"Memulai pembaruan basis pengetahuan dari {API_URL}...")
    
    # 1. Inisialisasi Model dan DB
    print("Memuat model embedding (ini mungkin perlu waktu)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    client = chromadb.PersistentClient(path=DB_PATH)

    # 2. Proses Pengetahuan Statis (Panduan Situs)
    try:
        with open(os.path.join(SCRIPT_DIR, STATIC_KNOWLEDGE_FILE), 'r', encoding='utf-8') as f:
            static_data = json.load(f)
        
        try:
            client.delete_collection(name="panduan_situs")
        except Exception:
            pass # Tidak apa-apa jika belum ada
        
        site_guide_collection = client.get_or_create_collection(name="panduan_situs")
        
        panduan_docs = []
        panduan_ids = []
        for key, value in static_data.get("panduan_situs", {}).items():
            if "description" in value:
                panduan_docs.append(value["description"])
                panduan_ids.append(f"panduan_{key}")
        
        if panduan_docs:
            panduan_embeddings = model.encode(panduan_docs).tolist()
            site_guide_collection.add(
                embeddings=panduan_embeddings,
                documents=panduan_docs,
                ids=panduan_ids
            )
            print(f"Berhasil memuat {len(panduan_docs)} dokumen panduan situs.")
        
    except Exception as e:
        print(f"Gagal memuat pengetahuan statis: {e}")

    # 3. Proses Pengetahuan Dinamis (Dataset dari API)
    print(f"Menghubungi API di {API_URL}...")
    try:
        response = requests.get(API_URL, timeout=30)
        response.raise_for_status() 
        api_data = response.json().get('data', [])
        
        if not api_data:
            print("Peringatan: API tidak mengembalikan data dataset.")
            return

        print(f"Berhasil mengambil {len(api_data)} dataset dari API.")

        try:
            client.delete_collection(name="kumpulan_dataset")
        except Exception:
            pass 
            
        dataset_collection = client.get_or_create_collection(name="kumpulan_dataset")

        dataset_docs = []
        dataset_metadatas = []
        dataset_ids = []

        for i, dataset in enumerate(api_data):
            # Ambil data level atas
            title = dataset.get('judul', 'Tanpa Judul')
            description = dataset.get('deskripsi', 'Tidak ada deskripsi.')
            publisher = dataset.get('publisher', {}).get('name', 'Tidak diketahui') # Berdasarkan struktur Anda
            url = dataset.get('landingPage', '#')
            
            # --- PERBAIKAN: Ekstrak downloadURL dari 'distribution' ---
            download_url = "-" # Default
            distribusi = dataset.get('distribution', [])
            if distribusi and isinstance(distribusi, list) and len(distribusi) > 0:
                # Ambil downloadURL dari distribusi pertama yang valid
                if distribusi[0].get('downloadURL'):
                    download_url = distribusi[0].get('downloadURL')

            # Dokumen untuk di-embed
            doc_content = f"Judul: {title}. Deskripsi: {description}. Diterbitkan oleh: {publisher}."
            
            dataset_docs.append(doc_content)
            dataset_ids.append(f"dataset_{dataset.get('identifier', i)}") # Gunakan identifier unik
            dataset_metadatas.append({
                "title": title,
                "url": url,
                "publisher": publisher,
                "download_url": download_url # Simpan link download
            })

        if dataset_docs:
            print("Membuat embedding untuk data API...")
            dataset_embeddings = model.encode(dataset_docs).tolist()
            dataset_collection.add(
                embeddings=dataset_embeddings,
                documents=dataset_docs,
                metadatas=dataset_metadatas,
                ids=dataset_ids
            )
            print(f"Berhasil memuat dan memproses {len(dataset_docs)} dataset dari API.")

    except Exception as e:
        print(f"Gagal mengambil atau memproses data dari API: {e}")

    print("\nPembaruan basis pengetahuan selesai!")

if __name__ == "__main__":
    main()