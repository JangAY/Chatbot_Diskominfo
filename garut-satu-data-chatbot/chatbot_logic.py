import os
import re
import io
import json
import sys
import requests
import chromadb
import pandas as pd
from tabulate import tabulate
from sentence_transformers import SentenceTransformer
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import google.generativeai as genai

# --- 1. INISIALISASI ---
print("Flask: Memulai aplikasi...")
load_dotenv() 

try:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY tidak ditemukan. Pastikan ada file .env dan variabel tersebut sudah diatur.")
    genai.configure(api_key=api_key)
    print("Flask: Google Gemini API berhasil dikonfigurasi.")
except Exception as e:
    print(f"!!! KESALAHAN FATAL: Gagal mengkonfigurasi Gemini API. {e}")
    sys.exit(1)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, "chatbot_db")

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "http://localhost:8000"}})

print("Flask: Memuat model embedding...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
generation_model = genai.GenerativeModel('gemini-1.5-flash')

try:
    client = chromadb.PersistentClient(path=DB_PATH)
    site_guide_collection = client.get_collection(name="panduan_situs")
    dataset_collection = client.get_collection(name="kumpulan_dataset")
    print("Flask: Model embedding dan database berhasil dimuat.")
except Exception as e:
    print(f"!!! KESALAHAN FATAL: Tidak dapat memuat database ChromaDB. Pastikan 'build_knowledge_base.py' sudah dijalankan. Detail: {e}")
    site_guide_collection = None
    dataset_collection = None

# Ambang batas untuk filter relevansi (semakin kecil, semakin relevan)
DISTANCE_THRESHOLD = 1.1

# --- 2. FUNGSI LOGIKA CHATBOT ---

def classify_intent(query: str) -> str:
    dataset_keywords = ['data', 'dataset', 'jumlah', 'angka', 'statistik', 'laporan', 'daftar', 'tampilkan', 'berikan', 'cari']
    return 'dataset_search' if any(keyword in query.lower() for keyword in dataset_keywords) else 'general_question'

def handle_general_question(query: str) -> str:
    """Menjawab pertanyaan umum menggunakan RAG dan LLM Gemini."""
    if not site_guide_collection:
        return "Error: Database panduan situs tidak dapat diakses."

    query_embedding = embedding_model.encode([query]).tolist()
    results = site_guide_collection.query(
        query_embeddings=query_embedding,
        n_results=3
    )
    
    relevant_docs = []
    if results and results['documents'][0]:
        for i, doc in enumerate(results['documents'][0]):
            distance = results['distances'][0][i]
            print(f"Dokumen Panduan Ditemukan: '{doc[:30]}...' (Distance: {distance:.4f})")
            if distance < DISTANCE_THRESHOLD:
                relevant_docs.append(doc)

    if not relevant_docs:
        return "Maaf, saya tidak dapat menemukan informasi yang cukup relevan dengan pertanyaan Anda di dalam data saya."
    
    context = "\n\n".join(relevant_docs)

    prompt = f"""
    Anda adalah asisten AI "Satu Data Garut". Jawab pertanyaan pengguna tentang portal Satu Data Garut berdasarkan konteks yang diberikan.
    Gunakan Bahasa Indonesia yang ringkas dan jelas. Jika konteks tidak relevan, katakan Anda tidak tahu.

    KONTEKS:
    ---
    {context}
    ---

    PERTANYAAN: {query}
    JAWABAN:
    """
    
    try:
        response = generation_model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error saat memanggil Gemini API: {e}")
        return "Maaf, terjadi masalah saat mencoba menghasilkan jawaban."

def load_dataset_preview_from_url(url: str) -> str:
    """Membaca dataset dari URL dan menampilkan 5 baris pertama sebagai tabel markdown."""
    try:
        response = requests.get(url, timeout=20, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()

        file_bytes = io.BytesIO(response.content)
        ext = os.path.splitext(url.split("?")[0])[-1].lower()

        df = None
        if ext == ".csv":
            df = pd.read_csv(file_bytes)
        elif ext in [".xlsx", ".xls"]:
            df = pd.read_excel(file_bytes)
        elif ext == ".json":
            df = pd.read_json(file_bytes)
        else:
            return f"_(Pratinjau tidak tersedia untuk format `{ext}`)_"

        df_preview = df.head(5)
        df_preview = df_preview.applymap(lambda x: str(x)[:50] + '...' if isinstance(x, str) and len(x) > 50 else x)
        
        table_md = tabulate(df_preview, headers="keys", tablefmt="github", showindex=False)
        return f"```\n{table_md}\n```"
    except requests.exceptions.RequestException as e:
        print(f"Gagal mengunduh file dari URL: {url}. Error: {e}")
        return f"_(Gagal mengakses link unduhan untuk pratinjau.)_"
    except Exception as e:
        print(f"Gagal memproses file dari URL: {url}. Error: {e}")
        return f"_(Tidak dapat menampilkan pratinjau isi dataset. Error: {e})_"

def handle_dataset_search(query: str) -> str:
    """Mencari dataset dengan filter relevansi dan tahun yang lebih ketat."""
    if not dataset_collection:
        return "Error: Database dataset tidak dapat diakses."
        
    year_match = re.search(r'\b(20\d{2})\b', query)
    year = year_match.group(1) if year_match else None
    topic_query = re.sub(r'\b(20\d{2})\b', '', query).strip()

    query_embedding = embedding_model.encode([topic_query]).tolist()
    results = dataset_collection.query(query_embeddings=query_embedding, n_results=15)

    candidate_datasets = []
    if results['ids'][0]:
        for i, doc_id in enumerate(results['ids'][0]):
            distance = results['distances'][0][i]
            metadata = results['metadatas'][0][i]
            
            print(f"Dataset Kandidat: '{metadata.get('title', 'N/A')[:40]}...' (Distance: {distance:.4f})")

            # Filter 1: Relevansi Semantik (Jarak)
            if distance > DISTANCE_THRESHOLD:
                continue
            
            candidate_datasets.append(metadata)
            
    if not candidate_datasets:
        return "Maaf, saya tidak dapat menemukan dataset yang sesuai dengan permintaan Anda. Coba gunakan kata kunci yang lebih spesifik."

    # Filter 2: Logika Spesifik Berdasarkan Tahun
    final_datasets = []
    if year:
        # Jika ada tahun, cari yang paling cocok dan HANYA ambil satu.
        print(f"Melakukan filter spesifik untuk tahun: {year}")
        for data in candidate_datasets:
            title = data.get('title', '')
            # Cek apakah tahun yang diminta ada di dalam judul dataset
            if year in title:
                final_datasets.append(data)
                break # Langsung berhenti setelah menemukan kecocokan pertama
    else:
        # Jika tidak ada tahun, ambil hingga 3 kandidat teratas
        final_datasets = candidate_datasets[:3]

    if not final_datasets:
        if year:
            return f"Maaf, saya menemukan dataset yang relevan, tetapi tidak ada yang spesifik untuk tahun {year}. Coba cari tanpa menyebutkan tahun."
        else:
            return "Maaf, saya tidak dapat menemukan dataset yang cocok setelah proses penyaringan."

    # Proses dan format output
    responses = []
    for dataset_info in final_datasets:
        dataset_name = dataset_info.get('title', 'Tanpa Judul')
        dataset_url = dataset_info.get('url', '#')
        download_url = dataset_info.get('download_url')

        data_preview = "_(Tidak ada link unduhan langsung untuk pratinjau.)_"
        if download_url and download_url != "-":
            data_preview = load_dataset_preview_from_url(download_url)
        
        response_part = (
            f"### {dataset_name}\n"
            f"🔗 [Lihat Halaman Dataset]({dataset_url})\n\n"
            f"**Pratinjau Data (5 baris pertama):**\n"
            f"{data_preview}"
        )
        responses.append(response_part)
    
    if year and len(responses) == 1:
        return f"Berikut adalah data yang paling cocok untuk permintaan Anda:\n\n---\n\n" + responses[0]

    return "\n\n---\n\n".join(responses)

# --- 3. ENDPOINT API FLASK ---
@app.route("/api/chat", methods=["POST"])
def handle_chat():
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({"error": "Permintaan tidak valid, 'query' dibutuhkan."}), 400

    user_query = data['query']
    intent = classify_intent(user_query)
    
    if intent == 'general_question':
        response_text = handle_general_question(user_query)
    elif intent == 'dataset_search':
        response_text = handle_dataset_search(user_query)
    else:
        response_text = handle_general_question(user_query)

    return jsonify({"reply": response_text})

# --- 4. JALANKAN SERVER FLASK ---
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)

