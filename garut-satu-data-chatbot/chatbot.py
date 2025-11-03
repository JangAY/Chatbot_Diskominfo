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

# --- Import Fungsi Preprocessing ---
from preprocessing_utils import preprocess_text

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
# PERBAIKAN: Izinkan permintaan HANYA dari backend Laravel Anda
CORS(app, resources={r"/api/*": {"origins": "http://localhost:8000"}})

print("Flask: Memuat model embedding...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
generation_model = genai.GenerativeModel('gemini-2.5-flash')

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

def classify_intent(processed_query: str, raw_query: str) -> str:
    """
    Mengklasifikasikan intent berdasarkan query yang sudah diproses (di-stem)
    DAN query asli (raw) untuk menangkap keyword yang mungkin terhapus.
    """
    raw_lower = raw_query.lower().strip()
    
    # --- PERUBAHAN ---
    # 1. Deteksi permintaan daftar sektor
    if raw_lower == "apa saja dataset yang tersedia?":
        return "list_sectors"
        
    # 2. Deteksi pencarian berdasarkan sektor
    if raw_lower.startswith("tampilkan dataset sektor"):
        return "dataset_sector_search"

    # 3. Deteksi pencarian data umum (untuk analisis)
    stemmed_keywords = [
        'data', 'dataset', 'jumlah', 'angka', 'statistik', 
        'lapor', 'daftar', 'tampil', 'beri', 'cari'
    ]
    raw_keywords_to_check = ['jumlah', 'statistik', 'angka', 'berapa']
    
    if any(keyword in processed_query for keyword in stemmed_keywords):
        return 'dataset_search'
    
    if any(keyword in raw_query_lower for keyword in raw_keywords_to_check):
        return 'dataset_search'
        
    # 4. Pertanyaan umum
    return 'general_question'

def handle_general_question(query: str) -> str:
    """
    Menjawab pertanyaan umum menggunakan RAG dan LLM Gemini.
    """
    if not site_guide_collection:
        return "Error: Database panduan situs tidak dapat diakses."

    query_embedding = embedding_model.encode([query]).tolist()
    results = site_guide_collection.query(
        query_embeddings=query_embedding,
        n_results=3
    )
    
    relevant_docs = []
    if results and results.get('documents') and results['documents'][0]:
        distances = results['distances'][0]
        documents = results['documents'][0]
        
        for i in range(len(distances)):
            distance = distances[i]
            doc = documents[i]
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
        block_reason = response.prompt_feedback.block_reason.name
        if block_reason != "BLOCK_REASON_UNSPECIFIED":
            print(f"Error: Panggilan LLM (general) diblokir karena '{block_reason}'.")
            return f"_(Maaf, AI tidak dapat memproses pertanyaan ini. Alasan: {block_reason})_"
        return response.text.strip()
    except Exception as e:
        print(f"Error saat memanggil Gemini API: {e}")
        return "Maaf, terjadi masalah saat mencoba menghasilkan jawaban."

def load_full_dataframe_from_url(url: str) -> pd.DataFrame | None:
    """Membaca seluruh data dari URL dan mengembalikannya sebagai DataFrame."""
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
            print(f"Format tidak didukung untuk pemuatan penuh: {ext}")
            return None
        return df
    except Exception as e:
        print(f"Gagal memuat DataFrame penuh dari URL: {url}. Error: {e}")
        return None

def analyze_data_with_llm(df: pd.DataFrame, query: str) -> str:
    """Menganalisis DataFrame untuk menjawab pertanyaan spesifik menggunakan LLM."""
    if df is None:
        return "_(Gagal memuat data untuk dianalisis.)_"
    table_md = tabulate(df.head(25), headers="keys", tablefmt="github", showindex=False)
    prompt = f"""
    Tugas Anda adalah bertindak sebagai pencari data.
    Gunakan HANYA tabel data di bawah ini untuk menjawab pertanyaan pengguna.
    Pertanyaan Pengguna:
    "{query}"
    Tabel Data (sampel 25 baris):
    ---
    {table_md}
    ---
    Jawaban (Jawab dalam satu kalimat singkat dan sebutkan angkanya dari tabel. 
    Contoh: 'Berdasarkan tabel, jumlah A adalah 123.'
    Jika data spesifik tidak ada di tabel, katakan 'Maaf, data spesifik tersebut tidak ditemukan di dalam pratinjau tabel.'):
    """
    try:
        response = generation_model.generate_content(prompt)
        block_reason = response.prompt_feedback.block_reason.name
        if block_reason != "BLOCK_REASON_UNSPECIFIED":
             print(f"Error: Panggilan LLM (analisis) diblokir karena '{block_reason}'.")
             return f"_(Maaf, AI tidak dapat menganalisis data ini. Alasan: {block_reason})_"
        return response.text.strip()
    except Exception as e:
        error_message = str(e)
        print(f"Error saat menganalisis data dengan LLM: {error_message}")
        if "prompt" in error_message.lower() or "token" in error_message.lower():
            return "_(Terjadi kesalahan: Data terlalu besar untuk dianalisis oleh AI.)_"
        if "safety" in error_message.lower():
            return "_(Terjadi kesalahan: Konten data diblokir oleh filter keamanan.)_"
        return f"_(Terjadi kesalahan saat menganalisis data: {error_message})_"

def summarize_with_llm(text_to_summarize: str) -> str:
    """Meringkas deskripsi dataset menjadi satu kalimat menggunakan LLM."""
    try:
        if len(text_to_summarize.split()) < 15:
            return text_to_summarize
        prompt = f"Ringkas deskripsi dataset berikut dalam satu kalimat deskriptif dalam Bahasa Indonesia:\n\nDESKRIPSI ASLI:\n{text_to_summarize}\n\nRINGKASAN (satu kalimat):"
        response = generation_model.generate_content(prompt)
        block_reason = response.prompt_feedback.block_reason.name
        if block_reason != "BLOCK_REASON_UNSPECIFIED":
            print(f"Error: Panggilan LLM (ringkasan) diblokir: {block_reason}")
            return text_to_summarize[:150].strip() + "..." # Fallback
        return response.text.strip()
    except Exception as e:
        print(f"Error saat meringkas teks: {e}")
        return text_to_summarize[:150].strip() + "..."

# --- FUNGSI BARU ---
def handle_list_sectors() -> dict:
    """Mengambil daftar sektor (penerbit) unik dari database."""
    if not dataset_collection:
        return {"reply": "Error: Database dataset tidak dapat diakses."}
    
    try:
        # Ambil semua data. get() tanpa argumen mengembalikan semua.
        data = dataset_collection.get(include=["metadatas"])
        
        # Ekstrak 'publisher' dari metadata
        publishers = set()
        for metadata in data['metadatas']:
            if 'publisher' in metadata:
                publishers.add(metadata['publisher'])
        
        # Buat daftar tombol baru
        new_replies = []
        for publisher in sorted(list(publishers)): # Urutkan A-Z
            new_replies.append({
                "label": publisher,
                "value": f"Tampilkan dataset sektor {publisher}"
            })

        return {
            "reply": "Tentu, berikut adalah daftar sektor (OPD) yang datanya tersedia. Silakan pilih salah satu:",
            "newQuickReplies": new_replies
        }
    except Exception as e:
        print(f"Error saat mengambil daftar sektor: {e}")
        return {"reply": "Maaf, terjadi kesalahan saat mengambil daftar sektor."}

# --- FUNGSI BARU ---
def handle_sector_search(query: str) -> dict:
    """Mencari dataset berdasarkan sektor (penerbit)."""
    if not dataset_collection:
        return {"reply": "Error: Database dataset tidak dapat diakses."}
        
    # Ekstrak nama sektor dari query "Tampilkan dataset sektor [Nama Sektor]"
    try:
        sector_name = query.split("Tampilkan dataset sektor ", 1)[1]
    except IndexError:
        return {"reply": "Maaf, format pencarian sektor tidak valid."}

    print(f"Mencari dataset untuk sektor: {sector_name}")
    
    try:
        results = dataset_collection.query(
            query_texts=[sector_name], # Gunakan nama sektor sebagai query semantik
            n_results=5,
            where={"publisher": sector_name} # Filter ketat berdasarkan metadata
        )
        
        found_docs = results['documents'][0]
        if not found_docs:
            return {"reply": f"Maaf, saya tidak menemukan dataset untuk sektor '{sector_name}'."}

        # Format sebagai daftar markdown
        response_list = [f"Berikut adalah 5 dataset teratas untuk sektor '{sector_name}':\n"]
        for i, doc in enumerate(found_docs):
            title = results['metadatas'][0][i].get('title', 'Tanpa Judul')
            response_list.append(f"* **{title}**")
        
        return {"reply": "\n".join(response_list)}
        
    except Exception as e:
        print(f"Error saat mencari berdasarkan sektor: {e}")
        return {"reply": "Maaf, terjadi kesalahan saat mencari dataset sektor."}

def handle_dataset_search(query: str) -> dict:
    """
    Mencari dataset SPESIFIK, menganalisisnya, dan menampilkan jawaban.
    """
    if not dataset_collection:
        return {"reply": "Error: Database dataset tidak dapat diakses."}
        
    year_match = re.search(r'\b(20\d{2})\b', query)
    year = year_match.group(1) if year_match else None
    topic_query = re.sub(r'\b(20\d{2})\b', '', query).strip()

    query_embedding = embedding_model.encode([topic_query]).tolist()
    results = dataset_collection.query(
        query_embeddings=query_embedding, 
        n_results=15,
        include=["metadatas", "documents", "distances"]
    )

    candidate_datasets = []
    if results and results.get('distances') and results['distances'][0]:
        distances = results['distances'][0]
        metadatas = results['metadatas'][0]
        documents = results['documents'][0]
        for i in range(len(distances)):
            if distances[i] < DISTANCE_THRESHOLD:
                candidate_datasets.append((metadatas[i], documents[i]))
            
    if not candidate_datasets:
        return {"reply": "Maaf, saya tidak dapat menemukan dataset yang sesuai dengan permintaan Anda. Coba gunakan kata kunci yang lebih spesifik."}

    final_datasets = []
    if year:
        for data, doc in candidate_datasets:
            if year in data.get('title', ''):
                final_datasets.append((data, doc))
                break 
    else:
        final_datasets = candidate_datasets[:1]

    if not final_datasets:
        if year:
            return {"reply": f"Maaf, saya menemukan dataset yang relevan, tetapi tidak ada yang spesifik untuk tahun {year}."}
        else:
            return {"reply": "Maaf, saya tidak dapat menemukan dataset yang cocok setelah proses penyaringan."}

    responses = []
    for dataset_info, dataset_doc in final_datasets:
        dataset_name = dataset_info.get('title', 'Tanpa Judul')
        dataset_url = dataset_info.get('url', '#')
        download_url = dataset_info.get('download_url')
        
        data_analysis_response = ""
        table_preview = "_(Tidak ada link unduhan langsung untuk pratinjau.)_"
        full_df = None

        if download_url and download_url != "-":
            full_df = load_full_dataframe_from_url(download_url)
            if full_df is not None:
                data_analysis_response = analyze_data_with_llm(full_df, query)
                df_preview = full_df.head(5)
                df_preview = df_preview.applymap(lambda x: str(x)[:50] + '...' if isinstance(x, str) and len(x) > 50 else x)
                table_preview = f"```\n{tabulate(df_preview, headers='keys', tablefmt='github', showindex=False)}\n```"
            else:
                data_analysis_response = "_(Gagal memuat data dari link untuk dianalisis.)_"
                table_preview = "_(Gagal memuat pratinjau tabel.)_"
        
        response_part = (
            f"**Jawaban:**\n"
            f"{data_analysis_response}\n\n"
            f"---\n\n"
            f"**Pratinjau Data (5 baris pertama):**\n"
            f"{table_preview}\n\n"
            f"---\n\n"
            f"🔗 **[Lihat Sumber Data Asli]({dataset_url})**"
        )
        responses.append(response_part)
    
    if year and len(responses) == 1:
        return {"reply": f"Berikut adalah data yang paling cocok untuk permintaan Anda:\n\n---\n\n" + responses[0]}

    return {"reply": "\n\n---\n\n".join(responses)}

# --- 3. ENDPOINT API FLASK ---
@app.route("/api/chat", methods=["POST"])
def handle_chat():
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({"error": "Permintaan tidak valid, 'query' dibutuhkan."}), 400

    user_query = data['query']
    processed_query = preprocess_text(user_query)
    intent = classify_intent(processed_query, user_query)
    
    print(f"Query Asli: '{user_query}'")
    print(f"Query Proses: '{processed_query}'")
    print(f"Intent Dideteksi: '{intent}'")

    response_data = {} # Gunakan dict untuk menampung semua balasan
    
    if intent == 'general_question':
        response_data['reply'] = handle_general_question(user_query)
    elif intent == 'dataset_search':
        response_data = handle_dataset_search(user_query) # Ini sudah mengembalikan dict
    elif intent == 'list_sectors':
        response_data = handle_list_sectors() # Ini mengembalikan dict (termasuk newQuickReplies)
    elif intent == 'dataset_sector_search':
        response_data = handle_sector_search(user_query) # Ini mengembalikan dict
    else:
        response_data['reply'] = handle_general_question(user_query)

    # Pastikan selalu ada 'reply' di balasan
    if 'reply' not in response_data:
        response_data['reply'] = "Maaf, terjadi kesalahan yang tidak terduga."

    return jsonify(response_data) # Kirim seluruh data (bisa berisi 'reply' dan 'newQuickReplies')

# --- 4. JALANKAN SERVER FLASK ---
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)