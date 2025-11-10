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
import traceback # --- TAMBAHAN UNTUK DEBUGGING ---

# --- Import Fungsi Preprocessing ---
from preprocessing_utils import preprocess_text

# --- 1. INISIALISASI ---
print("Flask: Memulai aplikasi...")
load_dotenv() 

try:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY tidak ditemukan.")
    
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    ]
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
generation_model = genai.GenerativeModel(
    model_name='gemini-2.5-flash',
    safety_settings=safety_settings
)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')


try:
    client = chromadb.PersistentClient(path=DB_PATH)
    site_guide_collection = client.get_collection(name="panduan_situs")
    dataset_collection = client.get_collection(name="kumpulan_dataset")
    print("Flask: Model embedding dan database berhasil dimuat.")
except Exception as e:
    print(f"!!! KESALAHAN FATAL: Tidak dapat memuat database ChromaDB. Pastikan 'update_knowledge_base.py' sudah dijalankan. Detail: {e}")
    site_guide_collection = None
    dataset_collection = None

DISTANCE_THRESHOLD = 1.1

# --- 2. FUNGSI LOGIKA CHATBOT ---

# --- FUNGSI BARU: Koreksi Typo ---
def correct_query_with_llm(user_query: str) -> str:
    """
    Menggunakan LLM untuk memperbaiki typo pada query pengguna.
    """
    print(f"Memulai koreksi typo untuk: {user_query}")
    try:
        # Prompt ini sangat ketat untuk mencegah LLM mengubah makna
        prompt = (
            "Anda adalah asisten koreksi ejaan (spell corrector).\n"
            "Konteks pencarian adalah data untuk 'Kabupaten Garut'.\n"
            "Koreksi hanya kesalahan ejaan (typo) pada 'Pertanyaan Pengguna' berikut.\n"
            "JANGAN mengubah makna, JANGAN menambah kata baru, JANGAN menjawab pertanyaan.\n"
            "Jika tidak ada typo, kembalikan Pertanyaan Pengguna apa adanya.\n\n"
            f"Pertanyaan Pengguna: \"{user_query}\"\n\n"
            "Hasil Koreksi (hanya teks yang dikoreksi):"
        )
        response = generation_model.generate_content(prompt)
        # Ambil teks, hapus tanda kutip yang mungkin ditambahkan LLM
        corrected_query = response.text.strip().strip('"')
        
        if corrected_query.lower() != user_query.lower():
            print(f"Typo dikoreksi: '{user_query}' -> '{corrected_query}'")
        else:
            print("Tidak ada typo ditemukan.")
        return corrected_query
    except Exception as e:
        print(f"Error saat koreksi typo: {e}. Menggunakan query asli.")
        return user_query

def classify_intent(processed_query: str, raw_query: str) -> str:
    raw_lower = raw_query.lower().strip()
    
    if raw_lower == "apa saja dataset yang tersedia?":
        return "list_sectors"
    
    # --- PERBAIKAN: Deteksi pencarian sektor ---
    if raw_lower.startswith("tampilkan dataset sektor"):
        return "dataset_sector_search"

    general_keywords = [
        "siapa kamu", "apa itu", "bagaimana cara", "jelaskan", "apa yang dimaksud"
    ]
    if any(raw_lower.startswith(key) for key in general_keywords):
        return "general_question"
    
    # Jika bukan di atas, baru jalankan agen data
    return "run_data_agent"

def handle_general_question(query: str) -> dict:
    # (Fungsi ini tidak berubah)
    if not site_guide_collection:
        return {"reply": "Error: Database panduan situs tidak dapat diakses."}
    query_embedding = embedding_model.encode([query]).tolist()
    results = site_guide_collection.query(
        query_embeddings=query_embedding,
        n_results=1,
        include=["documents", "distances"]
    )
    if not results['documents'][0] or results['distances'][0][0] > DISTANCE_THRESHOLD:
        return {"reply": "Maaf, saya tidak dapat menemukan informasi yang relevan dengan pertanyaan umum Anda."}
    context = results['documents'][0][0]
    
    # --- PERBAIKAN: Konteks Garut & Perbaikan F-String ---
    prompt = (
        "Anda adalah asisten AI \"Satu Data Garut\". Jawab pertanyaan pengguna tentang portal Satu Data Garut berdasarkan konteks yang diberikan.\n"
        "Gunakan Bahasa Indonesia yang ringkas dan jelas. JANGAN PERNAH menyebut 'Indonesia', fokus hanya pada 'Kabupaten Garut'.\n"
        "KONTEKS:\n"
        "---\n"
        f"{context}\n"
        "---\n"
        f"PERTANYAAN: {query}\n"
        "JAWABAN:"
    )
    
    try:
        response = generation_model.generate_content(prompt)
        return {"reply": response.text.strip()}
    except Exception as e:
        print(f"Error saat memanggil Gemini API (General): {e}")
        return {"reply": "Maaf, terjadi masalah saat mencoba menghasilkan jawaban."}

def handle_list_sectors() -> dict:
    # (Fungsi ini tidak berubah)
    if not dataset_collection:
        return {"reply": "Error: Database dataset tidak dapat diakses."}
    try:
        data = dataset_collection.get(include=["metadatas"])
        publishers = set(metadata['publisher'] for metadata in data['metadatas'] if 'publisher' in metadata and metadata['publisher'])
        if not publishers:
            return {"reply": "Maaf, saat ini tidak ada sektor yang terdaftar di database."}
        new_replies = [{"label": p, "value": f"Tampilkan dataset sektor {p}"} for p in sorted(list(publishers))]
        return {
            "reply": "Tentu, berikut adalah daftar sektor (OPD) yang datanya tersedia. Silakan pilih salah satu:",
            "newQuickReplies": new_replies
        }
    except Exception as e:
        print(f"Error saat mengambil daftar sektor: {e}")
        return {"reply": "Maaf, terjadi kesalahan saat mengambil daftar sektor."}

# --- FUNGSI ANALISIS DATA (Inti) ---

def load_full_dataframe_from_url(url: str) -> pd.DataFrame | None:
    # (Fungsi ini tidak berubah)
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
            return None
        return df
    except Exception as e:
        print(f"Gagal memuat DataFrame penuh dari URL: {url}. Error: {e}")
        return None

def analyze_data_with_llm(df: pd.DataFrame, query: str) -> str:
    # (Fungsi ini tidak berubah, tetapi prompt-nya diperbaiki)
    if df is None:
        return "_(Gagal memuat data untuk dianalisis.)_"
    table_md = tabulate(df.head(25), headers="keys", tablefmt="github", showindex=False)
    
    # --- PERBAIKAN: Konteks Garut & Perbaikan F-String ---
    prompt = (
        "Tugas Anda adalah bertindak sebagai analis data yang sangat teliti untuk 'Kabupaten Garut'.\n"
        "Gunakan HANYA tabel data di bawah ini untuk menjawab pertanyaan pengguna.\n"
        "Ikuti langkah-langkah berpikir ini:\n"
        "Langkah 1: Baca Pertanyaan Pengguna. Identifikasi kata kunci utamanya (misal: 'penduduk miskin', '2022').\n"
        "Langkah 2: Saring baris tabel. Cari baris di tabel yang kolomnya cocok dengan kata kunci.\n"
        "Langkah 3: Ekstrak data. Temukan kolom yang berisi angka/jawaban.\n"
        "Langkah 4: Format jawaban. Berikan jawaban dalam satu kalimat singkat.\n"
        "PENTING:\n"
        "- Konteks data ini adalah 'Kabupaten Garut'. JANGAN menyebut 'Indonesia'.\n"
        "- Jika Anda tidak dapat menemukan baris yang cocok persis, katakan \"Maaf, data spesifik tersebut tidak ditemukan di dalam pratinjau tabel.\"\n"
        f"Pertanyaan Pengguna:\n\"{query}\"\n\n"
        "Tabel Data (sampel 25 baris):\n"
        "---\n"
        f"{table_md}\n"
        "---\n"
        "JAWABAN ANDA (satu kalimat, fokus pada Garut):"
    )
    
    try:
        response = generation_model.generate_content(prompt)
        block_reason = "BLOCK_REASON_UNSPECIFIED"
        if response.prompt_feedback:
            block_reason = response.prompt_feedback.block_reason.name
        if block_reason != "BLOCK_REASON_UNSPECIFIED":
             return f"_(Maaf, AI tidak dapat menganalisis data ini. Alasan: {block_reason})_"
        return response.text.strip()
    except Exception as e:
        error_message = str(e)
        if "prompt" in error_message.lower() or "token" in error_message.lower():
            return "_(Terjadi kesalahan: Data terlalu besar untuk dianalisis oleh AI.)_"
        return f"_(Terjadi kesalahan saat menganalisis data.)_"

def summarize_with_llm(text_to_summarize: str) -> str:
    # (Fungsi ini tidak berubah, tetapi prompt-nya diperbaiki)
    try:
        if len(text_to_summarize.split()) < 15:
            return text_to_summarize
        
        # --- PERBAIKAN: Konteks Garut & Perbaikan F-String ---
        prompt = (
            "Ringkas deskripsi dataset berikut dalam satu kalimat deskriptif dalam Bahasa Indonesia. Konteksnya adalah data 'Kabupaten Garut'.\n\n"
            "DESKRIPSI ASLI:\n"
            f"{text_to_summarize}\n\n"
            "RINGKASAN (satu kalimat):"
        )
        
        response = generation_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return text_to_summarize[:150].strip() + "..."

# --- FUNGSI "AGENT" BARU ---

def decompose_query_with_llm(user_query: str) -> list[str]:
    # (Fungsi ini tidak berubah, tetapi prompt-nya diperbaiki)
    print(f"Memulai dekomposisi untuk: {user_query}")
    schema = {
        "type": "ARRAY",
        "items": { "type": "STRING" }
    }
    
    # --- PERBAIKAN: Konteks Garut & Perbaikan F-String ---
    prompt = (
        "Anda adalah asisten pemecah masalah. Tugas Anda adalah memecah pertanyaan kompleks dari pengguna menjadi daftar pertanyaan data yang sederhana dan spesifik.\n"
        "Konteksnya adalah data untuk 'Kabupaten Garut'.\n\n"
        "ATURAN:\n"
        "1.  Hanya fokus pada permintaan DATA. Abaikan sapaan atau basa-basi.\n"
        "2.  Jika ada rentang tahun (misal: \"2022 sampai 2024\"), pecah menjadi TIGA query terpisah (satu untuk 2022, 2023, dan 2024).\n"
        "3.  Jika ada DUA topik (misal: \"penduduk miskin DAN harga bawang\"), pecah menjadi DUA query terpisah.\n"
        "4.  Jika query sudah sederhana (misal: \"laju inflasi 2023\"), kembalikan sebagai daftar berisi satu item.\n\n"
        "Contoh 1:\n"
        "Pertanyaan: \"tampilkan data penduduk miskin Kabupaten Garut pada tahun 2022 sampai tahun 2024\"\n"
        "Jawaban JSON:\n"
        "[\"data penduduk miskin Kabupaten Garut tahun 2022\", \"data penduduk miskin Kabupaten Garut tahun 2023\", \"data penduduk miskin Kabupaten Garut tahun 2024\"]\n\n"
        "Contoh 2:\n"
        "Pertanyaan: \"hi, bisa bantu tampilkan data penduduk miskin 2022 dan juga harga bawang merah di garut\"\n"
        "Jawaban JSON:\n"
        "[\"data penduduk miskin 2022 di garut\", \"data harga bawang merah di garut\"]\n\n"
        "Sekarang, proses pertanyaan pengguna berikut:\n"
        f"Pertanyaan: \"{user_query}\"\n"
        "Jawaban JSON:"
    )
    
    try:
        response = generation_model.generate_content(
            prompt,
            generation_config={
                "response_mime_type": "application/json",
                "response_schema": schema
            }
        )
        block_reason = "BLOCK_REASON_UNSPECIFIED"
        if response.prompt_feedback:
            block_reason = response.prompt_feedback.block_reason.name
        if block_reason != "BLOCK_REASON_UNSPECIFIED":
            raise Exception(f"Dekomposisi diblokir: {block_reason}")
        result_json = json.loads(response.text)
        print(f"Hasil Dekomposisi: {result_json}")
        return result_json
    except Exception as e:
        print(f"Error saat dekomposisi query: {e}")
        return [user_query]


def handle_dataset_search(sub_query: str) -> dict:
    # (Fungsi ini tidak berubah)
    print(f"--- Memulai handle_dataset_search untuk sub-query: '{sub_query}' ---")
    
    if not dataset_collection:
        return {"status": "error", "query": sub_query, "error_message": "Error: Database dataset tidak dapat diakses."}
        
    try:
        # --- PERBAIKAN LOGIKA PENCARIAN (Bug "Padi") ---
        # 1. Gunakan sub-query LENGKAP untuk pencarian semantik.
        topic_query = sub_query
        
        print(f"[1] Membuat embedding untuk query LENGKAP: '{topic_query}'...")
        embedding_list = embedding_model.encode([topic_query]).tolist()
        
        print("[2] Menjalankan query ke ChromaDB...")
        results = dataset_collection.query(
            query_embeddings=embedding_list,
            n_results=5, # Kita hanya butuh beberapa hasil teratas
            include=["metadatas", "documents", "distances"]
        )
        # --- AKHIR PERBAIKAN LOGIKA PENCARIAN ---
        
        found_docs = results.get('documents')
        if not found_docs or not found_docs[0]:
            return {"status": "error", "query": sub_query, "error_message": f"Maaf, saya tidak menemukan dataset yang cocok untuk '{sub_query}'."}
        
        # 2. Filter Kandidat Relevan (Berdasarkan Topik)
        candidate_datasets = []
        distances = results['distances'][0]
        metadatas = results['metadatas'][0]
        documents = results['documents'][0]
        for i in range(len(distances)):
            if distances[i] < DISTANCE_THRESHOLD:
                candidate_datasets.append((metadatas[i], documents[i]))
        
        if not candidate_datasets:
            return {"status": "error", "query": sub_query, "error_message": f"Maaf, saya tidak dapat menemukan dataset yang cukup relevan untuk '{sub_query}'."}
        
        # 3. Ambil hasil teratas (paling relevan).
        final_dataset = candidate_datasets[0] 
        print(f"Mengambil hasil teratas yang relevan: {final_dataset[0].get('title')}")
        # --- AKHIR PERBAIKAN LOGIKA FILTER ---

        dataset_info, dataset_doc = final_dataset
        dataset_name = dataset_info.get('title', 'Tanpa Judul')
        dataset_url = dataset_info.get('url', '#')
        download_url = dataset_info.get('download_url')

        description_sentence = summarize_with_llm(dataset_doc)
        data_analysis_response = "_(Tidak ada pertanyaan spesifik untuk dianalisis.)_"
        table_preview = "_(Tidak ada link unduhan langsung untuk pratinjau.)_"
        
        if download_url and download_url != "-":
            full_df = load_full_dataframe_from_url(download_url)
            if full_df is not None:
                data_analysis_response = analyze_data_with_llm(full_df, sub_query)
                df_preview = full_df.head(5)
                df_preview = df_preview.applymap(lambda x: str(x)[:50] + '...' if isinstance(x, str) and len(x) > 50 else x)
                table_preview = f"```\n{tabulate(df_preview, headers='keys', tablefmt='github', showindex=False)}\n```"
            else:
                data_analysis_response = "_(Gagal memuat data dari link untuk dianalisis.)_"
                table_preview = "_(Gagal memuat pratinjau tabel.)_"
        
        return {
            "status": "success",
            "query": sub_query,
            "analysis_answer": data_analysis_response,
            "description_sentence": description_sentence,
            "table_preview": table_preview,
            "dataset_name": dataset_name,
            "dataset_url": dataset_url
        }

    except Exception as e:
        print(f"!!!!!!!!!!!!!!! KESALAHAN FATAL DI handle_dataset_search !!!!!!!!!!!!!!!")
        print(f"Error saat mencari data untuk '{sub_query}': {e}")
        traceback.print_exc() 
        return {"status": "error", "query": sub_query, "error_message": f"Maaf, terjadi kesalahan saat mencari data untuk '{sub_query}'."}

# --- 3. ENDPOINT API FLASK (Diperbarui) ---
@app.route("/api/chat", methods=["POST"])
def handle_chat():
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({"error": "Permintaan tidak valid, 'query' dibutuhkan."}), 400

    user_query = data['query']
    
    # --- PERUBAHAN: Alur baru dengan Koreksi Typo ---
    print(f"======== PERMINTAAN BARU ========")
    print(f"Query Asli: '{user_query}'")
    
    # Langkah 1: Koreksi Typo
    corrected_user_query = correct_query_with_llm(user_query)
    
    # Langkah 2: Preprocessing
    processed_query = preprocess_text(corrected_user_query)
    
    # Langkah 3: Klasifikasi Intent
    intent = classify_intent(processed_query, corrected_user_query)
    
    print(f"Query Terkoreksi: '{corrected_user_query}'")
    print(f"Intent Dideteksi: '{intent}'")

    response_data = {} 
    
    if intent == 'general_question':
        response_data = handle_general_question(corrected_user_query)
    
    elif intent == 'list_sectors':
        response_data = handle_list_sectors()
    
    elif intent == 'run_data_agent':
        try:
            # Langkah 4a: Dekomposisi Query
            sub_queries = decompose_query_with_llm(corrected_user_query)
            if not sub_queries:
                raise Exception("Gagal memecah query.")
            
            # Langkah 4b: Eksekusi setiap sub-query
            all_results = []
            for q in sub_queries:
                print(f"--- Mengeksekusi sub-query: {q} ---")
                result_dict = handle_dataset_search(q) 
                all_results.append(result_dict)
            
            successful_results = [res for res in all_results if res['status'] == 'success']
            failed_messages = [res['error_message'] for res in all_results if res['status'] == 'error']

            final_summary = ""
            if successful_results:
                individual_answers = [
                    f"Untuk pertanyaan '{res['query']}', jawabannya adalah: {res['analysis_answer']}" 
                    for res in successful_results
                ]
                
                # --- PERBAIKAN: Mengeluarkan '\n' dari f-string expression ---
                answers_list_string = "\n- ".join(individual_answers)
                
                # Langkah 4c: Membuat Ringkasan Akhir (dengan Konteks Garut)
                summary_prompt = (
                    "Anda adalah asisten AI 'Satu Data Garut'. Peran Anda adalah menjawab pertanyaan pengguna HANYA berdasarkan fakta-fakta yang ditemukan.\n\n"
                    "ATURAN PENTING:\n"
                    "1.  Semua fakta yang diberikan adalah tentang 'Kabupaten Garut'.\n"
                    "2.  JANGAN PERNAH menyebut 'Indonesia' atau negara/provinsi lain. Fokus hanya pada Garut.\n"
                    "3.  Buat ringkasan yang menggabungkan semua fakta menjadi satu paragraf yang mudah dibaca.\n\n"
                    f"Pertanyaan Asli Pengguna:\n\"{corrected_user_query}\"\n\n"
                    "Fakta-fakta yang Ditemukan (Semua tentang Garut):\n"
                    f"- {answers_list_string}\n\n"
                    "Ringkasan Jawaban Anda (satu paragraf, fokus hanya pada Garut):"
                )
                
                try:
                    print("Membuat ringkasan akhir...")
                    summary_response = generation_model.generate_content(summary_prompt)
                    final_summary = summary_response.text.strip()
                except Exception as e:
                    print(f"Gagal membuat ringkasan akhir: {e}")
                    final_summary = "Berikut adalah data yang berhasil saya temukan:"
            else:
                final_summary = "Maaf, saya tidak dapat menemukan data spesifik yang Anda minta."

            # Langkah 4d: Format Blok Data
            data_blocks = []
            for res in successful_results:
                
                # --- PERBAIKAN: Mengganti f-string multi-baris ---
                block = (
                    f"### {res['dataset_name']}\n\n"
                    f"**Ringkasan Deskripsi:**\n{res['description_sentence']}\n\n"
                    f"**Pratinjau Data (5 baris pertama):**\n{res['table_preview']}\n\n"
                    f"🔗 **[Lihat Sumber Data Asli]({res['dataset_url']})**"
                )
                data_blocks.append(block)
            
            # Langkah 4e: Gabungkan Semua
            combined_reply = final_summary 
            if data_blocks:
                # --- PERBAIKAN TYPO HTML ---
                combined_reply += "\n\n<hr class='my-4 border-gray-300'>\n\n" + "\n\n<hr class='my-4 border-gray-300'>\n\n".join(data_blocks)
            if failed_messages:
                combined_reply += "\n\n" + "\n".join(failed_messages)

            response_data = {'reply': combined_reply}
        
        except Exception as e:
            print(f"Error pada Data Agent: {e}")
            traceback.print_exc()
            response_data = {'reply': 'Maaf, terjadi kesalahan besar saat memproses permintaan data Anda.'}
    
    else:
        response_data = handle_general_question(corrected_user_query) # Default

    if 'reply' not in response_data:
        response_data['reply'] = "Maaf, terjadi kesalahan yang tidak terduga."

    print("======== RESPON SELESAI ========\n")
    return jsonify(response_data)

# --- 4. JALANKAN SERVER FLASK ---
if __name__ == "__main__":
    # Ganti 'chatbot.py' dengan nama file Anda jika berbeda
    app.run(host='0.0.0.0', port=5000, debug=True)