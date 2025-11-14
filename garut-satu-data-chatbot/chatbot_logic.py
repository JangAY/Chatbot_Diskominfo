# chatbot.py
import os
import re
import io
import json
import sys
import time
import traceback
from typing import Optional, List, Dict, Any, Tuple

import logging
import requests
import chromadb
import pandas as pd
from tabulate import tabulate
from sentence_transformers import SentenceTransformer
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import google.generativeai as genai

# -------------------------
# CONFIG LOGGING (debug-friendly)
# -------------------------
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger("chatbot")

# -------------------------
# LOAD ENV
# -------------------------
load_dotenv()

# -------------------------
# PREPROCESS (fallback)
# -------------------------
try:
    from preprocessing_utils import preprocess_text  # type: ignore
except Exception:
    log.debug("[INIT] preprocess_text not found, using simple fallback.")
    def preprocess_text(x: str) -> str:
        return x.strip()

# -------------------------
# INIT Gemini (LLM) + Embedding model
# -------------------------
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    log.error("GOOGLE_API_KEY not set in environment. Exiting.")
    sys.exit(1)

# Safety settings for Gemini
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]
try:
    genai.configure(api_key=api_key)
    log.info("[GEMINI] configured")
except Exception as e:
    log.exception("[GEMINI] configure failed: %s", e)
    # We'll still try to continue, but LLM calls will fail gracefully.

# Generation model wrapper (we'll check availability)
try:
    generation_model = genai.GenerativeModel(model_name="gemini-2.5-flash", safety_settings=safety_settings)
    log.info("[GEMINI] generation model created (gemini-2.5)")
except Exception as e:
    log.exception("[GEMINI] failed to init generation_model: %s", e)
    generation_model = None

# -------------------------
# HELPER: Run Gemini safely
# -------------------------
def run_gemini(prompt: str) -> str:
    """Jalankan prompt ke Gemini dan kembalikan hasil teksnya."""
    if generation_model is None:
        log.warning("[GEMINI] Model belum siap, gunakan fallback.")
        return "Maaf, layanan AI sedang tidak merespons."
    try:
        response = generation_model.generate_content(prompt)
        # response may be object with .text or dict
        if hasattr(response, "text"):
            return response.text.strip()
        elif isinstance(response, dict) and "text" in response:
            return response["text"].strip()
        else:
            log.warning("[GEMINI] Unexpected response type from generate_content: %s", type(response))
            return "Maaf, layanan AI sedang tidak merespons."
    except Exception as e:
        # important: log full exception server-side but return friendly message to client
        log.exception("[GEMINI] Error saat generate_content: %s", e)
        return "Maaf, layanan AI sedang tidak merespons."

# ======================================================
# EMBEDDING MODEL & CHROMADB INIT
# ======================================================
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
DB_PATH = os.path.join(os.path.dirname(__file__), "chatbot_db")

try:
    chroma_client = chromadb.PersistentClient(path=DB_PATH)
    dataset_collection = chroma_client.get_or_create_collection("dataset_embeddings")
    site_guide_collection = chroma_client.get_or_create_collection("site_guides")
    log.info("[CHROMADB] Initialized successfully.")
except Exception as e:
    log.exception("[CHROMADB] Initialization failed: %s", e)
    dataset_collection, site_guide_collection = None, None

# === Inisialisasi Chroma Collections ===
try:
    chroma_collections = {
        "dataset": chroma_client.get_or_create_collection("dataset_embeddings"),
        "site_guide": chroma_client.get_or_create_collection("site_guides")
    }
    logging.info("[CHROMA] Koleksi dataset & panduan situs berhasil dimuat.")
except Exception as e:
    logging.exception("[CHROMA] Gagal membuat koleksi: %s", e)
    chroma_collections = {}

# =====================================================
# Fungsi: search_dataset_embeddings
# =====================================================
def search_dataset_embeddings(query: str, n_results: int = 5):
    try:
        if "dataset" not in chroma_collections:
            chroma_collections["dataset"] = chroma_client.get_or_create_collection("dataset_embeddings")

        collection = chroma_collections["dataset"]
        query_vector = embedding_model.encode([query]).tolist()

        # Ambil juga documents, ini penting!
        result = collection.query(
            query_embeddings=query_vector,
            n_results=n_results,
            include=["metadatas", "documents", "distances"]
        )

        datasets = []
        metas = result.get("metadatas", [[]])[0]
        docs = result.get("documents", [[]])[0]
        dists = result.get("distances", [[]])[0]

        for i, meta in enumerate(metas):
            if not isinstance(meta, dict):
                meta = {"title": str(meta)}

            meta["_distance"] = dists[i]
            meta["_document"] = docs[i]
            datasets.append(meta)

        return datasets

    except Exception as e:
        logging.exception("[SEARCH_DATASET] error: %s", e)
        return []
    

# Embedding distance threshold (tuneable)
DISTANCE_THRESHOLD = 1.10

# -------------------------
# UTIL FUNCTIONS
# -------------------------
def _content_type_is_html(ct: Optional[str]) -> bool:
    if not ct:
        return False
    return "text/html" in ct.lower()

def _guess_ext_from_content_type(ct: Optional[str]) -> Optional[str]:
    if not ct:
        return None
    ct = ct.lower()
    if "excel" in ct or "spreadsheet" in ct:
        return ".xlsx"
    if "csv" in ct or "text/csv" in ct:
        return ".csv"
    if "json" in ct:
        return ".json"
    return None

def safe_print_df_info(df: Optional[pd.DataFrame], label: str = "DF"):
    if df is None:
        log.debug("[%s] None", label)
        return
    log.debug("[%s] shape=%s columns=%s", label, getattr(df, "shape", None), list(df.columns[:20]) if hasattr(df, "columns") else [])

# -------------------------
# LOAD DATAFRAME FROM URL (robust)
# -------------------------
def load_full_dataframe_from_url(url: str) -> Optional[pd.DataFrame]:
    """Muat CSV/XLSX/JSON dari URL. Jika link mengarah ke HTML -> return None."""
    if not url:
        return None
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=30, allow_redirects=True)
        resp.raise_for_status()
        ct = resp.headers.get("Content-Type", "")
        if _content_type_is_html(ct) and not url.lower().endswith((".csv", ".xls", ".xlsx", ".json")):
            log.warning("[DATA] URL appears to be HTML page, not direct file: %s (Content-Type=%s)", url, ct)
            return None
        ext = os.path.splitext(url.split("?")[0])[-1].lower()
        if not ext:
            guessed = _guess_ext_from_content_type(ct)
            ext = guessed or ""
        fb = io.BytesIO(resp.content)
        df = None
        if ext == ".csv":
            df = pd.read_csv(fb)
        elif ext in (".xls", ".xlsx"):
            df = pd.read_excel(fb)
        elif ext == ".json":
            try:
                df = pd.read_json(fb)
            except Exception:
                txt = resp.content.decode("utf-8", errors="ignore")
                obj = json.loads(txt)
                df = pd.json_normalize(obj)
        else:
            # try a few readers
            try:
                df = pd.read_excel(fb)
            except Exception:
                fb.seek(0)
                try:
                    df = pd.read_csv(fb)
                except Exception:
                    fb.seek(0)
                    try:
                        df = pd.read_json(fb)
                    except Exception:
                        log.warning("[DATA] failed to parse file at %s", url)
                        return None
        # normalize columns
        df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
        safe_print_df_info(df, "LoadedDF")
        return df
    except Exception as e:
        log.debug("[DATA] load error for %s: %s", url, e)
        return None

# -------------------------
# DETERMINISTIC ROW MATCHING
# -------------------------
def find_relevant_rows(df: pd.DataFrame, query: str) -> pd.DataFrame:
    """
    Filter DataFrame hanya mengambil baris yang mengandung keyword dari query user.
    (Misal: 'Bawang Merah', '2022', 'Kecamatan Garut Kota')
    """
    if df is None or df.empty:
        return pd.DataFrame()

    # 1. Preprocessing Query
    q_clean = query.lower()
    # Ambil tahun (misal 2022)
    years = re.findall(r"\b(20[1-2][0-9])\b", q_clean)
    
    # Ambil kata kunci penting (abaikan kata sambung umum)
    stop_words = ["berapa", "jumlah", "total", "data", "di", "pada", "tahun", "tampilkan", "list", "daftar", "harga", "dan", "dari", "yang", "kabupaten", "garut"]
    keywords = [w for w in re.split(r"\W+", q_clean) if w and w not in stop_words and not w.isdigit()]

    # Jika tidak ada keyword spesifik (misal user cuma tanya "Data apa ini?"), kembalikan head
    if not keywords and not years:
        return df.head(20)

    # 2. Masking (Pencarian)
    # Kita cari di semua kolom string
    mask_total = pd.Series(False, index=df.index)
    
    # Konversi seluruh dataframe ke string lowercase untuk pencarian mudah
    df_str = df.astype(str).apply(lambda x: x.str.lower())

    # Filter Tahun (Sangat Ketat)
    if years:
        year_mask = pd.Series(False, index=df.index)
        for y in years:
            # Cek di semua kolom apakah ada tahun tersebut
            for col in df_str.columns:
                year_mask |= df_str[col].str.contains(y, na=False)
        # Terapkan filter tahun
        df = df[year_mask]
        df_str = df_str[year_mask] # Update string df juga
        # Reset mask utama karena kita sudah filter df nya
        mask_total = pd.Series(False, index=df.index)

    # Jika setelah filter tahun data habis, return kosong
    if df.empty:
        return df

    # Filter Kata Kunci (Misal: 'bawang', 'merah')
    # Kita gunakan logika AND untuk frasa, atau OR untuk kata terpisah yang relevan
    # Untuk simpel & robust: Tiap keyword HARUS ada di setidaknya satu kolom di baris itu
    if keywords:
        # Gabungkan keywords jadi satu frase jika memungkinkan untuk akurasi lebih tinggi
        # Tapi di sini kita cek satu per satu: baris harus mengandung setidaknya satu keyword utama user
        # ATAU: Jika user tanya "Bawang Merah", baris harus contain "Bawang" DAN "Merah"? 
        # Pendekatan aman: Baris mengandung salah satu keyword user yang panjangnya > 3 huruf
        
        relevant_keyword_mask = pd.Series(False, index=df.index)
        for kw in keywords:
            if len(kw) < 3: continue 
            for col in df_str.columns:
                relevant_keyword_mask |= df_str[col].str.contains(kw, na=False)
        
        df = df[relevant_keyword_mask]

    return df

def find_date_column(df: pd.DataFrame) -> Optional[str]:
    date_like_cols = [c for c in df.columns if re.search(r"(tanggal|date|waktu|periode|bulan|tahun|tgl)", c, flags=re.IGNORECASE)]
    for c in date_like_cols:
        try:
            if pd.api.types.is_datetime64_any_dtype(df[c]):
                return c
        except Exception:
            continue
    for c in date_like_cols:
        try:
            df[c + "_parsed_temp"] = pd.to_datetime(df[c], errors="coerce")
            if df[c + "_parsed_temp"].notna().sum() > 0:
                df.drop(columns=[c + "_parsed_temp"], inplace=True, errors=True)
                return c
            df.drop(columns=[c + "_parsed_temp"], inplace=True, errors=True)
        except Exception:
            continue
    return date_like_cols[0] if date_like_cols else None

# -------------------------
# ANALYZE SUBSET WITH LLM (only if subset present)
# -------------------------
def analyze_data_with_llm(query: str, df: pd.DataFrame, dataset_title: str) -> str:
    try:
        # 1. Cari baris relevan (Filtering)
        filtered_df = find_relevant_rows(df, query)

        if filtered_df.empty:
            return "Maaf, saya menemukan dataset yang mungkin relevan ('" + dataset_title + "'), tetapi setelah mencari di dalamnya, tidak ditemukan data spesifik (tahun/item) yang Anda minta."

        # Batasi konteks ke LLM (maksimal 30 baris agar token tidak jebol, tapi cukup informatif)
        # Kita prioritaskan baris-baris hasil filter
        context_df = filtered_df.head(30)
        
        # Konversi ke Markdown string
        data_md = tabulate(context_df, headers='keys', tablefmt='github', showindex=False)

        prompt = f"""
        Anda adalah asisten analis data pemerintah.
        
        **Tugas:** Jawab pertanyaan user berdasarkan tabel data berikut.
        
        **Pertanyaan User:** "{query}"
        
        **Data Tabel (Filter dari dataset '{dataset_title}'):**
        {data_md}
        
        **Instruksi:**
        1. Jawab Langsung ke poinnya. Ambil angka spesifik dari tabel jika ada.
        2. Jika user bertanya "Berapa harga X", sebutkan angkanya dari tabel.
        3. Jika data menunjukkan beberapa tahun, jelaskan trennya singkat.
        4. JANGAN berhalusinasi. Jika data tidak ada di tabel di atas, katakan "Data spesifik tidak tersedia di tabel ini."
        5. Gunakan Bahasa Indonesia yang sopan dan formal.
        """

        response = run_gemini(prompt)
        return response.strip()

    except Exception as e:
        log.exception("[LLM_ANALYZER] Error: %s", e)
        return "Maaf, terjadi kesalahan saat menganalisis data tabel."
    
# -------------------------
# HELPER: summarize dataset doc
# -------------------------
def summarize_with_llm(text_to_summarize: str) -> str:
    if not text_to_summarize:
        return ""
    # use safe wrapper
    return run_gemini(f"Ringkas deskripsi dataset berikut dalam satu kalimat (Bahasa Indonesia):\n\n{text_to_summarize}")

def check_relevance_with_llm(user_query: str, dataset_title: str, columns: list) -> bool:
    """
    Meminta LLM menilai apakah judul dataset dan kolom-kolomnya relevan dengan pertanyaan user.
    Mencegah kasus: User tanya 'Penduduk Miskin', Database kasih 'Linmas'.
    """
    if not dataset_title: return False
    
    col_str = ", ".join(columns[:10]) # Ambil 10 nama kolom pertama sebagai sampel
    prompt = f"""
    Bertindaklah sebagai validator data yang ketat.
    Pertanyaan User: "{user_query}"
    
    Kandidat Dataset yang ditemukan sistem:
    Judul: "{dataset_title}"
    Kolom Tabel: [{col_str}]
    
    Apakah dataset ini BENAR-BENAR relevan untuk menjawab pertanyaan user?
    Jawab HANYA dengan "YA" atau "TIDAK".
    Jika ragu atau topiknya berbeda (misal user tanya kemiskinan tapi data tentang linmas), jawab TIDAK.
    """
    
    response = run_gemini(prompt).strip().upper()
    log.info(f"[RELEVANCE CHECK] Query: {user_query} | Dataset: {dataset_title} | Result: {response}")
    
    return "YA" in response

# -------------------------
# DECOMPOSE QUERY (LLM then fallback)
# -------------------------
def decompose_query_with_llm(user_query: str) -> List[str]:
    if generation_model is None:
        return decompose_query_fallback(user_query)
    schema = {"type": "ARRAY", "items": {"type": "STRING"}}
    prompt = (
        "Pecah pertanyaan kompleks menjadi daftar query data yang spesifik. "
        "Jika rentang tahun diberikan (mis 2022 sampai 2024) pecah per tahun.\n\n"
        f"Pertanyaan: \"{user_query}\"\n\nJawaban JSON array of strings:"
    )
    try:
        # use run_gemini but we need json; attempt LLM then fallback
        raw = run_gemini(prompt)
        # try to parse JSON from raw text
        try:
            arr = json.loads(raw)
            if isinstance(arr, list) and arr:
                return arr
        except Exception:
            log.debug("[DECOMP] gagal parse JSON dari LLM, fallback ke heuristic")
    except Exception as e:
        log.debug("[AGENT] decompose LLM failed: %s", e)
    return decompose_query_fallback(user_query)

def decompose_query_fallback(user_query: str) -> List[str]:
    text = user_query.strip().lower()
    parts = re.split(r"\s+dan\s+|,|\s+serta\s+|\s+&\s+", text)
    parts = [p.strip() for p in parts if p.strip()]
    expanded: List[str] = []
    for p in parts:
        m = re.search(r"(20[0-3]\d)\s*(?:sampai|-|to)\s*(20[0-3]\d)", p)
        if m:
            y1, y2 = int(m.group(1)), int(m.group(2))
            if y1 <= y2 and (y2 - y1) <= 10:
                for y in range(y1, y2 + 1):
                    expanded.append(re.sub(r"(20[0-3]\d)\s*(?:sampai|-|to)\s*(20[0-3]\d)", str(y), p))
            else:
                expanded.append(p)
        else:
            expanded.append(p)
    return expanded

def user_wants_preview(query: str) -> bool:
    keywords = ["tampilkan preview", "lihat datanya", "tampilkan tabel", "tampilkan datanya"]
    q = query.lower()
    return any(k in q for k in keywords)

# -------------------------
# CHROMADB QUERY FUNCTION (returns list of candidates metadata)
# -------------------------
def query_datasets_semantic(query: str, n_results: int = 6) -> List[Dict[str, Any]]:
    """Return list of metadata dicts (metadatas) with distance info if available."""
    if dataset_collection is None:
        return []
    try:
        emb = embedding_model.encode([query]).tolist()
        res = dataset_collection.query(query_embeddings=emb, n_results=n_results, include=["metadatas", "distances", "documents"])
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        docs = res.get("documents", [[]])[0]
        candidates = []
        for i, meta in enumerate(metas):
            if not isinstance(meta, dict):
                meta = {"title": str(meta)}
            cand = dict(meta)
            cand["_distance"] = float(dists[i]) if i < len(dists) else None
            cand["_document"] = docs[i] if i < len(docs) else None
            candidates.append(cand)
        return candidates
    except Exception as e:
        log.debug("[CHROMADB] query failed: %s", e)
        return []

# -------------------------
# HANDLE DATASET SEARCH (core)
# -------------------------
def handle_dataset_search(query: str, show_preview: bool = False):
    log.info("[DATA_AGENT] Mencari dataset untuk query: %s", query)
    
    # 1. PERUBAHAN: Ambil lebih banyak kandidat (misal 20) untuk menghindari duplikat yang memenuhi list
    candidates = search_dataset_embeddings(query, n_results=20)
    
    if not candidates:
        return {"status": "error", "error_message": f"Data tidak ditemukan untuk topik '{query}'."}

    # 2. PERUBAHAN: Filter Duplikat URL & Distance Threshold
    unique_candidates = []
    seen_urls = set()
    
    for cand in candidates:
        url = cand.get("download_url")
        dist = cand.get("_distance", 10)
        
        # Skip jika tidak ada URL, jarak terlalu jauh (>1.4), atau URL sudah pernah dicek
        if not url or dist > 1.4 or url in seen_urls:
            continue
            
        seen_urls.add(url)
        unique_candidates.append(cand)

    log.info(f"[DATA_AGENT] Menemukan {len(unique_candidates)} dataset unik untuk divalidasi.")

    # 3. Iterasi kandidat unik (Cek maksimal 5 kandidat teratas)
    best_df = None
    best_title = ""
    best_url = ""
    best_landing = ""
    found_relevance = False

    for cand in unique_candidates[:5]:
        title = cand.get("title") or cand.get("judul") or "Dataset Tanpa Judul"
        download_url = cand.get("download_url")
        
        # -- LOAD DATA SEMENTARA --
        # Kita hanya load header (sebagian kecil) dulu untuk efisiensi, tapi fungsi load_full.. memuat semua.
        # Tidak apa-apa, karena kita sudah deduplikasi.
        temp_df = load_full_dataframe_from_url(download_url)
        if temp_df is None or temp_df.empty:
            continue
        
        # -- VALIDASI RELEVANSI DENGAN LLM --
        is_relevant = check_relevance_with_llm(query, title, list(temp_df.columns))
        
        if is_relevant:
            best_df = temp_df
            best_title = title
            best_url = download_url
            best_landing = cand.get("landing_page") or download_url
            found_relevance = True
            log.info(f"[DATA_AGENT] DATASET DITEMUKAN & RELEVAN: {title}")
            break # Ketemu! Stop looping.
        else:
            log.info(f"[DATA_AGENT] Dataset '{title}' ditolak oleh validator.")
    
    # 4. Handle Jika Tidak Ada yang Relevan
    if not found_relevance or best_df is None:
        # PERMINTAAN ANDA: "Jika dataset tidak ada, jangan tampilkan tabel, tampilkan saja jawabannya"
        return {
            "status": "success",
            "reply": "Mohon maaf, berdasarkan pencarian saya di database Satu Data, saya belum menemukan dataset yang spesifik memuat informasi tersebut (Penduduk Miskin). Saat ini data yang tersedia di sekitar topik tersebut kurang relevan."
        }

    # 5. Analisis Data Terpilih
    analysis_result = analyze_data_with_llm(query, best_df, best_title)
    
    # Cek hasil analisis, jika analyzer bilang data kosong, jangan print tabel
    if "tidak ditemukan data spesifik" in analysis_result.lower() and len(analysis_result) < 200:
         return {
            "status": "success",
            "reply": analysis_result 
        }

    # 6. Siapkan Output Preview
    # Gunakan find_relevant_rows agar previewnya cerdas (hanya baris yang dicari user)
    filtered_preview_df = find_relevant_rows(best_df, query).head(5)
    
    # Fallback jika filter kosong tapi dataset relevan
    if filtered_preview_df.empty:
        filtered_preview_df = best_df.head(5)

    preview_md = ""
    if show_preview and not filtered_preview_df.empty:
        preview_md = tabulate(filtered_preview_df, headers='keys', tablefmt='github')

    response_text = f"**Sumber Data: {best_title}**\n\n{analysis_result}\n"
    if show_preview and preview_md:
        response_text += f"\n**Pratinjau Data Relevan:**\n{preview_md}\n\n[Lihat Dataset Lengkap]({best_landing})"
    else:
        # Jika user minta preview tapi datanya cuma 1 baris atau rangkuman, kasih link aja
        response_text += f"\n[Lihat Dataset Lengkap]({best_landing})"

    return {
        "status": "success",
        "reply": response_text,
        "dataset_title": best_title,
        "landing_page": best_landing
    }

# -------------------------
# HANDLE GENERAL QUESTION (site guide)
# -------------------------
def handle_general_question(query: str) -> dict:
    try:
        # Gunakan LLM untuk menjawab langsung
        prompt = (
            "Anda adalah asisten AI 'Satu Data Garut'. Jawab pertanyaan berikut dengan singkat, ramah, "
            "dan hanya berdasarkan informasi yang relevan jika tersedia. "
            "Jika data resmi tidak ada, beri tahu pengguna secara wajar.\n\n"
            f"PERTANYAAN: {query}\nJAWABAN:"
        )
        resp_text = run_gemini(prompt)
        return {"reply": resp_text}
    except Exception as e:
        log.exception("[GENERAL] error: %s", e)
        return {"reply": "Maaf, terjadi masalah saat mencoba menjawab pertanyaan Anda."}

# -------------------------
# LIST SECTORS
# -------------------------
def handle_list_sectors() -> dict:
    if dataset_collection is None:
        return {"reply": "Error: Database dataset tidak dapat diakses."}
    try:
        data = dataset_collection.get(include=["metadatas"])
        publishers = set()

        for md in data.get("metadatas", []):
            # debug print isi md
            log.debug("[LIST] md content: %s", md)

            # jika md adalah list atau dict nested
            if isinstance(md, dict) and "publisher" in md:
                publishers.add(md["publisher"])
            elif isinstance(md, list):
                for sub in md:
                    if isinstance(sub, dict) and "publisher" in sub:
                        publishers.add(sub["publisher"])

        if not publishers:
            return {"reply": "Maaf, saat ini tidak ada sektor yang terdaftar di database."}

        new_replies = [{"label": p, "value": f"Tampilkan dataset sektor {p}"} for p in sorted(list(publishers))]
        return {"reply": "Tentu, berikut daftar sektor (OPD) yang datanya tersedia.", "newQuickReplies": new_replies}

    except Exception as e:
        log.debug("[LIST] error: %s", e)
        return {"reply": "Maaf, terjadi kesalahan saat mengambil daftar sektor."}
    
# -------------------------
# SECTOR SEARCH
# -------------------------
def handle_sector_search(sector_name: str) -> dict:
    if dataset_collection is None:
        return {"reply": "Error: Database dataset tidak dapat diakses."}
    try:
        data = dataset_collection.get(include=["metadatas"])
        # Filter berdasarkan publisher, bukan title
        metas = [
            md for md in data.get("metadatas", [])
            if md.get("publisher", "").strip().lower() == sector_name.strip().lower()
        ]
        if not metas:
            return {"reply": f"Maaf, saya tidak menemukan dataset untuk sektor '{sector_name}'."}

        response_list = [f"Tentu, berikut beberapa dataset teratas untuk **{sector_name}**:"]
        for md in metas[:10]:
            title = md.get("title", "Tanpa Judul")
            url = md.get("landing_page") or md.get("download_url") or "#"
            response_list.append(f"* **{title}** - {url}")

        return {"reply": "\n".join(response_list)}
    except Exception as e:
        log.debug("[SECTOR] error: %s", e)
        return {"reply": f"Maaf, terjadi kesalahan saat mencari data untuk sektor '{sector_name}'."}
    
# -------------------------
# INTENT CLASSIFIER
# -------------------------
def classify_intent(processed_query: str, raw_query: str) -> str:
    raw_lower = raw_query.lower().strip()
    if raw_lower == "apa saja dataset yang tersedia?":
        return "list_sectors"
    if raw_lower.startswith("tampilkan dataset sektor"):
        return "dataset_sector_search"
    general_keywords = ["siapa kamu", "apa itu", "bagaimana cara", "jelaskan", "apa yang dimaksud"]
    if any(raw_lower.startswith(k) for k in general_keywords):
        return "general_question"
    return "run_data_agent"

# -------------------------
# MAIN ROUTE /api/chat
# -------------------------
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

@app.route("/api/health", methods=["GET"])
def health():
    """Simple health check for orchestrator / frontend."""
    return jsonify({"status": "ok", "llm_available": generation_model is not None}), 200

@app.route("/api/chat", methods=["POST"])
def handle_chat():
    try:
        req = request.get_json(silent=True)
        if not req or "query" not in req:
            return jsonify({"error": "Permintaan tidak valid, 'query' dibutuhkan."}), 400

        user_query = str(req["query"]).strip()
        log.info("[REQUEST] %s", user_query)

        
        # Preprocess & classify
        processed = preprocess_text(user_query)
        intent = classify_intent(processed, user_query)
        log.debug("[REQUEST] intent=%s", intent)

        show_preview = user_wants_preview(user_query)

        # ---- GENERAL QUESTION ----
        if intent == "general_question":
            out = handle_general_question(user_query)
            reply = out.get("reply", "Maaf, terjadi kesalahan.")
            # cache_set(user_query, reply)
            return jsonify({"reply": reply}), 200

        # ---- LIST SECTORS ----
        if intent == "list_sectors":
            out = handle_list_sectors()
            reply = out.get("reply", "Maaf, terjadi kesalahan.")
            # cache_set(user_query, reply)
            return jsonify(out), 200

        # ---- SECTOR SEARCH ----
        if intent == "dataset_sector_search":
            try:
                sector_name = user_query.split("Tampilkan dataset sektor", 1)[-1].strip()
                out = handle_sector_search(sector_name)
                reply = out.get("reply", "")
                # cache_set(user_query, reply)
                return jsonify(out), 200
            except Exception as e:
                log.exception("sector parse error: %s", e)
                return jsonify({"reply": "Maaf, terjadi kesalahan saat memproses permintaan sektor Anda."}), 200

        # ---- DATA AGENT ----
        if intent == "run_data_agent":
            subs = decompose_query_with_llm(user_query)
            if not subs:
                subs = [user_query]

            all_results = []
            for s in subs:
                try:
                    res = handle_dataset_search(s, show_preview=show_preview)
                    all_results.append(res)
                except Exception as e:
                    log.debug("[AGENT] error for sub %s: %s", s, e)
                    all_results.append({"status": "error", "query": s, "error_message": "Error internal saat pencarian."})

            successful = [r for r in all_results if r.get("status") == "success"]
            errors = [r.get("error_message") for r in all_results if r.get("status") == "error"]

            if successful:
                # Combine analysis answers
                pieces = []
                for r in successful:
                    pieces.append(f"**{r.get('dataset_title')}**\n{r.get('ai_analysis')}\n")
                    if show_preview and r.get("data_preview"):
                        pieces.append(f"Pratinjau data (5 baris pertama):\n{tabulate(pd.DataFrame(r['data_preview']), headers='keys', tablefmt='github')}\n")
                        landing = r.get("landing_page") or r.get("dataset_url")
                        if landing:
                            pieces.append(f"[Kunjungi halaman dataset]({landing})\n")
                final_reply = "\n".join(pieces).strip()
                # cache_set(user_query, final_reply)
                return jsonify({"reply": final_reply, "results": successful}), 200
            else:
                combined = "\n".join(errors) if errors else "Maaf, saya tidak menemukan data yang dimaksud."
                # cache_set(user_query, combined)
                return jsonify({"reply": combined, "results": []}), 200

        # ---- FALLBACK ----
        reply = "Maaf, saya belum bisa menjawab pertanyaan ini."
        # cache_set(user_query, reply)
        return jsonify({"reply": reply}), 200

    except Exception as e:
        log.exception("[HANDLE_CHAT] Unhandled exception: %s", e)
        return jsonify({"reply": "Maaf, layanan AI sedang tidak merespons."}), 200
    
# -------------------------
# RUN FLASK
# -------------------------
if __name__ == "__main__":
    log.info("Starting chatbot on http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=True)
