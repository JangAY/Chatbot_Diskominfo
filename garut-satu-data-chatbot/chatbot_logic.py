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
    Cari baris relevan menggunakan logika AND yang ketat untuk kata kunci dan tahun.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    q_lower = query.lower()
    
    # 1. Ekstrak Entitas dari Query
    # Ambil kata kunci (abaikan kata umum)
    stop_words = ["berapa", "jumlah", "total", "data", "di", "pada", "tahun", "tampilkan", "list", "daftar", "harga", "dan", "dari", "yang", "kabupaten", "garut"]
    keywords = [
        w for w in re.split(r"\W+", q_lower) 
        if w and w not in stop_words and not w.isdigit() and len(w) > 2
    ]
    # Tambahkan frasa penting secara manual
    if "penduduk miskin" in q_lower:
        keywords.append("penduduk miskin")
    if "bawang merah" in q_lower:
        keywords.append("bawang merah")
        
    keywords = list(set(keywords)) # Unik
    years = re.findall(r"\b(20[1-2][0-9])\b", q_lower) # cari tahun 2010-2029

    # Jika tidak ada keyword/tahun spesifik, kembalikan 10 baris pertama
    if not keywords and not years:
        return df.head(10)

    # 2. Siapkan DataFrame string untuk pencarian
    df_str = df.astype(str).apply(lambda x: x.str.lower())
    
    # 3. Terapkan Filter (Logika AND)
    
    # Mulai dengan semua baris dianggap benar
    final_mask = pd.Series(True, index=df.index)

    # 3a. Filter berdasarkan Kata Kunci
    if keywords:
        keyword_mask = pd.Series(False, index=df.index)
        for kw in keywords:
            # Baris harus mengandung SETIDAKNYA SATU keyword
            for col in df_str.columns:
                keyword_mask |= df_str[col].str.contains(kw, na=False, case=False)
        
        # Terapkan filter keyword
        final_mask &= keyword_mask

    # 3b. Filter berdasarkan Tahun
    if years:
        year_mask = pd.Series(False, index=df.index)
        for y in years:
            # Baris harus mengandung SETIDAKNYA SATU tahun yang diminta
            for col in df_str.columns:
                year_mask |= df_str[col].str.contains(y, na=False)
        
        # Terapkan filter tahun
        final_mask &= year_mask

    # 4. Kembalikan subset
    subset = df[final_mask]
    
    # Jika hasil filter AND kosong, jangan menyerah. Coba fallback ke keyword saja.
    if subset.empty and keywords and not years:
         return df[keyword_mask]
    if subset.empty and years and not keywords:
         return df[year_mask] # Ini yang terjadi di log Anda
         
    # Jika subset masih kosong (karena filter AND gagal),
    # kita harus menganggapnya TIDAK COCOK.
    # Namun, jika logika di atas (3a & 3b) sudah benar,
    # 'subset' akan kosong jika 'APK PAUD' (tidak mengandung 'penduduk miskin') diperiksa.
    
    log.debug(f"[find_relevant_rows] Query: '{query}'. Keywords: {keywords}, Years: {years}. Found {len(subset)} rows.")
    
    # Jika setelah filter AND hasilnya kosong, berarti memang tidak relevan.
    return subset

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
def analyze_data_with_llm(query: str, df: pd.DataFrame, dataset_title: str = "Data Gabungan") -> str:
    """Menganalisis DataFrame (bisa jadi gabungan) dengan LLM."""
    try:
        df = df.copy()
        df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]

        # Ambil max 30 baris (karena ini mungkin data gabungan)
        subset = df.head(30)
        # Jika barisnya > 30, beri tahu LLM bahwa ini sampel
        sample_info = f"(menampilkan {len(subset)} baris pertama)" if len(df) > 30 else ""
        
        subset_md = tabulate(subset, headers="keys", tablefmt="github")

        prompt = f"""
Anda adalah asisten data resmi Satu Data Garut.
Tugas Anda adalah menganalisis tabel data untuk menjawab pertanyaan user.

Pertanyaan User: {query}

Tabel Data (Dari dataset: '{dataset_title}') {sample_info}:
{subset_md}

Instruksi:
1. Jawab pertanyaan user secara langsung menggunakan angka dari tabel.
2. JANGAN mengarang data. Jika data tidak ada, katakan "Data tidak tersedia di tabel".
3. Jika tabel berisi beberapa tahun (misal 2022, 2023, 2024), jelaskan trennya (apakah naik, turun, atau stabil).
4. Jika user hanya meminta data (misal 'tampilkan data'), jawaban Anda adalah tabel itu sendiri (dalam format markdown).
"""
        resp = run_gemini(prompt)
        return resp

    except Exception as e:
        log.exception("[LLM_ANALYZER] Error: %s", e)
        return "Maaf, terjadi kesalahan dalam analisis data."

# -------------------------
# HELPER: summarize dataset doc
# -------------------------
def summarize_with_llm(text_to_summarize: str) -> str:
    if not text_to_summarize:
        return ""
    # use safe wrapper
    return run_gemini(f"Ringkas deskripsi dataset berikut dalam satu kalimat (Bahasa Indonesia):\n\n{text_to_summarize}")

# -------------------------
# DECOMPOSE QUERY (LLM then fallback)
# -------------------------
def decompose_query_with_llm(user_query: str) -> List[str]:
    """Pecah query menggunakan LLM, dengan fallback yang kuat."""
    if generation_model is None:
        return decompose_query_fallback(user_query)
    
    # Prompt yang lebih ketat untuk JSON
    prompt = f"""
    Pecah pertanyaan user menjadi daftar query data yang spesifik.
    1. Jika ada rentang tahun (misal "2022 sampai 2024"), pecah menjadi query per tahun ("...2022", "...2023", "...2024").
    2. Jika ada "dan" (misal "data A dan data B"), pecah menjadi query terpisah ("data A", "data B").
    3. Jika query sudah spesifik, kembalikan sebagai array satu item.

    Contoh:
    User: "data penduduk miskin 2022 sampai 2024"
    Output: ["data penduduk miskin 2022", "data penduduk miskin 2023", "data penduduk miskin 2024"]

    User: "data kemiskinan dan stunting 2023"
    Output: ["data kemiskinan 2023", "data stunting 2023"]

    User: "harga bawang merah hari ini"
    Output: ["harga bawang merah hari ini"]

    User: "{user_query}"
    Output:
    """
    
    try:
        raw = run_gemini(prompt)
        # Bersihkan markdown
        raw = raw.replace("```json", "").replace("```", "").strip()
        
        arr = json.loads(raw)
        if isinstance(arr, list) and arr:
            log.debug(f"[DECOMP LLM] Berhasil memecah: {arr}")
            return [str(item) for item in arr]
    except Exception as e:
        log.debug("[DECOMP] Gagal parse JSON dari LLM (%s), fallback ke heuristic", e)
        
    return decompose_query_fallback(user_query)

def decompose_query_fallback(user_query: str) -> List[str]:
    """Fallback pemecah query yang kuat."""
    text = user_query.strip().lower()
    
    # 1. Cek Rentang Tahun (Prioritas Utama)
    m = re.search(r"(20[0-3]\d)\s*(?:sampai|-|to)\s*(20[0-3]\d)", text)
    if m:
        y1, y2 = int(m.group(1)), int(m.group(2))
        if y1 <= y2 and (y2 - y1) <= 10: # Batas aman 10 tahun
            expanded = []
            # Ganti rentang tahun dengan placeholder unik
            base_query = re.sub(r"(20[0-3]\d)\s*(?:sampai|-|to)\s*(20[0-3]\d)", "TAHUN_PLACEHOLDER", text)
            for y in range(y1, y2 + 1):
                expanded.append(base_query.replace("TAHUN_PLACEHOLDER", str(y)))
            
            log.debug(f"[DECOMP FALLBACK] Range found. Split into: {expanded}")
            return expanded

    # 2. Cek Pemisah (dan, koma)
    parts = re.split(r"\s+dan\s+|,|\s+serta\s+|\s+&\s+", text)
    parts = [p.strip() for p in parts if p.strip()]
    
    if len(parts) > 1:
        log.debug(f"[DECOMP FALLBACK] Splitter found. Split into: {parts}")
        return parts

    # 3. Default: Kembalikan query asli
    log.debug("[DECOMP FALLBACK] No split. Returning original query.")
    return [user_query]

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
def handle_dataset_search(query: str) -> Dict[str, Any]:
    """
    HANYA mencari dataset, memvalidasi, dan mengembalikan data mentah.
    TIDAK melakukan analisis LLM.
    """
    log.info("[WORKER] Mencari dataset untuk sub-query: %s", query)

    try:
        candidates = search_dataset_embeddings(query, n_results=5)
        if not candidates:
            return {"status": "error", "query": query, "error_message": f"Tidak ada dataset ditemukan."}

        # Iterasi dan Validasi (Penting)
        best_subset = None
        chosen_dataset_meta = None

        for cand in candidates:
            title = cand.get("title") or cand.get("judul") or "Dataset"
            download_url = cand.get("download_url") or cand.get("file_url")
            dist = cand.get("_distance", 99.0)

            if not download_url or dist > 1.4:
                continue

            log.debug(f"[WORKER] Validating candidate: '{title}' (Dist: {dist:.4f})")
            df = load_full_dataframe_from_url(download_url)
            
            if df is None or df.empty:
                continue
                
            subset = find_relevant_rows(df, query)
            
            if not subset.empty:
                log.info(f"[WORKER] Match! '{title}' contains relevant data.")
                best_subset = subset
                chosen_dataset_meta = cand
                break 
            else:
                log.debug(f"[WORKER] No relevant rows found in '{title}'.")

        # Handle jika tidak ada yang cocok
        if best_subset is None or chosen_dataset_meta is None:
            return {
                "status": "error",
                "query": query,
                "error_message": f"Tidak ada data spesifik untuk '{query}' ditemukan di dataset yang relevan."
            }

        # Berhasil! Kembalikan data mentah untuk digabungkan oleh orchestrator
        return {
            "status": "success",
            "query": query,
            "metadata": chosen_dataset_meta,
            "subset_df": best_subset
        }

    except Exception as e:
        log.exception("[WORKER] Unhandled Error: %s", e)
        return {"status": "error", "query": query, "error_message": "Error internal saat memproses dataset."}

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

def user_wants_preview(query: str) -> bool:
    """Cek apakah user secara eksplisit meminta tabel/preview."""
    keywords = [
        "tampilkan preview", "lihat datanya", "tampilkan tabel", 
        "tampilkan datanya", "tabelnya", "previewnya", "datanya", "tabel"
    ]
    q = query.lower().strip()
    
    # Cek kata kunci eksplisit
    if any(k in q for k in keywords):
        return True
    
    # Cek jika query HANYA "tabel" atau "data"
    if q in ["data", "tabel"]:
        return True
        
    return False

# -------------------------
# INTENT CLASSIFIER
# -------------------------
def classify_intent(processed_query: str, raw_query: str) -> str:
    raw_lower = raw_query.lower().strip()
    
    # Intent Kontekstual (Memory)
    if raw_lower in ["tampilkan datanya", "tampilkan tabelnya", "tampilkan data", "tampilkan tabel", "data", "tabel"]:
        return "show_preview_context"
    
    # Intent Standar
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

# Sesi 'memory' sederhana untuk menyimpan data terakhir (HANYA UNTUK DEBUG/SINGLE-USER)
global_chat_memory = {}

@app.route("/api/health", methods=["GET"])
def health():
    """Simple health check for orchestrator / frontend."""
    return jsonify({"status": "ok", "llm_available": generation_model is not None}), 200

@app.route("/api/chat", methods=["POST"])
def handle_chat():
    global global_chat_memory # Akses memory global
    
    try:
        req = request.get_json(silent=True)
        if not req or "query" not in req:
            return jsonify({"error": "Permintaan tidak valid, 'query' dibutuhkan."}), 400

        user_query = str(req["query"]).strip()
        log.info("[REQUEST] %s", user_query)

        processed = preprocess_text(user_query)
        intent = classify_intent(processed, user_query)
        log.debug("[REQUEST] intent=%s", intent)

        # Panggil fungsi helper baru di awal
        show_preview = user_wants_preview(user_query)

        # ---- Tampilkan Preview (KONTEKS) ----
        # Ini menangani jika user bilang "tampilkan datanya" di chat KEDUA
        if intent == "show_preview_context" and not show_preview: # (jika bukan query eksplisit)
            last_result = global_chat_memory.get("last_result")
            if last_result and last_result.get("status") == "success":
                df = last_result.get("combined_df")
                title = last_result.get("combined_title")
                
                if df is None or df.empty:
                     return jsonify({"reply": "Maaf, saya tidak ingat data apa yang relevan."}), 200

                preview_md = tabulate(df.head(10), headers="keys", tablefmt="github")
                reply = f"Tentu, berikut pratinjau data (10 baris pertama) dari **{title}**:\n\n{preview_md}"
                
                return jsonify({"reply": reply}), 200
            else:
                return jsonify({"reply": "Maaf, data apa yang Anda maksud? Saya tidak memiliki konteks data sebelumnya."}), 200

        # ---- GENERAL QUESTION ----
        if intent == "general_question":
            out = handle_general_question(user_query)
            return jsonify(out), 200

        # ---- LIST SECTORS ----
        if intent == "list_sectors":
            out = handle_list_sectors()
            return jsonify(out), 200

        # ---- SECTOR SEARCH ----
        if intent == "dataset_sector_search":
            try:
                sector_name = user_query.split("Tampilkan dataset sektor", 1)[-1].strip()
                out = handle_sector_search(sector_name)
                return jsonify(out), 200
            except Exception as e:
                log.exception("sector parse error: %s", e)
                return jsonify({"reply": "Maaf, terjadi kesalahan saat memproses permintaan sektor Anda."}), 200

        # ---- DATA AGENT (OTAK ORCHESTRATOR BARU) ----
        # (Termasuk intent "show_preview_context" jika user *juga* minta data baru)
        if intent == "run_data_agent" or (intent == "show_preview_context" and show_preview):
            
            # --- LOGIKA GATEWAY BARU YANG LEBIH BAIK ---
            query_lower = user_query.lower()
            # Cek sinyal kompleks: (regex rentang tahun) ATAU (kata kunci pemisah)
            is_complex = re.search(r"(20[0-3]\d)\s*(?:sampai|-|to|hingga)\s*(20[0-3]\d)", query_lower)
            is_complex = is_complex or any(k in query_lower for k in [" dan ", ",", " & ", " serta "])

            if is_complex:
                log.debug("[AGENT] Query kompleks terdeteksi, memanggil decomposer...")
                subs = decompose_query_with_llm(user_query)
            else:
                log.debug("[AGENT] Query sederhana, skip decomposer.")
                subs = [user_query]
            # --- AKHIR LOGIKA GATEWAY ---

            if not subs: # Fallback jika decomposer (jika dipanggil) gagal
                subs = [user_query]

            all_dfs = []
            all_metas = []
            errors = []

            # 2. Loop dan Kumpulkan Data
            for s in subs:
                res = handle_dataset_search(s) # Panggil "worker"
                if res.get("status") == "success":
                    all_dfs.append(res["subset_df"])
                    all_metas.append(res["metadata"])
                else:
                    errors.append(res.get("error_message"))
            
            # 3. Proses Hasil
            if not all_dfs:
                # TIDAK ADA DATA DITEMUKAN (Ini menjawab permintaan Anda "jika dataset tidak ada")
                combined_errors = ". ".join(list(set(errors))) # Unik
                return jsonify({"reply": f"Maaf, saya tidak dapat menemukan data yang Anda maksud. (Detail: {combined_errors})", "results": []}), 200

            # 4. SUKSES! Gabungkan Data dan Analisis
            try:
                # Ini adalah kunci untuk "2022-2024" dan "bawang merah"
                combined_df = pd.concat(all_dfs, ignore_index=True)
                
                # Buat judul gabungan
                titles = list(set([m.get("title", "data") for m in all_metas]))
                combined_title = ", ".join(titles)
                
                # 5. Panggil LLM untuk Analisis GABUNGAN
                # LLM akan menganalisis tren jika datanya (combined_df) berisi 2022, 2023, 2024
                # LLM akan menganalisis "bawang" jika datanya (combined_df) hanya berisi "bawang"
                ai_analysis = analyze_data_with_llm(user_query, combined_df, combined_title)
                
                # 6. Buat Link Sumber (Selalu Tampil)
                links = list(set([m.get("landing_page") or m.get("download_url") for m in all_metas if m.get("landing_page") or m.get("download_url")]))
                links_md = "\n".join([f"- [Lihat Sumber Data]({l})" for l in links])

                # 7. Buat Pratinjau (Tampil Jika Diminta / Multi-data)
                preview_md = ""
                # Tampilkan preview jika (a) user memintanya ATAU (b) ini adalah gabungan multi-data
                if show_preview or len(all_dfs) > 1:
                    preview_df = combined_df.head(10) # Ambil 10 baris
                    preview_md = tabulate(preview_df, headers="keys", tablefmt="github")
                    preview_md = f"\n**Pratinjau Data Gabungan (10 Baris Pertama):**\n{preview_md}\n"

                # 8. Susun Jawaban Final
                final_reply = (
                    f"**Data Ditemukan: {combined_title}**\n\n"
                    f"**Analisis:**\n{ai_analysis}\n"
                    f"{preview_md}\n" # Selipkan preview di sini (akan kosong jika tidak di-trigger)
                    f"{links_md}"    # Tampilkan link di akhir
                ).strip()

                # 9. SIMPAN KE MEMORY (untuk konteks "tampilkan datanya" nanti)
                global_chat_memory["last_result"] = {
                    "status": "success",
                    "combined_df": combined_df,
                    "combined_title": combined_title
                }
                
                return jsonify({"reply": final_reply, "results": titles}), 200

            except Exception as e:
                log.exception("[AGENT] Gagal menggabungkan atau menganalisis data: %s", e)
                return jsonify({"reply": "Saya menemukan datanya, tetapi gagal menganalisisnya."}), 200


        # ---- FALLBACK ----
        reply = "Maaf, saya belum bisa menjawab pertanyaan ini."
        return jsonify({"reply": reply}), 200

    except Exception as e:
        log.exception("[HANDLE_CHAT] Unhandled exception: %s", e)
        return jsonify({"reply": "MaFungsi `user_wants_preview` (Baru)af, layanan AI sedang tidak merespons."}), 500
        
# -------------------------
# RUN FLASK
# -------------------------
if __name__ == "__main__":
    log.info("Starting chatbot on http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=True)
