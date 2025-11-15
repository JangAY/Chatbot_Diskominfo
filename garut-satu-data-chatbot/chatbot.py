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
def analyze_data_with_llm(query: str, df: pd.DataFrame) -> str:
    try:
        df = df.copy()
        df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]

        # Ambil max 10 baris saja (untuk diringkas)
        subset = df.head(10)
        subset_md = tabulate(subset, headers="keys", tablefmt="github")

        prompt = f"""
Anda adalah asisten data resmi Satu Data Garut.

JANGAN membuat data baru atau mengarang angka.
Jawaban Anda HARUS berdasarkan tabel berikut.

Pertanyaan: {query}

Tabel data relevan (maks 10 baris):
{subset_md}

Tolong berikan:
1. Jawaban langsung berdasarkan tabel.
2. Tanpa opini.
3. Tanpa data tambahan yang tidak ada di tabel.

Jika data yang diminta tidak ada dalam tabel, katakan apa adanya.
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

    try:
        # 1. Ambil Top 5 kandidat semantik (seperti debug_query)
        candidates = search_dataset_embeddings(query, n_results=5) # n_results=5 dari default fungsi

        if not candidates:
            return {
                "status": "error",
                "query": query,
                "error_message": f"Tidak ditemukan dataset yang relevan untuk: '{query}'."
            }

        # 2. Iterasi dan Validasi (LOGIKA BARU YANG PENTING)
        # Kita cari kandidat pertama yang *benar-benar* berisi data yang diminta.
        
        best_subset = None
        chosen_dataset = None

        for cand in candidates:
            title = cand.get("title") or cand.get("judul") or "Dataset"
            download_url = cand.get("download_url") or cand.get("file_url")
            dist = cand.get("_distance", 99.0)

            if not download_url:
                log.debug(f"[DATA_AGENT] Skipping '{title}' (no download_url)")
                continue

            # Jangan buang kandidat hanya karena jarak, kecuali jaraknya sangat jauh
            if dist > 1.4: # Threshold yang lebih longgar, biarkan konten yang menentukan
                log.debug(f"[DATA_AGENT] Skipping '{title}' (Distance {dist:.4f} > 1.4)")
                continue

            log.debug(f"[DATA_AGENT] Validating candidate: '{title}' (Dist: {dist:.4f})")

            # Unduh file untuk diinspeksi
            df = load_full_dataframe_from_url(download_url)
            
            if df is None or df.empty:
                log.debug(f"[DATA_AGENT] Skipping '{title}' (failed to load or empty)")
                continue
                
            # Validasi: Apakah file ini berisi baris yang kita cari?
            # (Misal: Apakah file 'Penduduk 2024' ini berisi '2022'?)
            subset = find_relevant_rows(df, query)
            
            if not subset.empty:
                # --- DITEMUKAN! ---
                # File ini adalah yang kita cari.
                log.info(f"[DATA_AGENT] Match! '{title}' (Dist: {dist:.4f}) contains relevant data.")
                best_subset = subset
                chosen_dataset = cand
                break # Hentikan iterasi, kita sudah punya pemenangnya
            else:
                # File ini mirip, tapi tidak berisi data spesifik yang diminta.
                log.debug(f"[DATA_AGENT] No relevant rows found in '{title}' for query.")

        # 3. Handle jika TIDAK ADA yang cocok setelah iterasi
        if best_subset is None or chosen_dataset is None:
            log.warning(f"[DATA_AGENT] No validated match for: '{query}'. Top semantic hit: '{candidates[0].get('title')}'")
            return {
                "status": "error",
                "query": query,
                "error_message": (
                    f"Saya menemukan beberapa dataset yang mirip (misalnya: '{candidates[0].get('title')}'), "
                    f"tetapi tidak ada yang berisi data spesifik untuk: '{query}'."
                )
            }

        # 4. PROSES PEMENANG
        # Kita sekarang punya `best_subset` (DataFrame) dan `chosen_dataset` (Metadata)
        
        title = chosen_dataset.get("title") or "Dataset"
        landing_page = chosen_dataset.get("landing_page") or chosen_dataset.get("download_url")

        # Analisis menggunakan LLM HANYA pada baris yang relevan
        ai_analysis = analyze_data_with_llm(query, best_subset)

        # Siapkan preview jika diminta
        data_preview = []
        preview_md = ""
        if show_preview:
            preview_df = best_subset.head(5) # Ambil 5 baris pertama dari subset
            data_preview = preview_df.to_dict(orient="records")
            preview_md = tabulate(preview_df, headers="keys", tablefmt="github")

        # 5. Susun Respon Final
        response_text = (
            f"**{title}**\n\n"
            f"**Analisis:**\n{ai_analysis}\n"
        )

        if show_preview and preview_md:
            response_text += f"\n**Pratinjau Data Relevan:**\n{preview_md}"
        
        if landing_page:
            response_text += f"\n\n[Lihat Dataset Lengkap]({landing_page})"

        return {
            "status": "success",
            "dataset_title": title,
            "ai_analysis": ai_analysis,
            "data_preview": data_preview,
            "response_text": response_text, # Kita kirim teks lengkap
            "landing_page": landing_page
        }

    except Exception as e:
        log.exception("[DATA_AGENT] Unhandled Error: %s", e)
        return {
            "status": "error",
            "query": query,
            "error_message": "Terjadi error internal saat memproses dataset."
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

# GANTI BLOK INI DI DALAM FUNGSI handle_chat (mulai baris 792)

        # ---- DATA AGENT ----
        if intent == "run_data_agent":
            subs = decompose_query_with_llm(user_query)
            if not subs:
                subs = [user_query]

            all_results = []
            for s in subs:
                try:
                    # 'show_preview' sudah diteruskan. 
                    # 'handle_dataset_search' sekarang akan mengurus logikanya.
                    res = handle_dataset_search(s, show_preview=show_preview)
                    all_results.append(res)
                except Exception as e:
                    log.debug("[AGENT] error for sub %s: %s", s, e)
                    all_results.append({"status": "error", "query": s, "error_message": "Error internal saat pencarian."})

            successful = [r for r in all_results if r.get("status") == "success"]
            errors = [r.get("error_message") for r in all_results if r.get("status") == "error"]

            if successful:
                # LOGIKA BARU: Cukup gabungkan 'response_text' yang sudah jadi
                # dari setiap hasil yang sukses.
                final_reply = "\n\n---\n\n".join([r.get("response_text", "") for r in successful]).strip()
                
                return jsonify({"reply": final_reply, "results": successful}), 200
            else:
                # Logika error masih sama
                combined = "\n".join(errors) if errors else "Maaf, saya tidak menemukan data yang dimaksud."
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
