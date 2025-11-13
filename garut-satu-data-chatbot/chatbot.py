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
# SIMPLE IN-MEMORY CACHE (TTL)
# -------------------------
# key -> { "answer": str, "ts": epoch_seconds }
CACHE_TTL_SECONDS = 24 * 3600  # 24 jam
cache_memory: Dict[str, Dict[str, Any]] = {}

def cache_get(key: str) -> Optional[str]:
    k = key.strip().lower()
    v = cache_memory.get(k)
    if not v:
        return None
    if time.time() - v.get("ts", 0) > CACHE_TTL_SECONDS:
        log.debug("[CACHE] key expired: %s", k)
        cache_memory.pop(k, None)
        return None
    log.debug("[CACHE] hit for key: %s", k)
    return v.get("answer")

def cache_set(key: str, answer: str) -> None:
    k = key.strip().lower()
    cache_memory[k] = {"answer": answer, "ts": time.time()}
    log.debug("[CACHE] saved key: %s", k)

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
            logging.warning("[CHROMA] Koleksi dataset belum dimuat, membuat ulang...")
            chroma_collections["dataset"] = chroma_client.get_or_create_collection("dataset_embeddings")

        collection = chroma_collections["dataset"]
        query_vector = embedding_model.encode([query]).tolist()
        result = collection.query(
            query_embeddings=query_vector,
            n_results=n_results,
            include=["metadatas", "distances"]
        )

        # Parsing hasil pencarian
        datasets = []
        for meta, dist in zip(result["metadatas"][0], result["distances"][0]):
            meta["_distance"] = dist
            datasets.append(meta)

        return datasets
    except Exception as e:
        logging.exception("[SEARCH_DATASET] Terjadi error: %s", e)
        return []

# Embedding distance threshold (tuneable)
DISTANCE_THRESHOLD = 0.45

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
    """Cari baris relevan deterministik berdasarkan kata kunci dan tahun."""
    if df is None or df.empty:
        return pd.DataFrame()
    q = str(query).lower()
    phrase_candidates = re.split(r"\s+dan\s+|,|\s+serta\s+|\s+&\s+", q)
    phrase_candidates = [p.strip() for p in phrase_candidates if p.strip()]
    keywords = []
    if "penduduk miskin" in q:
        keywords.append("penduduk miskin")
    if "bawang merah" in q:
        keywords.append("bawang merah")
    if "bawang putih" in q:
        keywords.append("bawang putih")
    keywords += phrase_candidates
    years = re.findall(r"(20[0-3]\d)", q)
    keywords = [k for k in list(dict.fromkeys([k.lower() for k in keywords if k]))]
    mask_total = pd.Series(False, index=df.index)
    string_cols = [c for c in df.columns if pd.api.types.is_string_dtype(df[c]) or df[c].dtype == object]
    for col in string_cols:
        col_series = df[col].fillna("").astype(str).str.lower()
        for kw in keywords:
            if len(kw) >= 2:
                try:
                    m = col_series.str.contains(re.escape(kw), na=False)
                except Exception:
                    m = col_series.str.contains(kw, na=False)
                mask_total = mask_total | m
    for y in years:
        for col in df.columns:
            try:
                if pd.api.types.is_integer_dtype(df[col]) or pd.api.types.is_float_dtype(df[col]):
                    mask_n = (df[col] == int(y))
                    mask_total = mask_total | mask_n.fillna(False)
                else:
                    col_series = df[col].astype(str).str.lower()
                    mask_y = col_series.str.contains(y, na=False)
                    mask_total = mask_total | mask_y
            except Exception:
                continue
    if not mask_total.any():
        candidate_colnames = [c for c in df.columns if re.search(r"(uraian|komoditas|nama|jenis|tahun|keterangan|harga|jumlah|penduduk|periode)", c, flags=re.IGNORECASE)]
        for col in candidate_colnames:
            col_series = df[col].fillna("").astype(str).str.lower()
            for kw in keywords:
                if len(kw) >= 2:
                    try:
                        m = col_series.str.contains(re.escape(kw), na=False)
                    except Exception:
                        m = col_series.str.contains(kw, na=False)
                    mask_total = mask_total | m
    try:
        subset = df[mask_total]
    except Exception:
        subset = pd.DataFrame()
    if subset.shape[0] > 5000:
        subset = subset.head(500)
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
    """
    Analisis data menggunakan LLM dengan pemrosesan awal untuk memahami isi dataset.
    Menghasilkan jawaban deskriptif singkat berbasis data.
    """
    try:
        log.info("[LLM_ANALYZER] Memulai analisis LLM untuk query: %s", query)
        df = df.copy()

        # Normalisasi nama kolom agar seragam
        df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

        # Deteksi kolom utama
        tahun_col = next((c for c in df.columns if "tahun" in c), None)
        jumlah_col = next((c for c in df.columns if "jumlah" in c or "total" in c), None)
        persentase_col = next((c for c in df.columns if "persen" in c or "%" in c), None)

        # Buat ringkasan statistik awal
        summary_parts = []
        if tahun_col:
            tahun_terbaru = df[tahun_col].max()
            tahun_terlama = df[tahun_col].min()
            summary_parts.append(f"Data mencakup tahun {tahun_terlama} hingga {tahun_terbaru}.")
        if jumlah_col:
            nilai_max = df[jumlah_col].max()
            nilai_min = df[jumlah_col].min()
            summary_parts.append(f"Nilai maksimum: {nilai_max:,}, minimum: {nilai_min:,}.")
        if persentase_col:
            rata_rata = df[persentase_col].mean()
            summary_parts.append(f"Rata-rata persentase: {rata_rata:.2f}%.")

        summary_text = " ".join(summary_parts) if summary_parts else "Tidak ditemukan kolom numerik utama dalam dataset."

        # Pilih subset kecil dari data untuk dikirim ke LLM (maks 10 baris)
        subset_df = df.head(10)
        subset_md = tabulate(subset_df, headers='keys', tablefmt='github')

        # Buat prompt LLM kontekstual
        prompt = f"""
Anda adalah asisten data Kabupaten Garut yang bertugas memberikan analisis singkat berdasarkan dataset resmi dari portal Satu Data Garut.
Jawablah pertanyaan pengguna berikut dengan bahasa natural, ringkas (maks 3 kalimat), dan sertakan angka penting bila ada.

**Pertanyaan pengguna:**
{query}

**Ringkasan dataset:**
{summary_text}

**Contoh isi data (maks 10 baris):**
{subset_md}

Tulis analisis singkat tentang apa yang ditunjukkan data di atas. termasuk tren, perbandingan antar tahun, dan implikasinya.
"""

        # Jalankan LLM (safe wrapper)
        llm_response = run_gemini(prompt)
        if not llm_response or "Maaf" in llm_response:
            log.warning("[LLM_ANALYZER] LLM tidak memberikan respons yang valid, fallback ke ringkasan statistik.")
            llm_response = f"Analisis sederhana: {summary_text}"

        log.info("[LLM_ANALYZER] Analisis selesai.")
        return llm_response.strip()

    except Exception as e:
        log.exception("[LLM_ANALYZER] Gagal menganalisis data: %s", e)
        # fallback ringkasan jika error
        try:
            preview_md = tabulate(df.head(3), headers='keys', tablefmt='github')
        except Exception:
            preview_md = "Data tidak dapat ditampilkan."
        return f"Terjadi kesalahan saat analisis. Berikut pratinjau data:\n{preview_md}"

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
        candidates = search_dataset_embeddings(query)
        if not candidates:
            return {
                "status": "error",
                "query": query,
                "error_message": f"Tidak ditemukan dataset relevan untuk '{query}'."
            }

        # Ambil dataset terdekat
        chosen = candidates[0]
        title = chosen.get("title", "Dataset tanpa judul")
        # Ambil landing_page jika ada, fallback ke download_url
        landing_page = chosen.get("landing_page") or chosen.get("download_url")

        df = load_full_dataframe_from_url(chosen.get("download_url"))
        if df is None or df.empty:
            return {
                "status": "error",
                "query": query,
                "error_message": f"Dataset '{title}' tidak memiliki data yang dapat dianalisis."
            }

        # Analisis konten jika relevan
        analysis = analyze_data_with_llm(query, df)

        # Tampilkan preview jika diminta
        preview_md = ""
        data_preview = []
        if show_preview:
            preview_df = df.head(5)
            data_preview = preview_df.to_dict(orient="records")
            preview_md = tabulate(preview_df, headers='keys', tablefmt='github')

        response_text = f"**{title}**\n"
        response_text += f"**Analisis:** {analysis}\n"
        if show_preview:
            response_text += f"\n**Pratinjau Data:**\n{preview_md}\n"
            response_text += f"Sumber: {landing_page}"

        return {
            "status": "success",
            "query": query,
            "dataset_title": title,
            "dataset_url": landing_page,  # <-- ganti di sini
            "landing_page": landing_page,
            "data_preview": data_preview,
            "ai_analysis": analysis,
            "response_text": response_text
        }

    except Exception as e:
        log.exception("[DATA_AGENT] Terjadi error: %s", e)
        return {"status": "error", "query": query, "error_message": "Terjadi error internal saat mencari dataset."}

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
            if isinstance(md, dict) and md.get("publisher"):
                publishers.add(md.get("publisher"))
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
        res = dataset_collection.get(where={"publisher": sector_name}, include=["metadatas"])
        metas = res.get("metadatas", [])
        if not metas:
            return {"reply": f"Maaf, saya tidak menemukan dataset untuk sektor '{sector_name}'."}
        response_list = [f"Tentu, berikut beberapa dataset teratas untuk **{sector_name}**:"]
        for md in metas[:10]:
            title = md.get("title", "Tanpa Judul")
            landing = md.get("landing_page") or md.get("download_url") or "#"
            response_list.append(f"* **{title}** - [halaman sumber data]({landing})")
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

        # Cache check
        cached = cache_get(user_query)
        if cached:
            return jsonify({"reply": cached, "cached": True}), 200

        # Preprocess & classify
        processed = preprocess_text(user_query)
        intent = classify_intent(processed, user_query)
        log.debug("[REQUEST] intent=%s", intent)

        show_preview = user_wants_preview(user_query)

        # ---- GENERAL QUESTION ----
        if intent == "general_question":
            out = handle_general_question(user_query)
            reply = out.get("reply", "Maaf, terjadi kesalahan.")
            cache_set(user_query, reply)
            return jsonify({"reply": reply}), 200

        # ---- LIST SECTORS ----
        if intent == "list_sectors":
            out = handle_list_sectors()
            reply = out.get("reply", "Maaf, terjadi kesalahan.")
            cache_set(user_query, reply)
            return jsonify(out), 200

        # ---- SECTOR SEARCH ----
        if intent == "dataset_sector_search":
            try:
                sector_name = user_query.split("Tampilkan dataset sektor", 1)[-1].strip()
                out = handle_sector_search(sector_name)
                reply = out.get("reply", "")
                cache_set(user_query, reply)
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
                cache_set(user_query, final_reply)
                return jsonify({"reply": final_reply, "results": successful}), 200
            else:
                combined = "\n".join(errors) if errors else "Maaf, saya tidak menemukan data yang dimaksud."
                cache_set(user_query, combined)
                return jsonify({"reply": combined, "results": []}), 200

        # ---- FALLBACK ----
        reply = "Maaf, saya belum bisa menjawab pertanyaan ini."
        cache_set(user_query, reply)
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
