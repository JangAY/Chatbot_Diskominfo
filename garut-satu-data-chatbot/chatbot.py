# chatbot.py
import os
import re
import io
import json
import sys
import time
import traceback
from typing import Optional, List, Dict, Any, Tuple

import difflib
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


# =====================================================
# TAMBAHAN: Synonym Dictionary (Kamus Sinonim)
# =====================================================
# Ini membantu mencocokkan "Miskin" dengan "Kemiskinan", "Harga" dengan "Tarif", dll.
# ... (Imports dan Setup Log sama)

# =====================================================
# MODIFIKASI: Synonym & Stopwords
# =====================================================
SYNONYMS = {
    "miskin": ["kemiskinan", "prasejahtera", "pkh", "bdt", "sosial"],
    "penduduk": ["warga", "masyarakat", "populasi", "orang", "kependudukan"],
    "harga": ["tarif", "biaya", "nilai"],
    "sekolah": ["pendidikan", "sd", "smp", "sma", "smk", "madrasah", "belajar"],
    "kesehatan": ["puskesmas", "rsud", "sakit", "medis", "nakes"],
    "kecamatan": ["wilayah", "daerah"],
    "stunting": ["gizi", "buruk", "balita", "tumbuh", "kembang"]
}

def expand_keywords(query_words: List[str]) -> set:
    expanded = set(query_words)
    for word in query_words:
        # Cek sebagai Key
        if word in SYNONYMS:
            expanded.update(SYNONYMS[word])
        # Cek sebagai Value (Reverse lookup sederhana)
        for key, val_list in SYNONYMS.items():
            if word in val_list:
                expanded.add(key)
    return expanded

def is_title_relevant(query: str, title: str) -> bool:
    q_lower = query.lower()
    t_lower = title.lower()
    
    # 1. UPDATE STOPWORDS (Tambahkan kata perintah)
    stopwords = [
        "tampilkan", "data", "jumlah", "list", "daftar", "di", "kabupaten", "garut", 
        "tahun", "dan", "yang", "berapa", "statistik", "rekapitulasi", "laporan", 
        "berikan", "minta", "cari", "tolong", "mohon", "tentang", "semua"
    ]
    
    query_words = [w for w in re.split(r"\W+", q_lower) if w and w not in stopwords and not w.isdigit() and len(w) > 2]
    
    if not query_words:
        return True 
    
    hits = 0
    title_words = set(re.split(r"\W+", t_lower))
    
    # Cek kecocokan kata
    for qw in query_words:
        match_found = False
        # 1. Cek Exact Match
        if qw in title_words:
            match_found = True
        # 2. Cek Synonym (Key to Value)
        elif qw in SYNONYMS:
            for syn in SYNONYMS[qw]:
                if syn in title_words:
                    match_found = True; break
        # 3. Cek Synonym (Value to Key - misal query 'kemiskinan' title 'miskin')
        if not match_found:
             for key, vals in SYNONYMS.items():
                 if qw in vals and key in title_words:
                     match_found = True; break

        if match_found: hits += 1
            
    # CRITICAL KEYWORDS CHECK
    CRITICAL_KEYWORDS = ["miskin", "kemiskinan", "stunting", "inflasi", "pdrb", "bawang", "cabe", "beras", "ipm"]
    for crit in CRITICAL_KEYWORDS:
        if crit in q_lower:
            # Pastikan judul mengandung crit ATAU sinonimnya
            has_crit = (crit in t_lower)
            if not has_crit:
                # Cek sinonim dari crit
                if crit in SYNONYMS:
                    for syn in SYNONYMS[crit]:
                        if syn in t_lower: has_crit = True; break
                # Cek jika crit adalah value dari sinonim lain (misal query 'kemiskinan', cek 'miskin')
                if not has_crit:
                    for key, vals in SYNONYMS.items():
                        if crit in vals and key in t_lower: has_crit = True; break
            
            if not has_crit:
                log.debug(f"[FILTER] REJECT '{title}' -> Missing critical keyword '{crit}'")
                return False

    return hits > 0

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
embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
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
    stop_words = ["berapa", "jumlah", "total", "data", "di", "pada", "tahun", "tampilkan", "list", "daftar", "harga", "dan", "dari", "yang", "kabupaten", "garut", "persentase", "presentase"]
    keywords = [
        w for w in re.split(r"\W+", q_lower) 
        if w and w not in stop_words and not w.isdigit() and len(w) > 2
    ]
    if "penduduk miskin" in q_lower: keywords.append("penduduk miskin")
    keywords = list(set(keywords))
    years = re.findall(r"\b(20[1-2][0-9])\b", q_lower)

    if not keywords and not years:
        return df.head(10)

    df_str = df.astype(str).apply(lambda x: x.str.lower())
    final_mask = pd.Series(True, index=df.index)

    if keywords:
        keyword_mask = pd.Series(False, index=df.index)
        for kw in keywords:
            for col in df_str.columns:
                keyword_mask |= df_str[col].str.contains(kw, na=False, case=False)
        final_mask &= keyword_mask

    if years:
        year_mask = pd.Series(False, index=df.index)
        for y in years:
            for col in df_str.columns:
                year_mask |= df_str[col].str.contains(y, na=False)
        final_mask &= year_mask

    subset = df[final_mask]
    
    # Fallback logic (jika filter AND terlalu ketat)
    if subset.empty and keywords and not years:
         return df[keyword_mask]
    if subset.empty and years and not keywords:
         return df[year_mask]
         
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
Berikan jawaban analisis satu paragraf

Pertanyaan: {query}

Tabel data relevan (maks 10 baris):
{subset_md}

Tolong berikan:
1. Data berdasarkan tabel.
2. Tanpa data tambahan yang tidak ada di tabel.

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

# =====================================================
# HELPERS: SCORING & RANKING (ANTI TYPO & ROBUST)
# =====================================================
def calculate_smart_score(query: str, title: str) -> float:
    """
    Menghitung skor dengan toleransi TYPO.
    Jika user mengetik 'miskn', sistem tetap menganggap itu 'miskin' 
    dan menerapkan filter topik kritis.
    """
    q_lower = query.lower()
    t_lower = title.lower()
    score = 0.0
    
    # Pecah query menjadi kata-kata (token)
    q_tokens = [w for w in re.split(r"\W+", q_lower) if len(w) > 2]
    t_tokens = set(re.split(r"\W+", t_lower))
    
    # 1. Base Score (Fuzzy Match)
    match_count = 0
    for qw in q_tokens:
        # Cari kata di judul yang mirip dengan kata di query (cutoff 0.8 = 80% mirip)
        # Ini menangani 'penddk' -> 'penduduk'
        matches = difflib.get_close_matches(qw, t_tokens, n=1, cutoff=0.8)
        if matches:
            match_count += 1
            score += 1.0 
    
    # Normalisasi score
    if len(q_tokens) > 0:
        score = (score / len(q_tokens)) * 2.0

    # 2. CRITICAL SUBJECT FILTER (WAJIB ADA)
    # Daftar kata yang jika diminta, harus ada di judul (atau sinonimnya)
    CRITICAL_SUBJECTS = [
        "miskin", "kemiskinan", "stunting", "disabilitas", "inflasi", "pdrb", 
        "wisata", "sekolah", "puskesmas", "bawang", "cabe", "beras", "jagung",
        "penduduk", "warga", "korban", "bencana"
    ]
    
    # Cek intent user (walaupun typo)
    detected_intent = []
    for subject in CRITICAL_SUBJECTS:
        # Apakah query mengandung kata yang mirip subject?
        if subject in q_lower or difflib.get_close_matches(subject, q_tokens, n=1, cutoff=0.8):
            detected_intent.append(subject)

    # Terapkan Hukuman
    for subject in detected_intent:
        found_in_title = False
        
        # Cek di Title (Exact, Synonym, atau Typo)
        if subject in t_lower: 
            found_in_title = True
        elif subject in SYNONYMS: # Cek sinonim
            for syn in SYNONYMS[subject]:
                if syn in t_lower: found_in_title = True; break
        
        if not found_in_title: # Cek reverse sinonim
            for key, vals in SYNONYMS.items():
                if subject in vals and key in t_lower: found_in_title = True; break

        if not found_in_title: # Cek fuzzy di title
             if difflib.get_close_matches(subject, t_tokens, n=1, cutoff=0.85):
                 found_in_title = True

        if not found_in_title:
            score -= 10.0 # HUKUMAN: Intent terdeteksi tapi judul tidak cocok

    # 3. CONFLICT PENALTIES (Jumlah vs Persentase)
    has_jumlah_query = "jumlah" in q_lower or difflib.get_close_matches("jumlah", q_tokens, n=1, cutoff=0.85)
    has_persen_query = "persentase" in q_lower or "presentase" in q_lower
    
    has_jumlah_title = "jumlah" in t_lower
    has_persen_title = "persentase" in t_lower

    if has_jumlah_query and has_persen_title and not has_jumlah_title:
        score -= 5.0
    if has_persen_query and has_jumlah_title and not has_persen_title:
        score -= 5.0

    # 4. EXACT PHRASE BOOSTER
    if "penduduk miskin" in q_lower and "penduduk miskin" in t_lower:
        score += 3.0

    return score


# =====================================================
# MODIFIKASI: handle_dataset_search
# =====================================================
def handle_dataset_search(query: str, show_preview: bool = False):
    log.info("[DATA_AGENT] Mencari dataset untuk query: %s", query)

    # 1. Deteksi Tahun
    year_match = re.search(r"\b(20[1-2][0-9])\b", query)
    target_year = year_match.group(1) if year_match else None
    has_year = bool(target_year)
    
    # Ambil kandidat
    candidates = search_dataset_embeddings(query, n_results=30)

    if not candidates:
        return {"status": "not_found"}

    potential_matches = []
    seen_titles = set()

    # 2. FILTERING AWAL
    for cand in candidates:
        title = cand.get("title") or cand.get("judul") or "Dataset"
        download_url = cand.get("download_url")
        dist = cand.get("_distance", 99.0)

        # Threshold longgar
        if dist > 25.0: continue 
        if not download_url: continue
        
        # Validasi Topik Dasar
        if not is_title_relevant(query, title):
            continue 

        # Filter Tahun
        if has_year:
            if target_year in title:
                potential_matches.append(cand)
        else:
            if title not in seen_titles:
                potential_matches.append(cand)
                seen_titles.add(title)

    if not potential_matches:
        if has_year:
             # Fallback cek isi file jika judul tidak ada tahun
             potential_matches = candidates[:5]
        else:
             return {"status": "not_found"}

    # 3. SMART RERANKING
    scored_candidates = []
    for cand in potential_matches:
        title = cand.get("title", "")
        # Gunakan fungsi GLOBAL calculate_smart_score
        smart_score = calculate_smart_score(query, title)
        cand["_smart_score"] = smart_score
        scored_candidates.append(cand)

    # Urutkan Score Tertinggi
    scored_candidates.sort(key=lambda x: x["_smart_score"], reverse=True)

    # Cek kualitas kandidat terbaik
    if not scored_candidates or scored_candidates[0]["_smart_score"] < 0:
        return {"status": "not_found"}

    # ===============================================================
    # PERBAIKAN UTAMA DI SINI:
    # Jika tidak ada tahun (Query Umum), LANGSUNG return List Mode.
    # Jangan mencoba load file satu per satu.
    # ===============================================================
    if not has_year:
         options = []
         list_text = "Saya menemukan beberapa dataset yang relevan:\n"
         used_titles = set()
         
         # Hanya ambil yang skor positif (> 0)
         valid_options = [c for c in scored_candidates if c["_smart_score"] > 0]
         
         if not valid_options:
             return {"status": "not_found"}

         for c in valid_options:
             t = c.get("title", "").strip()
             if t in used_titles: continue
             used_titles.add(t)
             
             # Buat Value tombol
             options.append({"label": t, "value": f"Tampilkan data {t}"})
             list_text += f"\n* **{t}**"
             
             if len(options) >= 5: break
         
         return {
             "status": "multiple_options", 
             "response_text": list_text, 
             "options": options
         }

    # 4. LOGIC KHUSUS (Jika User Minta Tahun Spesifik)
    final_result = None
    
    for match_cand in scored_candidates[:3]:
        # Skip jika skor minus
        if match_cand["_smart_score"] < 0: continue

        title = match_cand.get("title")
        download_url = match_cand.get("download_url")
        
        df = load_full_dataframe_from_url(download_url)
        subset = find_relevant_rows(df, query)
        
        if df is not None and not subset.empty:
            final_result = {
                "candidate": match_cand,
                "subset": subset,
                "df": df
            }
            break 

    if final_result:
        cand = final_result["candidate"]
        subset = final_result["subset"]
        title = cand.get("title")
        landing_page = cand.get("landing_page") or cand.get("download_url")
        
        ai_analysis = analyze_data_with_llm(query, subset)
        
        preview_md = ""
        if show_preview:
            preview_df = subset.head(5)
            preview_md = tabulate(preview_df, headers="keys", tablefmt="github")
            
        response_text = f"**{title}**\n\n{ai_analysis}\n"
        if show_preview and preview_md:
            response_text += f"\n**Pratinjau Data:**\n{preview_md}"
        if landing_page:
            response_text += f"\n\n[Lihat Sumber Data]({landing_page})"
            
        return { "status": "success", "response_text": response_text }

    return {"status": "not_found"}

# -------------------------
# HANDLE GENERAL QUESTION (site guide)
# -------------------------
def handle_general_question(query: str, context: str = "") -> dict:
    try:
        # Prompt diperbarui: Lebih cerdas membedakan konteks
        prompt = f"""
Anda adalah Asisten AI untuk Portal "Satu Data Garut".

Tugas Anda adalah menjawab pertanyaan pengguna yang TIDAK ditemukan di database dataset statistik.

Aturan Menjawab:
1. **JIKA PERTANYAAN UMUM / PENGETAHUAN (General Knowledge):**
   (Contoh: "Kabupaten Tasikmalaya", "Apa itu Stunting", "Resep Nasi Goreng", "Lokasi Garut")
   - Jawablah secara LANGSUNG, informatif, dan membantu.
   - **JANGAN** meminta maaf.
   - **JANGAN** menyebutkan bahwa "dataset tidak ditemukan".
   - Anggaplah Anda sedang mengobrol biasa.

2. **JIKA PERTANYAAN MEMINTA DATA STATISTIK/ANGKA SPESIFIK:**
   (Contoh: "Berapa jumlah penduduk Tasikmalaya 2024?", "Data inflasi bulan ini")
   - Karena Anda tidak menemukan datanya di database internal, katakan dengan sopan bahwa "Data spesifik untuk ini belum tersedia di Satu Data Garut".
   - Lalu berikan informasi umum atau saran untuk mengecek ke instansi terkait (misal BPS).

Pertanyaan Pengguna: "{query}"

Jawaban Anda:
"""
        resp_text = run_gemini(prompt)
        return {"reply": resp_text}
    except Exception as e:
        log.exception("[GENERAL] error: %s", e)
        return {"reply": "Maaf, layanan AI sedang sibuk."}
# -------------------------
# HELPER: Get All Sectors (Publishers)
# -------------------------
def get_all_publishers() -> List[str]:
    """Mengambil daftar unik nama Publisher/Dinas dari ChromaDB."""
    if dataset_collection is None:
        return []
    try:
        # Ambil metadata saja untuk efisiensi
        data = dataset_collection.get(include=["metadatas"])
        publishers = set()
        
        for md in data.get("metadatas", []):
            if isinstance(md, dict) and "publisher" in md:
                p = md["publisher"]
                if p: publishers.add(p.strip())
            elif isinstance(md, list): # Jaga-jaga nested list
                for sub in md:
                    if isinstance(sub, dict) and "publisher" in sub:
                         p = sub["publisher"]
                         if p: publishers.add(p.strip())
        
        return list(publishers)
    except Exception as e:
        log.error(f"[GET_PUBLISHERS] Error: {e}")
        return []
              
# -------------------------
# LIST SECTORS
# -------------------------
def handle_list_sectors() -> dict:
    log.info("[LIST] Memproses permintaan daftar sektor...")
    if dataset_collection is None:
        return {"reply": "Error: Database dataset tidak dapat diakses."}
    try:
        data = dataset_collection.get(include=["metadatas"])
        publishers = set()

        metas = data.get("metadatas", [])
        # Jaga-jaga jika metas None
        if not metas: 
            metas = []

        for md in metas:
            # Kadang metadata bisa None atau list jika database korup/kosong
            if isinstance(md, dict) and "publisher" in md:
                p = md["publisher"]
                if p: publishers.add(p.strip())
            elif isinstance(md, list):
                for sub in md:
                    if isinstance(sub, dict) and "publisher" in sub:
                         p = sub["publisher"]
                         if p: publishers.add(p.strip())

        if not publishers:
            log.warning("[LIST] Tidak ada publisher ditemukan di metadata.")
            return {"reply": "Maaf, saat ini tidak ada sektor yang terdaftar di database."}

        # Urutkan secara alfabet
        sorted_pubs = sorted(list(publishers))
        
        # Buat Quick Replies (Tombol)
        new_replies = [{"label": p, "value": f"Tampilkan dataset sektor {p}"} for p in sorted_pubs]
        
        # Buat Text List
        list_text = "Tentu, berikut daftar sektor (OPD) yang datanya tersedia:\n"
        for p in sorted_pubs:
            list_text += f"\n* {p}"

        return {"reply": list_text, "newQuickReplies": new_replies}

    except Exception as e:
        log.exception("[LIST] error: %s", e)
        return {"reply": "Maaf, terjadi kesalahan saat mengambil daftar sektor."}
            
# -------------------------
# SECTOR SEARCH
# -------------------------
def handle_sector_search(sector_name: str) -> dict:
    if dataset_collection is None:
        return {"reply": "Error: Database dataset tidak dapat diakses."}
    try:
        data = dataset_collection.get(include=["metadatas"])
        
        # 1. Filter Metadata berdasarkan Nama Publisher/Sektor
        metas = []
        all_metas = data.get("metadatas", [])
        
        for md in all_metas:
            pub = md.get("publisher", "").strip()
            # Cek exact match atau contains
            if sector_name.lower() in pub.lower():
                metas.append(md)

        if not metas:
            return {"reply": f"Maaf, saya tidak menemukan dataset untuk sektor '{sector_name}'."}

        # 2. Urutkan (Opsional: berdasarkan tahun terbaru jika ada)
        # Kita coba sort by title dulu agar rapi
        metas.sort(key=lambda x: x.get("title", ""))

        # 3. Buat Clickable Options
        options = []
        list_text = f"Berikut adalah dataset yang tersedia dari **{sector_name}**. Silakan pilih data untuk ditampilkan:\n"
        
        # Batasi 10 dataset agar tidak terlalu panjang
        for md in metas[:10]:
            title = md.get("title", "Tanpa Judul")
            
            # Value ini akan dikirim balik ke chatbot seolah user mengetiknya
            trigger_query = f"Tampilkan data {title}"
            
            options.append({
                "label": title,
                "value": trigger_query
            })
            list_text += f"\n* **{title}**"

        # Return format yang mendukung opsi
        return {
            "status": "multiple_options",
            "response_text": list_text,
            "options": options
        }

    except Exception as e:
        log.debug("[SECTOR] error: %s", e)
        return {"reply": f"Maaf, terjadi kesalahan saat mencari data untuk sektor '{sector_name}'."}
        
# -------------------------
# INTENT CLASSIFIER
# -------------------------
def classify_intent(processed_query: str, raw_query: str) -> str:
    raw_lower = raw_query.lower().strip()
    
    # ---------------------------------------------------------
    # PRIORITAS 1: TOMBOL & PERINTAH BAKU (HARUS MENANG DULUAN)
    # ---------------------------------------------------------
    
    # 1. Cek Tombol "Apa saja dataset yang tersedia?"
    if raw_lower == "apa saja dataset yang tersedia?" or raw_lower == "apa saja dataset yang tersedia":
        return "list_sectors"

    # 2. Cek Tombol Sektor "Tampilkan dataset sektor X"
    if raw_lower.startswith("tampilkan dataset sektor"):
        return "dataset_sector_search"

    # 3. Cek Keyword Manual untuk List Sektor
    # Menangkap: "list sektor", "daftar dinas", "tampilkan opd"
    sector_keywords = ["sektor", "dinas", "opd", "skpd", "instansi"]
    ask_keywords = ["apa saja", "list", "daftar", "tampilkan", "sebutkan", "tersedia", "ada", "lihat", "menu"]
    
    has_sector = any(k in raw_lower for k in sector_keywords)
    has_ask = any(k in raw_lower for k in ask_keywords)
    
    # Trigger kata tunggal/pendek
    if raw_lower in ["sektor", "list sektor", "daftar sektor", "opd", "dinas", "menu", "list"]:
        return "list_sectors"
    
    # Trigger kombinasi
    if has_sector and has_ask:
        return "list_sectors"

    # ---------------------------------------------------------
    # PRIORITAS 2: DIRECT PUBLISHER SEARCH (NAMING CHECK)
    # ---------------------------------------------------------
    # Logika ini ditaruh SETELAH List Sektor agar tidak salah tangkap.
    
    known_publishers = get_all_publishers()
    
    # Urutkan dari nama terpanjang agar "Dinas Kesehatan" terdeteksi sebelum "Dinas"
    known_publishers.sort(key=len, reverse=True)
    
    for pub in known_publishers:
        # Filter: Jangan trigger jika nama publisher terlalu pendek (misal singkatan < 3 huruf)
        if len(pub) < 3: continue
        
        # Cek apakah nama publisher ada di query
        if pub.lower() in raw_lower:
            return "dataset_sector_search_direct"

    # ---------------------------------------------------------
    # PRIORITAS 3: GENERAL QUESTION & DATA AGENT
    # ---------------------------------------------------------
    general_keywords = ["siapa kamu", "apa itu", "bagaimana cara", "jelaskan", "apa yang dimaksud", "selamat", "halo", "hai", "pagi", "siang", "sore", "malam", "terima kasih"]
    
    if any(raw_lower.startswith(k) for k in general_keywords):
        return "general_question"
        
    # Default: Anggap user sedang mencari data spesifik
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

        # ==========================================================
        # 1. LIST SECTORS
        # ==========================================================
        if intent == "list_sectors":
            out = handle_list_sectors()
            return jsonify(out), 200

        # ==========================================================
        # 2. SECTOR SEARCH - DIRECT MENTION (Misal: "Kecamatan Cilawu")
        # ==========================================================
        if intent == "dataset_sector_search_direct":
            try:
                known_publishers = get_all_publishers()
                target_sector = None
                known_publishers.sort(key=len, reverse=True)
                
                for pub in known_publishers:
                    if pub.lower() in user_query.lower():
                        target_sector = pub
                        break
                
                if target_sector:
                    log.info(f"[DIRECT SECTOR] Found: {target_sector}")
                    out = handle_sector_search(target_sector)
                    
                    # --- PERUBAHAN DI SINI ---
                    # Petakan 'response_text' ke 'reply' dan 'options' ke 'related_queries'
                    return jsonify({
                        "reply": out.get("response_text", out.get("reply", "")),
                        "related_queries": out.get("options", [])
                    }), 200
                else:
                    return jsonify({"reply": "Maaf, saya mendeteksi nama dinas tapi tidak menemukannya di database."}), 200
            except Exception as e:
                log.exception("direct sector error: %s", e)
                return jsonify({"reply": "Terjadi kesalahan sistem."}), 200

        # ==========================================================
        # 3. SECTOR SEARCH - BUTTON CLICK (Format Baku)
        # ==========================================================
        if intent == "dataset_sector_search":
            try:
                # Format: "Tampilkan dataset sektor [Nama Sektor]"
                if "sektor" in user_query.lower():
                    parts = re.split(r"sektor", user_query, flags=re.IGNORECASE)
                    if len(parts) > 1:
                        sector_name = parts[-1].strip()
                        out = handle_sector_search(sector_name)
                        
                        # --- PERUBAHAN DI SINI ---
                        return jsonify({
                            "reply": out.get("response_text", out.get("reply", "")),
                            "related_queries": out.get("options", [])
                        }), 200
                
                return jsonify({"reply": "Maaf, format request sektor tidak dikenali."}), 200
            except Exception as e:
                log.exception("sector parse error: %s", e)
                return jsonify({"reply": "Maaf, terjadi kesalahan saat memproses permintaan sektor."}), 200
                                    
        # ---- DATA AGENT ----
        if intent == "run_data_agent":
            subs = decompose_query_with_llm(user_query)
            if not subs: subs = [user_query]
            
            final_responses = []
            related_options = []
            found_any_data = False

            for s in subs:
                res = handle_dataset_search(s, show_preview=show_preview)
                
                if res["status"] == "success":
                    found_any_data = True
                    final_responses.append(res["response_text"])
                elif res["status"] == "multiple_options":
                    found_any_data = True
                    final_responses.append(res["response_text"])
                    if "options" in res:
                        related_options.extend(res["options"])
                
                # Jika status == "not_found", loop lanjut ke query berikutnya
            
            # 2. JIKA DATA DITEMUKAN -> Tampilkan Data
            if found_any_data:
                combined_text = "\n\n---\n\n".join(final_responses)
                return jsonify({
                    "reply": combined_text, 
                    "related_queries": related_options 
                }), 200
            
            # 3. JIKA TIDAK ADA DATA SAMA SEKALI -> FALLBACK KE LLM
            # Ini yang menjawab request "Jawaban bukan mencari data terdekat, tapi jawaban LLM"
            
            log.info("[FALLBACK] Dataset strict filter rejected all candidates. Asking LLM General Knowledge.")
            
            fallback_out = handle_general_question(user_query)
            llm_reply = fallback_out.get("reply", "")
            
            return jsonify({
                "reply": llm_reply, # Langsung kirim jawaban LLM yang sopan tadi
                "results": []
            }), 200

        # ... (Fallback global)
        return jsonify({"reply": "Maaf, saya tidak mengerti."}), 200

    except Exception as e:
        log.exception("Error utama: %s", e)
        return jsonify({"reply": "Terjadi kesalahan internal server."}), 500
        
# -------------------------
# RUN FLASK
# -------------------------
if __name__ == "__main__":
    log.info("Starting chatbot on http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=True)
