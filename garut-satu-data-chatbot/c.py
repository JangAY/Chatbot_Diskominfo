# chatbot.py
import os
import re
import io
import json
import sys
import traceback
from typing import Optional, List

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
# IMPORT PREPROCESS (fallback bila tidak ada)
# -------------------------
try:
    from preprocessing_utils import preprocess_text
except Exception as e:
    log.warning("[INIT] Gagal mengimpor preprocess_text, menggunakan fallback sederhana. (%s)", e)
    def preprocess_text(x: str) -> str:
        return x.strip()

# -------------------------
# INISIALISASI & KONFIGURASI
# -------------------------
log.info("[INIT] Memulai aplikasi...")
load_dotenv()

# Konfigurasi Gemini API
try:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY tidak ditemukan di environment.")
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    ]
    genai.configure(api_key=api_key)
    log.info("[GEMINI] Gemini API configured.")
except Exception as e:
    log.error("[GEMINI] Gagal mengkonfigurasi Gemini API: %s", e)
    log.debug(traceback.format_exc())
    sys.exit(1)

# Path DB Chroma
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, "chatbot_db")

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Load embedding model
try:
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    log.info("[EMBED] SentenceTransformer loaded: all-MiniLM-L6-v2")
except Exception as e:
    log.exception("[EMBED] Gagal memuat model embedding: %s", e)
    raise

# Generation model (sesuaikan nama model dengan akun Anda)
try:
    generation_model = genai.GenerativeModel(
        model_name='gemini-2.5-flash',  # ubah jika perlu / anda punya akses model lain
        safety_settings=safety_settings
    )
    log.info("[GEMINI] Generation model inited: gemini-2.5")
except Exception as e:
    log.exception("[GEMINI] Gagal membuat generation_model: %s", e)
    # jangan sys.exit; biarkan berjalan tapi blok panggilan LLM nanti akan error ditangani
    generation_model = None

# Muat ChromaDB
try:
    log.debug("[CHROMADB] Mencoba load ChromaDB di path: %s", DB_PATH)
    if not os.path.exists(DB_PATH):
        log.warning("[CHROMADB] Folder database tidak ditemukan: %s", DB_PATH)
    else:
        log.debug("[CHROMADB] Isi folder DB: %s", os.listdir(DB_PATH))

    client = chromadb.PersistentClient(path=DB_PATH)
    log.debug("[CHROMADB] PersistentClient berhasil dibuat.")
    collections = client.list_collections()
    log.debug("[CHROMADB] Koleksi tersedia: %s", [c.name for c in collections])

    # Nama koleksi sesuai proyek Anda
    site_guide_collection = client.get_collection(name="panduan_situs")
    dataset_collection = client.get_collection(name="kumpulan_dataset")
    log.info("[CHROMADB] Koleksi dimuat: panduan_situs, kumpulan_dataset")
except Exception as e:
    log.error("[CHROMADB] Tidak dapat memuat ChromaDB: %s", e)
    log.debug(traceback.format_exc())
    site_guide_collection = None
    dataset_collection = None

# Ambang jarak embedding
DISTANCE_THRESHOLD = 0.4  # ubah berdasarkan experimen (0.35-0.9)

# -------------------------
# UTIL: helper
# -------------------------
def safe_print_df_info(df: Optional[pd.DataFrame], label: str = "DF"):
    if df is None:
        log.debug("[%s] None", label)
        return
    log.debug("[%s] shape=%s, columns=%s", label, df.shape, list(df.columns[:20]))

def _content_type_is_html(ct: Optional[str]) -> bool:
    if not ct:
        return False
    return "text/html" in ct.lower()

def _guess_ext_from_content_type(ct: Optional[str]) -> Optional[str]:
    if not ct:
        return None
    ct = ct.lower()
    if "excel" in ct or "spreadsheet" in ct or "application/vnd.ms-excel" in ct or "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" in ct:
        return ".xlsx"
    if "csv" in ct or "text/csv" in ct:
        return ".csv"
    if "json" in ct:
        return ".json"
    return None

# -------------------------
# LOAD DATAFRAME DARI URL (robust)
# -------------------------
def load_full_dataframe_from_url(url: str) -> Optional[pd.DataFrame]:
    """
    Muat CSV/XLSX/JSON dari URL. Kembalikan DataFrame atau None.
    Menangani:
     - ekstensi file (.csv, .xls, .xlsx, .json)
     - content-type header
     - fallback bila URL mengarah ke HTML (kembalikan None)
    """
    if not url:
        log.debug("[DATA] Tidak ada URL diberikan.")
        return None

    log.debug("[DATA] Memuat file dari URL: %s", url)
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        resp = requests.get(url, timeout=30, headers=headers, allow_redirects=True)
        resp.raise_for_status()

        content_type = resp.headers.get("Content-Type", "")
        if _content_type_is_html(content_type):
            # ini halaman web (HTML) — bukan file data langsung
            log.warning("[DATA] URL nampak mengarah ke halaman HTML (bukan file langsung): %s (Content-Type: %s)", url, content_type)
            return None

        # tentukan ekstensi
        ext = os.path.splitext(url.split("?")[0])[-1].lower()
        if not ext or ext == "":
            # coba dari content-type
            guessed = _guess_ext_from_content_type(content_type)
            if guessed:
                ext = guessed
                log.debug("[DATA] Ext diduga dari content-type: %s", ext)

        file_bytes = io.BytesIO(resp.content)

        if ext == ".csv":
            df = pd.read_csv(file_bytes)
        elif ext in [".xls", ".xlsx"]:
            # pandas.read_excel mendukung BytesIO
            df = pd.read_excel(file_bytes)
        elif ext == ".json":
            try:
                df = pd.read_json(file_bytes)
            except ValueError:
                # coba decode text dan loads lalu normalize
                text = resp.content.decode('utf-8', errors='ignore')
                data = json.loads(text)
                df = pd.json_normalize(data)
        else:
            # coba beberapa pembaca secara berturut-turut
            # 1) coba read_excel
            try:
                df = pd.read_excel(file_bytes)
            except Exception:
                file_bytes.seek(0)
                try:
                    df = pd.read_csv(file_bytes)
                except Exception:
                    file_bytes.seek(0)
                    try:
                        df = pd.read_json(file_bytes)
                    except Exception:
                        log.warning("[DATA] Gagal mendeteksi/parse file untuk URL: %s", url)
                        return None

        # normalisasi kolom
        df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
        safe_print_df_info(df, "LoadedDF")
        return df

    except requests.HTTPError as he:
        log.error("[DATA] HTTPError saat memuat %s : %s", url, he)
        log.debug(traceback.format_exc())
        return None
    except Exception as e:
        log.error("[DATA] Gagal memuat DataFrame dari URL: %s. Error: %s", url, e)
        log.debug(traceback.format_exc())
        return None

# -------------------------
# FIND RELEVANT ROWS (Deterministic)
# -------------------------
def find_relevant_rows(df: pd.DataFrame, query: str) -> pd.DataFrame:
    if df is None or df.empty:
        log.debug("[AGENT] find_relevant_rows: DataFrame kosong atau None.")
        return pd.DataFrame()

    q = str(query).lower()
    log.debug("[AGENT] find_relevant_rows: Mencari kata kunci untuk query: '%s'", q)

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
    if years:
        log.debug("[AGENT] Years detected: %s", years)

    keywords = [k for k in list(dict.fromkeys([k.lower() for k in keywords if k]))]
    if not keywords and not years:
        log.debug("[AGENT] No explicit keywords or years found; will try heuristics.")
    else:
        log.debug("[AGENT] Keywords for search: %s", keywords)

    mask_total = pd.Series(False, index=df.index)

    string_cols = [c for c in df.columns if pd.api.types.is_string_dtype(df[c]) or df[c].dtype == object]
    log.debug("[AGENT] String-like columns: %s", string_cols[:20])

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
        log.debug("[AGENT] Column-name heuristics candidate cols: %s", candidate_colnames)
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
    except Exception as e:
        log.error("[AGENT] Error applying mask to dataframe: %s", e)
        subset = pd.DataFrame()

    log.debug("[AGENT] find_relevant_rows: Found %d matching rows.", subset.shape[0])
    if subset.shape[0] > 5000:
        log.debug("[AGENT] Subset > 5000 rows, truncating to 500 for safety.")
        subset = subset.head(500)

    return subset

# -------------------------
# DATE COLUMN DETECTION
# -------------------------
def find_date_column(df: pd.DataFrame) -> Optional[str]:
    date_like_cols = []
    for c in df.columns:
        if re.search(r"(tanggal|date|waktu|periode|bulan|tahun|tgl)", c, flags=re.IGNORECASE):
            date_like_cols.append(c)
    for c in date_like_cols:
        try:
            if pd.api.types.is_datetime64_any_dtype(df[c]):
                return c
        except Exception:
            continue
    for c in date_like_cols:
        try:
            df[c + "_parsed_temp"] = pd.to_datetime(df[c], errors='coerce')
            if df[c + "_parsed_temp"].notna().sum() > 0:
                df.drop(columns=[c + "_parsed_temp"], inplace=True, errors=True)
                return c
            df.drop(columns=[c + "_parsed_temp"], inplace=True, errors=True)
        except Exception:
            continue
    return date_like_cols[0] if date_like_cols else None

# -------------------------
# ANALISIS DENGAN LLM (HANYA bila subset ada)
# -------------------------
def analyze_data_with_llm(df: pd.DataFrame, query: str) -> str:
    if df is None:
        return "_(Gagal memuat data untuk dianalisis.)_"

    subset = find_relevant_rows(df, query)
    if subset is None or subset.empty:
        log.debug("[AGENT] analyze_data_with_llm: subset kosong.")
        return "Maaf, data spesifik tersebut tidak tersedia di dataset."

    if re.search(r"\bhari ini\b|\bterbaru\b|\bterkini\b", query.lower()):
        date_col = find_date_column(subset)
        if date_col:
            log.debug("[AGENT] Detected date-like column for 'terbaru': %s", date_col)
            try:
                parsed = pd.to_datetime(subset[date_col], errors='coerce')
                subset = subset.assign(_parsed_date=parsed).sort_values("_parsed_date").drop(columns=["_parsed_date"])
            except Exception:
                pass

    preview = subset.head(20).copy()
    preview = preview.applymap(lambda x: str(x)[:120] + ("..." if isinstance(x, str) and len(str(x)) > 120 else x))
    table_md = tabulate(preview, headers="keys", tablefmt="github", showindex=False)

    prompt = (
        "Anda adalah analis data yang teliti untuk 'Kabupaten Garut'.\n"
        "Gunakan HANYA tabel data yang diberikan untuk menjawab PERTANYAAN PENGGUNA.\n"
        "Jawaban harus singkat (maks 2 kalimat). Jika data menunjukkan jumlah/harga untuk tahun tertentu, sebutkan nilainya beserta tahun.\n"
        "Jika tabel berisi beberapa baris untuk kata kunci yang sama, berikan agregat sederhana bila relevan.\n\n"
        f"PERTANYAAN: {query}\n\n"
        "TABEL (maks 20 baris):\n"
        f"{table_md}\n\n"
        "JAWABAN (bahasa Indonesia, 1-2 kalimat, fokus hanya pada Kabupaten Garut):"
    )

    if generation_model is None:
        log.warning("[GEMINI] generation_model tidak tersedia. Mengembalikan pratinjau tabel saja.")
        return "Saya menemukan data relevan, tetapi LLM tidak tersedia untuk meringkas. Berikut pratinjau:\n\n" + table_md

    try:
        log.debug("[GEMINI] Mengirim subset ke LLM untuk analisis.")
        response = generation_model.generate_content(prompt)
        block_reason = "BLOCK_REASON_UNSPECIFIED"
        if getattr(response, "prompt_feedback", None):
            block_reason = response.prompt_feedback.block_reason.name
        if block_reason != "BLOCK_REASON_UNSPECIFIED":
            return f"_(Maaf, AI tidak dapat menganalisis data ini. Alasan: {block_reason})_"
        return response.text.strip()
    except Exception as e:
        log.error("[GEMINI] Error saat memanggil LLM: %s", e)
        log.debug(traceback.format_exc())
        return "_(Terjadi kesalahan saat menganalisis data dengan LLM.)_"

# -------------------------
# RINGKASAN DESKRIPSI DATASET
# -------------------------
def summarize_with_llm(text_to_summarize: str) -> str:
    if not text_to_summarize or len(text_to_summarize.split()) < 12:
        return (text_to_summarize or "")[:300]
    prompt = (
        "Ringkas deskripsi dataset berikut dalam satu kalimat deskriptif dalam Bahasa Indonesia. Konteksnya adalah data 'Kabupaten Garut'.\n\n"
        "DESKRIPSI ASLI:\n"
        f"{text_to_summarize}\n\n"
        "RINGKASAN (satu kalimat):"
    )
    if generation_model is None:
        return (text_to_summarize or "")[:200]
    try:
        response = generation_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        log.debug("[GEMINI] summarize_with_llm error: %s", e)
        return text_to_summarize[:150].strip() + "..."

# -------------------------
# DECOMPOSE QUERY (LLM -> fallback)
# -------------------------
def decompose_query_with_llm_llm(user_query: str) -> List[str]:
    schema = {"type": "ARRAY", "items": {"type": "STRING"}}
    prompt = (
        "Anda adalah asisten pemecah masalah. ...\n"  # (sama seperti sebelumnya, dipersingkat di sini)
        f"Pertanyaan: \"{user_query}\"\n"
        "Jawaban JSON array of strings:"
    )
    if generation_model is None:
        return decompose_query_fallback(user_query)
    try:
        response = generation_model.generate_content(
            prompt,
            generation_config={
                "response_mime_type": "application/json",
                "response_schema": schema
            }
        )
        block_reason = "BLOCK_REASON_UNSPECIFIED"
        if getattr(response, "prompt_feedback", None):
            block_reason = response.prompt_feedback.block_reason.name
        if block_reason != "BLOCK_REASON_UNSPECIFIED":
            raise Exception(f"Dekomposisi diblokir: {block_reason}")
        result = json.loads(response.text)
        log.debug("[AGENT] decompose result (LLM): %s", result)
        return result
    except Exception as e:
        log.warning("[AGENT] decompose_query_with_llm_llm gagal: %s", e)
        log.debug(traceback.format_exc())
        return decompose_query_fallback(user_query)

def decompose_query_fallback(user_query: str) -> List[str]:
    log.debug("[AGENT] Menggunakan fallback dekomposisi (deterministik).")
    text = user_query.strip().lower()
    parts = re.split(r"\s+dan\s+|,|\s+serta\s+|\s+&\s+", text)
    parts = [p.strip() for p in parts if p.strip()]
    expanded = []
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
    log.debug("[AGENT] decompose fallback result: %s", expanded)
    return expanded

def decompose_query_with_llm(user_query: str) -> List[str]:
    try:
        return decompose_query_with_llm_llm(user_query)
    except Exception as e:
        log.debug("[AGENT] decompose_query_with_llm exception: %s", e)
        return decompose_query_fallback(user_query)

# -------------------------
# HANDLE DATASET SEARCH (utama) - perbaikan
# -------------------------
def handle_dataset_search(sub_query: str) -> dict:
    log.debug("[AGENT] handle_dataset_search: sub_query='%s'", sub_query)
    if not dataset_collection:
        return {"status": "error", "query": sub_query, "error_message": "Error: Database dataset tidak dapat diakses."}

    try:
        embedding_list = embedding_model.encode([sub_query]).tolist()
        results = dataset_collection.query(
            query_embeddings=embedding_list,
            n_results=6,
            include=["metadatas", "documents", "distances"]
        )

        # debug printing of results
        try:
            dists = results.get('distances', [[]])[0]
            metas = results.get('metadatas', [[]])[0]
            docs = results.get('documents', [[]])[0]
            log.debug("[CHROMADB] distances: %s", dists)
            log.debug("[CHROMADB] titles: %s", [m.get('title') for m in metas])
        except Exception:
            log.debug("[CHROMADB] Hasil query tidak lengkap: %s", results)

        found_docs = results.get('documents')
        if not found_docs or not found_docs[0]:
            return {"status": "error", "query": sub_query, "error_message": f"Maaf, saya tidak menemukan dataset yang cocok untuk '{sub_query}'."}

        # kumpulkan kandidat yang jaraknya kurang dari threshold
        candidate_datasets = []
        distances = results['distances'][0]
        metadatas = results['metadatas'][0]
        documents = results['documents'][0]
        for i in range(len(distances)):
            log.debug("[CHROMADB] Candidate %d: title='%s' distance=%.4f", i, metadatas[i].get('title'), distances[i])
            if distances[i] < DISTANCE_THRESHOLD:
                candidate_datasets.append((metadatas[i], documents[i], distances[i]))

        if not candidate_datasets:
            # beri informasi distances teratas untuk debugging
            top_info = []
            for i in range(min(3, len(distances))):
                top_info.append(f"{metadatas[i].get('title')} (dist={distances[i]:.4f})")
            log.debug("[CHROMADB] No candidate below threshold. Top candidates: %s", top_info)
            return {"status": "error", "query": sub_query, "error_message": f"Maaf, saya tidak dapat menemukan dataset yang cukup relevan untuk '{sub_query}'. (Top: {top_info})"}

        # pilih yang paling relevan
        candidate_datasets.sort(key=lambda tup: tup[2])
        dataset_info, dataset_doc, _ = candidate_datasets[0]
        dataset_name = dataset_info.get('title', 'Tanpa Judul')
        dataset_url = dataset_info.get('url', '#')
        download_url = dataset_info.get('download_url')
        log.info("[AGENT] Top candidate: %s | download_url: %s", dataset_name, download_url)

        description_sentence = summarize_with_llm(dataset_doc)

        # jika tidak ada link unduh langsung atau link mengarah ke halaman HTML -> beri penjelasan
        if not download_url or str(download_url).strip() in ["", "-"]:
            log.debug("[AGENT] Dataset ditemukan tetapi tidak ada download_url.")
            return {
                "status": "success",
                "query": sub_query,
                "analysis_answer": "Maaf, dataset ditemukan tetapi tidak ada tautan unduhan langsung untuk menampilkan data.",
                "description_sentence": description_sentence,
                "table_preview": None,
                "dataset_name": dataset_name,
                "dataset_url": dataset_url
            }

        # cek apakah download_url tampak file nyata (ekstensi atau content-type)
        try:
            head = requests.head(download_url, timeout=15, headers={'User-Agent': 'Mozilla/5.0'}, allow_redirects=True)
            ct = head.headers.get("Content-Type", "") if head is not None else ""
            log.debug("[DATA] HEAD %s -> Content-Type: %s", download_url, ct)
            if _content_type_is_html(ct) and not download_url.lower().endswith((".csv", ".xls", ".xlsx", ".json")):
                # link mengarah ke halaman HTML, jangan coba download langsung
                log.warning("[DATA] download_url tampak mengarah ke halaman HTML, tidak akan di-download: %s", download_url)
                return {
                    "status": "success",
                    "query": sub_query,
                    "analysis_answer": "Dataset ditemukan, tetapi tautan unduhan mengarah ke halaman, bukan file data langsung. Silakan buka sumber data: " + dataset_url,
                    "description_sentence": description_sentence,
                    "table_preview": None,
                    "dataset_name": dataset_name,
                    "dataset_url": dataset_url
                }
        except Exception:
            log.debug("[DATA] HEAD request gagal/ditolak, akan mencoba GET langsung.")

        # muat file (GET) via load_full_dataframe_from_url
        full_df = load_full_dataframe_from_url(download_url)
        if full_df is None:
            log.warning("[DATA] Gagal memuat file dataset (full_df None) untuk: %s", download_url)
            return {"status": "error", "query": sub_query, "error_message": "Maaf, gagal memuat file dataset dari tautan yang tersedia. Coba buka: " + dataset_url}

        # jalankan analisis
        data_analysis_response = analyze_data_with_llm(full_df, sub_query)

        # buat table_preview bila ada subset relevan
        subset = find_relevant_rows(full_df, sub_query)
        table_preview = None
        if subset is not None and not subset.empty:
            df_preview = subset.head(5).copy()
            df_preview = df_preview.applymap(lambda x: str(x)[:50] + ('...' if isinstance(x, str) and len(str(x)) > 50 else x))
            table_preview = f"```\n{tabulate(df_preview, headers='keys', tablefmt='github', showindex=False)}\n```"

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
        log.error("[AGENT] Error di handle_dataset_search: %s", e)
        log.debug(traceback.format_exc())
        return {"status": "error", "query": sub_query, "error_message": f"Maaf, terjadi kesalahan saat mencari data untuk '{sub_query}'."}

# -------------------------
# HANDLE GENERAL QUESTION
# -------------------------
def handle_general_question(query: str) -> dict:
    if not site_guide_collection:
        return {"reply": "Error: Database panduan situs tidak dapat diakses."}
    try:
        query_embedding = embedding_model.encode([query]).tolist()
        results = site_guide_collection.query(
            query_embeddings=query_embedding,
            n_results=1,
            include=["documents", "distances"]
        )
        if not results or not results.get("documents") or not results['documents'][0]:
            return {"reply": "Maaf, saya tidak dapat menemukan informasi yang relevan dengan pertanyaan umum Anda."}
        context = results['documents'][0][0]
        prompt = (
            "Anda adalah asisten AI 'Satu Data Garut'. Jawab pertanyaan pengguna tentang portal Satu Data Garut berdasarkan konteks yang diberikan.\n"
            "Gunakan Bahasa Indonesia yang ringkas dan jelas. JANGAN PERNAH menyebut 'Indonesia', fokus hanya pada 'Kabupaten Garut'.\n"
            "KONTEKS:\n"
            f"{context}\n\n"
            f"PERTANYAAN: {query}\nJAWABAN:"
        )
        if generation_model is None:
            return {"reply": "Maaf, layanan LLM tidak tersedia saat ini untuk menjawab pertanyaan umum."}
        response = generation_model.generate_content(prompt)
        return {"reply": response.text.strip()}
    except Exception as e:
        log.error("[GENERAL] Error handle_general_question: %s", e)
        log.debug(traceback.format_exc())
        return {"reply": "Maaf, terjadi masalah saat mencoba menghasilkan jawaban."}

# -------------------------
# LIST SECTORS
# -------------------------
def handle_list_sectors() -> dict:
    if not dataset_collection:
        return {"reply": "Error: Database dataset tidak dapat diakses."}
    try:
        data = dataset_collection.get(include=["metadatas"])
        publishers = set()
        for metadata in data.get('metadatas', []):
            if isinstance(metadata, dict) and metadata.get('publisher'):
                publishers.add(metadata.get('publisher'))
        if not publishers:
            return {"reply": "Maaf, saat ini tidak ada sektor yang terdaftar di database."}
        new_replies = [{"label": p, "value": f"Tampilkan dataset sektor {p}"} for p in sorted(list(publishers))]
        return {
            "reply": "Tentu, berikut adalah daftar sektor (OPD) yang datanya tersedia. Silakan pilih salah satu:",
            "newQuickReplies": new_replies
        }
    except Exception as e:
        log.error("[LIST] Error handle_list_sectors: %s", e)
        log.debug(traceback.format_exc())
        return {"reply": "Maaf, terjadi kesalahan saat mengambil daftar sektor."}

# -------------------------
# Intent classifier & sector search
# -------------------------
def classify_intent(processed_query: str, raw_query: str) -> str:
    raw_lower = raw_query.lower().strip()
    if raw_lower == "apa saja dataset yang tersedia?":
        return "list_sectors"
    if raw_lower.startswith("tampilkan dataset sektor"):
        return "dataset_sector_search"
    general_keywords = ["siapa kamu", "apa itu", "bagaimana cara", "jelaskan", "apa yang dimaksud"]
    if any(raw_lower.startswith(key) for key in general_keywords):
        return "general_question"
    return "run_data_agent"

def handle_sector_search(sector_name: str) -> dict:
    log.debug("[SECTOR] handle_sector_search: publisher='%s'", sector_name)
    if not dataset_collection:
        return {"reply": "Error: Database dataset tidak dapat diakses."}
    try:
        res = dataset_collection.get(where={"publisher": sector_name}, include=["metadatas"])
        metadatas = res.get('metadatas', [])
        if not metadatas:
            return {"reply": f"Maaf, saya tidak menemukan dataset untuk sektor '{sector_name}'."}
        response_list = [f"Tentu, berikut adalah beberapa dataset teratas untuk **{sector_name}**:\n"]
        for metadata in metadatas[:10]:
            title = metadata.get('title', 'Tanpa Judul')
            url = metadata.get('url', '#')
            response_list.append(f"* **{title}** - {url}")
        return {"reply": "\n".join(response_list)}
    except Exception as e:
        log.error("[SECTOR] Error handle_sector_search: %s", e)
        log.debug(traceback.format_exc())
        return {"reply": f"Maaf, terjadi kesalahan saat mencari data untuk sektor '{sector_name}'."}

# -------------------------
# ROUTE /api/chat
# -------------------------
@app.route("/api/chat", methods=["POST"])
def handle_chat():
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({"error": "Permintaan tidak valid, 'query' dibutuhkan."}), 400

    user_query = data['query']
    log.info("[REQUEST] Query Asli: %s", user_query)

    try:
        corrected_user_query = user_query.strip()
    except Exception:
        corrected_user_query = user_query

    processed_query = preprocess_text(corrected_user_query)
    log.debug("[REQUEST] Processed query: %s", processed_query)

    intent = classify_intent(processed_query, corrected_user_query)
    log.debug("[REQUEST] Detected intent: %s", intent)

    response_data = {}

    if intent == 'general_question':
        response_data = handle_general_question(corrected_user_query)
    elif intent == 'list_sectors':
        response_data = handle_list_sectors()
    elif intent == 'dataset_sector_search':
        try:
            sector_name = corrected_user_query.split("Tampilkan dataset sektor ", 1)[-1].strip()
            log.debug("[REQUEST] Parsed sector_name: %s", sector_name)
            response_data = handle_sector_search(sector_name)
        except Exception as e:
            log.error("[REQUEST] Error parsing sector name: %s", e)
            response_data = {"reply": "Maaf, terjadi kesalahan saat memproses permintaan sektor Anda."}
    elif intent == 'run_data_agent':
        try:
            sub_queries = decompose_query_with_llm(corrected_user_query)
            if not sub_queries:
                raise Exception("Gagal memecah query menjadi sub-query.")

            all_results = []
            for q in sub_queries:
                log.debug("[AGENT] Mengeksekusi sub-query: %s", q)
                res = handle_dataset_search(q)
                all_results.append(res)

            successful_results = [r for r in all_results if r.get('status') == 'success']
            failed_messages = [r.get('error_message') for r in all_results if r.get('status') == 'error']

            final_summary = ""
            if successful_results:
                individual_answers = [
                    f"Untuk pertanyaan '{res['query']}', jawabannya adalah: {res['analysis_answer']}"
                    for res in successful_results
                ]
                answers_list_string = "\n- ".join(individual_answers)
                summary_prompt = (
                    "Anda adalah asisten AI 'Satu Data Garut'. Peran Anda adalah menjawab pertanyaan pengguna HANYA berdasarkan fakta-fakta yang diberikan.\n"
                    "Buat satu paragraf ringkasan singkat yang menggabungkan fakta-fakta berikut (fokus hanya pada Kabupaten Garut):\n\n"
                    f"Pertanyaan Asli: {corrected_user_query}\n\n"
                    "Fakta-fakta:\n"
                    f"- {answers_list_string}\n\n"
                    "Ringkasan (1 paragraf):"
                )
                if generation_model is not None:
                    try:
                        log.debug("[GEMINI] Membuat ringkasan akhir via LLM...")
                        summary_response = generation_model.generate_content(summary_prompt)
                        final_summary = summary_response.text.strip()
                    except Exception as e:
                        log.error("[GEMINI] Gagal membuat ringkasan akhir: %s", e)
                        log.debug(traceback.format_exc())
                        final_summary = "Berikut adalah data yang berhasil saya temukan:\n- " + "\n- ".join(individual_answers)
                else:
                    final_summary = "Berikut adalah data yang berhasil saya temukan:\n- " + "\n- ".join(individual_answers)
            else:
                final_summary = "Maaf, saya tidak dapat menemukan data spesifik yang Anda minta."

            data_blocks = []
            for res in successful_results:
                block = (
                    f"### {res['dataset_name']}\n\n"
                    f"**Ringkasan Deskripsi:**\n{res['description_sentence']}\n\n"
                )
                if res.get('table_preview'):
                    block += f"**Pratinjau Data (beberapa baris relevan):**\n{res['table_preview']}\n\n"
                else:
                    block += f"**Pratinjau Data (tidak tersedia/terbatas):**\n(Tidak ada pratinjau tabel)\n\n"
                block += f"🔗 Sumber: {res['dataset_url']}"
                data_blocks.append(block)

            combined_reply = final_summary
            if data_blocks:
                combined_reply += "\n\n<hr>\n\n" + "\n\n<hr>\n\n".join(data_blocks)

            if not successful_results and failed_messages:
                combined_reply += "\n\n" + "\n".join(failed_messages)

            response_data = {'reply': combined_reply}

        except Exception as e:
            log.error("[AGENT] Error pada Data Agent: %s", e)
            log.debug(traceback.format_exc())
            response_data = {'reply': 'Maaf, terjadi kesalahan besar saat memproses permintaan data Anda.'}
    else:
        response_data = handle_general_question(corrected_user_query)

    if 'reply' not in response_data:
        response_data['reply'] = "Maaf, terjadi kesalahan yang tidak terduga."

    log.info("[REQUEST] RESPON SELESAI")
    return jsonify(response_data)

# -------------------------
# RUN FLASK
# -------------------------
if __name__ == "__main__":
    log.info("[INIT] Menjalankan Flask (debug mode).")
    app.run(host='0.0.0.0', port=5000, debug=True)
