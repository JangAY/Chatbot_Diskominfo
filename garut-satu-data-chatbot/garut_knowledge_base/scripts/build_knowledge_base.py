import json
import logging
import pandas as pd
import requests
import time
from io import BytesIO
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from garut_knowledge_base.config import API_BASE, RAW_API_PATH, KNOWLEDGE_PATH

log = logging.getLogger("build_knowledge_base")

# ====================================================================
#  CONFIGURE SESSION WITH RETRIES
# ====================================================================
def get_retry_session(retries=3, backoff_factor=1, status_forcelist=(500, 502, 504)):
    """
    Creates a requests session that retries on connection errors and timeouts.
    """
    session = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

# ====================================================================
#  LOAD DATASET FILE
# ====================================================================
def load_dataset_file(download_url: str, title: str):
    """
    Load dataset from URL with robust error handling and retries.
    """
    if not download_url:
        return None

    # Skip known invalid extensions or likely web pages to save time
    if download_url.endswith(('.php', '.html', '.htm')):
         print(f"⏩ Skipping likely HTML URL for: {title}")
         return None

    session = get_retry_session()
    
    try:
        # Increase timeout to 45 seconds
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        r = session.get(download_url, timeout=45, headers=headers)
        
        # Check content type header before reading content
        content_type = r.headers.get('Content-Type', '').lower()
        if 'text/html' in content_type:
             print(f"⚠️ HTML content detected in header for: {title} ({download_url})")
             return None
             
        file_data = r.content

        # Double check content for HTML tags
        if file_data.strip().startswith(b"<") or b"<html" in file_data[:500].lower():
            print(f"⚠️ Body contains HTML, not dataset: {title}")
            return None

        header = file_data[:8]

        # CSV autodetect
        if download_url.endswith(".csv") or (b"," in file_data[:500] and b"\n" in file_data[:500]):
            try:
                return pd.read_csv(BytesIO(file_data), on_bad_lines='skip')
            except Exception as csv_e:
                 print(f"⚠️ CSV parsing failed for {title}: {csv_e}")

        # XLSX (ZIP -> PK header)
        if header[:2] == b"PK":
            return pd.read_excel(BytesIO(file_data), engine="openpyxl")

        # XLS Legacy
        if header == b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1":
            return pd.read_excel(BytesIO(file_data), engine="xlrd")

        # Fallback for text files that might be CSVs without extension
        try:
             # Try reading as csv one last time if size suggests it's data
             if len(file_data) > 0:
                return pd.read_csv(BytesIO(file_data), on_bad_lines='skip')
        except:
            pass

        print(f"⚠️ Format tidak dikenali: {title} | Header: {header[:10]}")
        return None

    except requests.exceptions.ReadTimeout:
        print(f"❌ Timeout giving up on: {title}")
        return None
    except Exception as e:
        print(f"❌ Failed to parse dataset: {title}", e)
        return None


# ====================================================================
#  RINGKAS SEMUA ROW (LEBIH BAIK UNTUK SEMANTIC SEARCH)
# ====================================================================
def compact_rows(df: pd.DataFrame, max_rows: int = 200):
    """
    Merapikan isi dataset menjadi embedding-friendly text.
    """
    if df is None or df.empty:
        return ""

    if len(df) > max_rows:
        df = df.head(max_rows)

    result = []
    for _, row in df.iterrows():
        # Convert all to string and handle NaN
        clean_row = {str(k): ("" if pd.isna(v) else str(v)) for k, v in row.items()}
        result.append(clean_row)

    return json.dumps(result, ensure_ascii=False)


# ====================================================================
#  BUILD KNOWLEDGE BASE
# ====================================================================
def build_from_raw(raw: dict) -> dict:
    res = {
        "panduan_situs": {
            "about": {
                "description": "Garut Satu Data — metadata digabung untuk chatbot lokal.",
                "keywords": ["garut satu data", "dataset", "portal garut"]
            }
        },
        "kumpulan_dataset": []
    }

    datasets = raw.get("dataset", [])
    if not isinstance(datasets, list):
        log.error("Invalid dataset format")
        return res
    
    total = len(datasets)
    print(f"🔍 Found {total} datasets. Starting processing...")

    for i, d in enumerate(datasets):
        try:
            title = d.get("title")
            description = d.get("description")
            tahun = d.get("tahun")
            landing_page = d.get("landingPage")
            publisher = d.get("publisher", {}).get("name") if isinstance(d.get("publisher"), dict) else None

            # Prefer schema baru
            download_url = d.get("download_url")

            # Fallback old schema
            if not download_url:
                dist_list = d.get("distribution", [])
                if dist_list:
                    dist = dist_list[0]
                    download_url = dist.get("downloadURL") or dist.get("accessURL")
            
            # Progress marker
            if i % 10 == 0:
                print(f"⏳ Processing {i}/{total}: {title}")

            # -----------------------------------------
            # LOAD DATA FRAME
            # -----------------------------------------
            df = None
            columns = []
            sample_rows = []
            all_rows_compacted = ""

            if download_url:
                # Add a small delay to be polite to the server and avoid rate limits
                time.sleep(0.5) 
                df = load_dataset_file(download_url, title)

                if isinstance(df, pd.DataFrame) and not df.empty:
                    # Clean column names (strip whitespace)
                    df.columns = df.columns.astype(str).str.strip()
                    columns = list(df.columns)
                    sample_rows = df.head(5).to_dict(orient="records")
                    all_rows_compacted = compact_rows(df)

            # -----------------------------------------
            # SIMPAN ITEM KB
            # -----------------------------------------
            kb_item = {
                "title": title,
                "tahun": tahun,
                "publisher": publisher,
                "description": description,
                "landing_page": landing_page,
                "download_url": download_url,
                "columns": columns,
                "sample": sample_rows,
                "rows": all_rows_compacted,
            }

            res["kumpulan_dataset"].append(kb_item)

        except Exception as e:
            print(f"Error processing item {i}: {e}")
            continue

    KNOWLEDGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    KNOWLEDGE_PATH.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    log.info("Wrote knowledge base to %s", KNOWLEDGE_PATH)
    return res


def build_knowledge_base():
    if RAW_API_PATH.exists():
        raw = json.loads(RAW_API_PATH.read_text(encoding="utf-8"))
        return build_from_raw(raw)
    else:
        print("ERROR: raw_api_data.json not found — run fetch_api_data.py")
        return None

# Keep the build_embedding_text function if it's imported elsewhere, 
# but it is not strictly used inside this file's main execution flow.
def build_embedding_text(item):
    title = item.get("title", "")
    desc = item.get("description", "")
    tahun = item.get("tahun", "")
    publisher = item.get("publisher", "")
    columns_text = ", ".join(item.get("columns", []))
    sample_text = json.dumps(item.get("sample", []), ensure_ascii=False)
    rows_text = item.get("rows", "")

    return f"""
Dataset: {title}
Tahun: {tahun}
Publisher: {publisher}

Deskripsi:
{desc}

Kolom:
{columns_text}

Contoh Baris Pertama:
{sample_text}

Isi Dataset (diringkas):
{rows_text}
"""

if __name__ == "__main__":
    build_knowledge_base()