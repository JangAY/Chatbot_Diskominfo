import json
import logging
import pandas as pd
import requests
from io import BytesIO
from garut_knowledge_base.config import API_BASE, RAW_API_PATH, KNOWLEDGE_PATH

log = logging.getLogger("build_knowledge_base")


# ====================================================================
#  LOAD DATASET FILE
# ====================================================================
def load_dataset_file(download_url: str, title: str):
    """
    Load dataset dari URL:
    - Skip HTML
    - CSV
    - XLSX (.xlsx)
    - XLS  (97-2003)
    """
    try:
        r = requests.get(download_url, timeout=15)
        file_data = r.content

        # HTML → skip
        if file_data.strip().startswith(b"<") or b"html" in file_data[:200].lower():
            print(f"⚠️ File HTML, bukan dataset: {title}")
            raise ValueError("HTML content returned, not a dataset")

        header = file_data[:8]

        # CSV autodetect
        if download_url.endswith(".csv") or (b"," in file_data[:200] and b"\n" in file_data[:200]):
            return pd.read_csv(BytesIO(file_data))

        # XLSX (ZIP → PK header)
        if header[:2] == b"PK":
            return pd.read_excel(BytesIO(file_data), engine="openpyxl")

        # XLS Legacy
        if header == b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1":
            return pd.read_excel(BytesIO(file_data), engine="xlrd")

        print(f"⚠️ Format tidak dikenali: {title} | Header: {header}")
        raise ValueError("Unknown file format")

    except Exception as e:
        print(f"Failed to parse dataset: {title}", e)
        return None


# ====================================================================
#  RINGKAS SEMUA ROW (LEBIH BAIK UNTUK SEMANTIC SEARCH)
# ====================================================================
def compact_rows(df: pd.DataFrame, max_rows: int = 200):
    """
    Merapikan isi dataset menjadi embedding-friendly text.
    Truncate ke 200 baris agar aman.
    """
    if df is None or df.empty:
        return ""

    # Limit jumlah row agar embedding tidak terlalu besar
    if len(df) > max_rows:
        df = df.head(max_rows)

    result = []
    for _, row in df.iterrows():
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

    for d in datasets:
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

            # -----------------------------------------
            # LOAD DATA FRAME
            # -----------------------------------------
            df = None
            columns = []
            sample_rows = []
            all_rows_compacted = ""

            if download_url:
                df = load_dataset_file(download_url, title)

                if isinstance(df, pd.DataFrame):
                    columns = list(df.columns)
                    sample_rows = df.head(5).to_dict(orient="records")

                    # NEW: bawa isi dataset untuk embedding
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

                # NEW
                "rows": all_rows_compacted,
            }

            res["kumpulan_dataset"].append(kb_item)

        except Exception as e:
            print("Error parsing dataset:", e)
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


# ====================================================================
#  BUILD EMBEDDING TEXT (VERSI FINAL)
# ====================================================================
def build_embedding_text(item):
    """Teks embedding super lengkap agar pertanyaan numerik bisa dijawab."""

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
