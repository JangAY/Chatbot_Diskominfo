import os
import json
import logging
import traceback
import pandas as pd
import requests
from fastapi import FastAPI, Request
from dotenv import load_dotenv
from chromadb import PersistentClient
from openai import OpenAI

# -------------------------------------------------------------------
# INITIAL SETUP
# -------------------------------------------------------------------

load_dotenv()
logging.basicConfig(
    level=logging.DEBUG,
    format="\n[%(levelname)s] %(asctime)s - %(message)s"
)
log = logging.getLogger("CHATBOT_DEBUG")

app = FastAPI()

# Variabel environment
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DATASET_DIR = "chatbot_db"

client = OpenAI(api_key=GEMINI_API_KEY)

# -------------------------------------------------------------------
# CHROMA CLIENT
# -------------------------------------------------------------------

chroma_client = PersistentClient(path=DATASET_DIR)
dataset_collection = chroma_client.get_or_create_collection(name="dataset_embeddings")

# -------------------------------------------------------------------
# SEARCH EMBEDDINGS
# -------------------------------------------------------------------

def search_dataset_embeddings(query: str, n_results: int = 5):
    log.debug(f"[SEARCH_DATASET] Query received: {query}")

    try:
        result = dataset_collection.query(
            query_texts=[query],
            n_results=n_results,
            include=["distances", "metadatas", "documents", "ids"]
        )
    except Exception as e:
        log.error(f"[SEARCH_DATASET] ERROR querying Chroma: {e}")
        return []

    log.debug(f"[SEARCH_DATASET] Raw chroma result: {result}")

    ids = result.get("ids", [[]])[0]
    metas = result.get("metadatas", [[]])[0]
    dists = result.get("distances", [[]])[0]

    candidates = []
    for i, meta in enumerate(metas):
        dist = dists[i] if i < len(dists) else -1
        log.debug(
            f"[SEARCH_DATASET] Candidate {i} | title={meta.get('title')} | distance={dist}"
        )
        meta["_distance"] = float(dist)
        meta["_id"] = ids[i] if i < len(ids) else None
        candidates.append(meta)

    return candidates

# -------------------------------------------------------------------
# LOAD DATASET
# -------------------------------------------------------------------

def load_full_dataframe_from_url(url: str):
    log.debug(f"[LOAD_DF] Start loading file: {url}")

    try:
        resp = requests.get(url)
    except Exception as e:
        log.error(f"[LOAD_DF] Request failed: {e}")
        return None

    content_type = resp.headers.get("content-type", "").lower()
    ext = url.split(".")[-1].lower()

    log.debug(f"[LOAD_DF] content-type={content_type}, ext={ext}")

    try:
        if "csv" in content_type or ext == "csv":
            df = pd.read_csv(url)

        elif "excel" in content_type or ext in ("xlsx", "xls"):
            df = pd.read_excel(url)

        else:
            log.error(f"[LOAD_DF] Unsupported file type: {content_type}")
            return None

        log.debug(f"[LOAD_DF] Loaded DF shape={df.shape}, columns={list(df.columns)}")
        return df

    except Exception as e:
        log.error(f"[LOAD_DF] Failed parsing DF: {e}")
        log.error(traceback.format_exc())
        return None

# -------------------------------------------------------------------
# FILTER ROWS
# -------------------------------------------------------------------

def find_relevant_rows(df: pd.DataFrame, query: str):
    log.debug(f"[FIND_ROWS] Start matching for query: {query}")
    log.debug(f"[FIND_ROWS] DF shape before filter: {df.shape}")

    try:
        subset = df[df.apply(lambda r: r.astype(str).str.contains(query, case=False).any(), axis=1)]
        log.debug(f"[FIND_ROWS] Subset shape after filter: {subset.shape}")
        return subset
    except Exception as e:
        log.error(f"[FIND_ROWS] ERROR filtering: {e}")
        return pd.DataFrame()

# -------------------------------------------------------------------
# AI ANALYSIS
# -------------------------------------------------------------------

def ai_analysis(df_subset: pd.DataFrame, query: str):
    prompt = f"""
You are a data analyst. Analyze the following dataset rows based on the user question.

User question:
{query}

Dataset preview:
{df_subset.head().to_string(index=False)}

Provide a short explanation and answer clearly.
"""

    log.debug("[AI_ANALYSIS] Running analysis...")
    log.debug(f"[AI_ANALYSIS] Prompt used:\n{prompt}")

    try:
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": "You are a data analysis assistant."},
                {"role": "user", "content": prompt}
            ]
        )

        answer = response.choices[0].message["content"]
        log.debug(f"[AI_ANALYSIS] Raw LLM response: {answer}")
        return answer

    except Exception as e:
        log.error(f"[AI_ANALYSIS] ERROR: {e}")
        log.error(traceback.format_exc())
        return "⚠️ Gagal melakukan analisis AI."

# -------------------------------------------------------------------
# MAIN DATASET HANDLER
# -------------------------------------------------------------------

def handle_dataset_search(query: str):
    log.debug(f"[DATA_AGENT] Received query: {query}")

    # 1) Cari dataset via embedding
    candidates = search_dataset_embeddings(query, n_results=5)
    log.debug(f"[DATA_AGENT] Raw candidates: {candidates}")

    if not candidates:
        return {"answer": "Tidak ada dataset cocok ditemukan."}

    # Filter threshold
    filtered = [c for c in candidates if c["_distance"] < 1.2]
    log.debug(f"[DATA_AGENT] Filtered candidates: {filtered}")

    if not filtered:
        return {"answer": "Dataset ada tapi tidak relevan dengan pertanyaan."}

    chosen = filtered[0]
    title = chosen.get("title")
    url = chosen.get("download_url")

    log.debug(f"[DATA_AGENT] Dataset terpilih: {title} | URL={url}")

    # 2) Load dataframe
    df = load_full_dataframe_from_url(url)
    log.debug(f"[DATA_AGENT] DF loaded: OK={df is not None}")

    if df is None or df.empty:
        return {"answer": "Dataset tidak bisa dibaca atau kosong."}

    # 3) Cari subset
    subset = find_relevant_rows(df, query)
    log.debug(f"[DATA_AGENT] Subset rows found: {len(subset)}")

    if subset.empty:
        return {"answer": f"Dataset ditemukan ({title}), tetapi tidak ada baris relevan."}

    # 4) AI analysis
    analysis = ai_analysis(subset, query)

    return {
        "dataset": title,
        "rows": len(subset),
        "preview": subset.head().to_dict(orient="records"),
        "analysis": analysis
    }

# -------------------------------------------------------------------
# FASTAPI ENDPOINT
# -------------------------------------------------------------------

@app.post("/chat")
async def chat_endpoint(req: Request):
    data = await req.json()
    user_query = data.get("query", "")

    log.info(f"[REQUEST] User query: {user_query}")
    log.debug(f"[REQUEST_DEBUG] Raw payload: {data}")

    try:
        result = handle_dataset_search(user_query)
        return result
    except Exception as e:
        log.error(f"[FATAL] Unhandled error: {e}")
        log.error(traceback.format_exc())
        return {"error": "Internal error."}
