"""Convert raw API data into a compact knowledge_base.json used by the bot and for embeddings."""
import json
import logging
from garut_knowledge_base.config import API_BASE, RAW_API_PATH, KNOWLEDGE_PATH

log = logging.getLogger("build_knowledge_base")

def build_from_raw(raw: dict) -> dict:
    """Transform raw API structure into internal knowledge format.
    Produces two main keys: panduan_situs, kumpulan_dataset (list).
    """
    res = {"panduan_situs": {}, "kumpulan_dataset": []}


    # Basic panduan_situs minimal (can be extended)
    res["panduan_situs"] = {
        "about": {
            "description": "Garut Satu Data — metadata digabung untuk keperluan chatbot lokal.",
            "keywords": ["garut satu data", "dataset", "portal garut"]
        }
    }

    datasets = raw.get("dataset") or raw.get("results") or raw
    if isinstance(datasets, dict) and "dataset" in datasets:
        datasets = datasets["dataset"]

    for d in datasets or []:
        try:
            # many fields exist inside; we keep the important ones
            title = d.get("title") or d.get("judul") or d.get("label")
            publisher = None
            pub = d.get("publisher")
            if isinstance(pub, dict):
                publisher = pub.get("name")
            elif isinstance(pub, str):
                publisher = pub


            # distribution array -> choose first usable distribution
            dist = None
            downloads = d.get("distribution") or d.get("dataset") or []
            if isinstance(downloads, dict):
                dist = downloads
            else:
                if downloads:
                    dist = downloads[0]


            download_url = None
            landing_page = d.get("landingPage") or d.get("accessURL") or d.get("landing_page")
            tahun = d.get("tahun") or d.get("issued")
            description = d.get("description") or (dist and dist.get("description")) or ""


            if dist:
                download_url = dist.get("downloadURL") or dist.get("accessURL") or dist.get("format")


            kb_item = {
                "original_title": title,
                "title": title,
                "description": description,
                "download_url": download_url,
                "landing_page": landing_page,
                "publisher": publisher,
                "tahun": tahun,
                "knowledge": f"Judul: {title or '-'}; Diterbitkan oleh: {publisher or '-'}; Deskripsi: {description or '-'}"
            }
            res["kumpulan_dataset"].append(kb_item)
        except Exception:
            continue



    # save
    KNOWLEDGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(KNOWLEDGE_PATH, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    log.info("Wrote knowledge base to %s", KNOWLEDGE_PATH)
    return res

# Alias untuk kompatibilitas dengan main.py
def build_knowledge_base():
    import json
    from garut_knowledge_base.config import RAW_API_PATH

    raw = {}
    if RAW_API_PATH.exists():
        raw = json.loads(RAW_API_PATH.read_text(encoding="utf-8"))
    else:
        print("raw_api_data.json not found — run fetch_api_data.py first")

    return build_from_raw(raw)

if __name__ == "__main__":
    try:
        raw = {}
        if RAW_API_PATH.exists():
            raw = json.loads(RAW_API_PATH.read_text(encoding="utf-8"))
        else:
            print("raw_api_data.json not found — run fetch_api_data.py first")
        build_from_raw(raw)
    except Exception as e:
        print("Error building knowledge base:", e)