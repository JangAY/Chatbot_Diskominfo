import json
import logging
from garut_knowledge_base.config import API_BASE, RAW_API_PATH, KNOWLEDGE_PATH

log = logging.getLogger("build_knowledge_base")

def build_from_raw(raw: dict) -> dict:
    """Fix parser for Satu Data Garut API structure."""
    
    res = {
        "panduan_situs": {
            "about": {
                "description": "Garut Satu Data — metadata digabung untuk keperluan chatbot lokal.",
                "keywords": ["garut satu data", "dataset", "portal garut"]
            }
        },
        "kumpulan_dataset": []
    }

    datasets = raw.get("dataset", [])
    if not isinstance(datasets, list):
        log.error("Invalid dataset format in raw API")
        return res

    for d in datasets:
        try:
            # Main metadata fields
            title = d.get("title")
            description = d.get("description")
            tahun = d.get("tahun")
            landing_page = d.get("landingPage")
            
            # Publisher
            publisher = None
            pub = d.get("publisher")
            if isinstance(pub, dict):
                publisher = pub.get("name")

            # Extract distribution section (list)
            dist = None
            dist_list = d.get("distribution", [])
            if isinstance(dist_list, list) and len(dist_list) > 0:
                dist = dist_list[0]

            download_url = None
            if dist:
                download_url = dist.get("downloadURL") or dist.get("accessURL")

            # Build dataset item for knowledge
            kb_item = {
                "original_title": title,
                "title": title,
                "description": description,
                "landing_page": landing_page,
                "download_url": download_url,
                "publisher": publisher,
                "tahun": tahun,
                "knowledge": f"""
                Judul: {title};
                Tahun: {tahun};
                Penerbit: {publisher};
                Deskripsi: {description};
                Landing Page: {landing_page};
                """
            }

            res["kumpulan_dataset"].append(kb_item)

        except Exception as e:
            print(f"Error parsing dataset: {e}")
            continue

    # Save file
    KNOWLEDGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    KNOWLEDGE_PATH.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    log.info("Wrote knowledge base to %s", KNOWLEDGE_PATH)
    return res


def build_knowledge_base():
    if RAW_API_PATH.exists():
        raw = json.loads(RAW_API_PATH.read_text(encoding="utf-8"))
        return build_from_raw(raw)
    else:
        print("ERROR: raw_api_data.json not found — run fetch_api_data.py first")
        return None


if __name__ == "__main__":
    build_knowledge_base()
