"""Fetch metadata from Garut Satu Data API and save raw JSON."""
import json
import logging
from garut_knowledge_base.config import API_BASE, RAW_API_PATH
import requests


log = logging.getLogger("fetch_api_data")




def fetch_and_save(api_url: str = API_BASE) -> dict:
    """Fetch API data and write to RAW_API_PATH. Return parsed JSON."""
    try:
        headers = {"User-Agent": "garut-knowledge-sync/1.0"}
        r = requests.get(api_url, timeout=30, headers=headers)
        r.raise_for_status()
        data = r.json()
        RAW_API_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(RAW_API_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        log.info("Saved raw API data to %s", RAW_API_PATH)
        return data
    except Exception as e:
        log.exception("Failed to fetch/save API data: %s", e)
        return {}




if __name__ == "__main__":
    fetch_and_save()