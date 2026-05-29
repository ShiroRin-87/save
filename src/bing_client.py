"""Google Search API wrapper via SerpAPI."""
import urllib3
import requests
from src.config import SERP_API_KEY, SEARCH_TOP_K, SERPAPI_ENDPOINT

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def search(query: str, top_k: int = SEARCH_TOP_K) -> list[dict]:
    """Search Google via SerpAPI and return structured results."""
    params = {
        "engine": "google",
        "q": query,
        "api_key": SERP_API_KEY,
        "num": top_k,
    }
    resp = requests.get(SERPAPI_ENDPOINT, params=params, timeout=30, verify=False)
    resp.raise_for_status()
    data = resp.json()

    results = []
    for i, item in enumerate(data.get("organic_results", [])[:top_k], start=1):
        results.append({
            "rank": i,
            "title": item.get("title", ""),
            "url": item.get("link", ""),
            "snippet": item.get("snippet", ""),
            "full_text": item.get("snippet", ""),
        })
    return results
