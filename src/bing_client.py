"""Google Search API wrapper via SerpAPI, with Jina Reader full-text enrichment."""
import time
import urllib3
import requests
from src.config import SERP_API_KEY, SEARCH_TOP_K, SERPAPI_ENDPOINT

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

JINA_ENDPOINT = "https://r.jina.ai"


def _jina_fetch(url: str, timeout: int = 30) -> str | None:
    """Fetch full page content via Jina Reader. Returns markdown text or None."""
    try:
        resp = requests.get(
            f"{JINA_ENDPOINT}/{url}",
            headers={"Accept": "text/markdown"},
            timeout=timeout,
        )
        if resp.status_code == 200 and resp.text.strip():
            return resp.text.strip()
    except Exception:
        pass
    return None


def search(query: str, top_k: int = SEARCH_TOP_K, use_jina: bool = True) -> list[dict]:
    """Search Google via SerpAPI and return structured results.

    When use_jina=True, fetch full page text via Jina Reader for each URL.
    Falls back to snippet if Jina fails.
    """
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
        url = item.get("link", "")
        snippet = item.get("snippet", "")

        full_text = snippet
        full_text_source = "snippet"

        if use_jina and url:
            jina_text = _jina_fetch(url)
            if jina_text:
                full_text = jina_text
                full_text_source = "jina"
            time.sleep(0.3)  # rate limit

        results.append({
            "rank": i,
            "title": item.get("title", ""),
            "url": url,
            "snippet": snippet,
            "full_text": full_text,
            "full_text_source": full_text_source,
            "full_text_len": len(full_text),
        })
    return results
