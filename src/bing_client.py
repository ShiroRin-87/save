"""Search via SerpAPI (Google) with Jina Reader full-text enrichment.

Simplified: Jina full-text is truncated to first JINA_MAX_CHARS chars instead
of keyword-based context windows, because riddle-style BrowseComp questions
share no keywords with their answers.
"""
import time
import urllib3
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.config import SERP_API_KEY, SEARCH_TOP_K, SERPAPI_ENDPOINT, JINA_MAX_CHARS

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Simple rate limiter for SerpAPI free tier
_last_search = 0.0
_SEARCH_GAP = 3.0

JINA_ENDPOINT = "https://r.jina.ai"

# Jina quality thresholds
JINA_MIN_CHARS = 500  # below this, treat as fetch failure
JINA_CAPTCHA_SIGNS = ["CAPTCHA", "安全验证", "captcha", "verify", "Please verify"]


def _jina_fetch(url: str, timeout: int = 30) -> tuple[str | None, bool]:
    """Fetch full page content via Jina Reader.

    Returns (markdown_text_or_None, is_low_quality).
    is_low_quality=True means the result may be incomplete (CAPTCHA, short, etc).
    """
    try:
        resp = requests.get(
            f"{JINA_ENDPOINT}/{url}",
            headers={"Accept": "text/markdown"},
            timeout=timeout,
        )
        if resp.status_code == 200 and resp.text.strip():
            text = resp.text.strip()
            is_bad = len(text) < JINA_MIN_CHARS or any(
                sign in text for sign in JINA_CAPTCHA_SIGNS
            )
            return text, is_bad
    except Exception:
        pass
    return None, True


def _truncate(text: str, max_chars: int = JINA_MAX_CHARS) -> str:
    """Truncate text to first max_chars characters."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def search(query: str, top_k: int = SEARCH_TOP_K, use_jina: bool = True) -> list[dict]:
    """Search via SerpAPI (Google) and return structured results.

    When use_jina=True, fetch full page text via Jina Reader, truncate to
    JINA_MAX_CHARS. Falls back to snippet if Jina fails.
    """
    params = {
        "engine": "google",
        "q": query,
        "api_key": SERP_API_KEY,
        "num": top_k,
    }
    time.sleep(3.0)  # avoid 429 on free tier
    resp = requests.get(SERPAPI_ENDPOINT, params=params, timeout=30, verify=False)
    resp.raise_for_status()
    data = resp.json()

    items = list(enumerate(data.get("organic_results", [])[:top_k], start=1))

    # Build base results first (snippet-only, no Jina yet)
    results = []
    jina_tasks = []
    for i, item in items:
        url = item.get("link", "")
        snippet = item.get("snippet", "")
        results.append({
            "rank": i,
            "title": item.get("title", ""),
            "url": url,
            "snippet": snippet,
            "full_text": snippet,
            "full_text_source": "snippet",
            "full_text_len": len(snippet),
            "jina_quality": "ok",
        })
        if use_jina and url:
            jina_tasks.append((i - 1, url))

    # Parallel Jina fetches — dominant time cost, now concurrent
    if jina_tasks:
        def _fetch_one(idx, url):
            jina_text, is_bad = _jina_fetch(url)
            return idx, jina_text, is_bad

        with ThreadPoolExecutor(max_workers=8) as ex:
            futures = {ex.submit(_fetch_one, idx, url): idx for idx, url in jina_tasks}
            for f in as_completed(futures):
                idx, jina_text, is_bad = f.result()
                if jina_text is not None:
                    results[idx]["full_text"] = _truncate(jina_text)
                    results[idx]["full_text_source"] = "jina"
                    results[idx]["full_text_len"] = len(results[idx]["full_text"])
                    results[idx]["jina_quality"] = "low_quality" if is_bad else "ok"

    return results
