"""Baidu Search API wrapper via SerpAPI, with Jina Reader full-text enrichment."""
import re
import time
import urllib3
import requests
from src.config import SERP_API_KEY, SEARCH_TOP_K, SERPAPI_ENDPOINT, JINA_MAX_CHARS

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

JINA_ENDPOINT = "https://r.jina.ai"
CONTEXT_WINDOW = 2000  # chars before and after each keyword match


def _split_query_keywords(query: str) -> list[str]:
    """Split question into searchable keyword segments, longest first."""
    # Split by Chinese/English punctuation
    parts = re.split(r"[，。、；：？！,\.;:!\?\s]+", query)
    parts = [p.strip() for p in parts if len(p.strip()) >= 4]
    # Also extract quoted substrings and entities
    quoted = re.findall(r"[「『\"]([^」』\"]+)[」』\"]", query)
    parts.extend(quoted)
    # Deduplicate, longest first (more specific matches)
    seen = set()
    unique = []
    for p in sorted(parts, key=len, reverse=True):
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique[:8]  # at most 8 keywords


def _extract_context(full_text: str, keywords: list[str],
                     window: int = CONTEXT_WINDOW) -> str:
    """Extract context windows around keyword matches in full text.

    Returns concatenated context snippets (up to JINA_MAX_CHARS total).
    Deduplicates overlapping windows. Falls back to beginning of text if
    no keywords match.
    """
    if not full_text or not keywords:
        return full_text[:JINA_MAX_CHARS]

    text_len = len(full_text)
    ranges = []  # list of (start, end) match windows

    for kw in keywords:
        pos = 0
        while pos < text_len:
            idx = full_text.find(kw, pos)
            if idx == -1:
                break
            start = max(0, idx - window)
            end = min(text_len, idx + len(kw) + window)
            ranges.append((start, end))
            pos = idx + len(kw)

    if not ranges:
        # No keyword match — fall back to first JINA_MAX_CHARS chars
        return full_text[:JINA_MAX_CHARS]

    # Merge overlapping windows
    ranges.sort()
    merged = []
    for r in ranges:
        if merged and r[0] <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], r[1]))
        else:
            merged.append(r)

    # Concatenate, respecting JINA_MAX_CHARS limit
    total = 0
    snippets = []
    for start, end in merged:
        chunk = full_text[start:end]
        if total + len(chunk) > JINA_MAX_CHARS and snippets:
            break
        snippets.append(chunk)
        total += len(chunk)

    return "\n...\n".join(snippets)


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
    """Search via SerpAPI and return structured results.

    When use_jina=True, fetch full page text via Jina Reader, then extract
    context windows around query keywords. Falls back to snippet if Jina fails.
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

    keywords = _split_query_keywords(query)

    results = []
    for i, item in enumerate(data.get("organic_results", [])[:top_k], start=1):
        url = item.get("link", "")
        snippet = item.get("snippet", "")

        full_text = snippet
        full_text_source = "snippet"

        if use_jina and url:
            jina_text = _jina_fetch(url)
            if jina_text:
                full_text = _extract_context(jina_text, keywords)
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
