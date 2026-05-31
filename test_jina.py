"""Test Jina Reader: fetch full text for cached search results."""
import json
import time
import requests
from pathlib import Path

ROOT = Path(__file__).parent
CACHE_PATH = ROOT / "data" / "search_cache.json"
JINA_ENDPOINT = "https://r.jina.ai"
# 可选: 注册后拿 token 提速率，空着也能用（限速较严）
JINA_TOKEN = ""


def jina_read(url: str) -> str | None:
    """Fetch clean markdown from a URL via Jina Reader. Returns None on failure."""
    headers = {"Accept": "text/markdown"}
    if JINA_TOKEN:
        headers["Authorization"] = f"Bearer {JINA_TOKEN}"
    try:
        resp = requests.get(
            f"{JINA_ENDPOINT}/{url}",
            headers=headers,
            timeout=30,
        )
        if resp.status_code == 200 and resp.text.strip():
            return resp.text.strip()
        else:
            print(f"    Jina status={resp.status_code}, len={len(resp.text)}")
            return None
    except Exception as e:
        print(f"    Jina error: {e}")
        return None


def test_on_question(qid: str):
    """Test Jina on one question's cached search results."""
    cache = json.loads(CACHE_PATH.read_text("utf-8"))
    entry = cache.get(qid)
    if not entry:
        print(f"QID {qid} not in cache")
        return

    results = entry["results"]
    print(f"问题: {entry['query']}")
    print(f"结果数: {len(results)}")
    print()

    for r in results:
        url = r["url"]
        snippet_len = len(r.get("snippet", ""))
        print(f"[{r['rank']}] {r['title']}")
        print(f"    URL: {url[:100]}...")
        print(f"    Snippet: {snippet_len} 字符")

        # Call Jina
        print(f"    正在抓取全文...")
        full = jina_read(url)
        if full:
            print(f"    Jina 全文: {len(full)} 字符")
            print(f"    前 200 字符: {full[:200]}...")
        else:
            print(f"    Jina 失败")
        print()

        time.sleep(0.5)  # 避免触发限速


if __name__ == "__main__":
    import sys
    qid = sys.argv[1] if len(sys.argv) > 1 else "FQ070"
    test_on_question(qid)
