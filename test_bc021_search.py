"""Test whether SerpAPI can find BC021's answer with well-crafted Chinese queries."""
import json, sys, time
from pathlib import Path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from src.bing_client import search

queries = [
    "青霉素 中国 正式投产 年份",
    "青霉素 中国 1953 年 投产",
    "中国 第一批 青霉素 生产 1953",
    "青霉素 国产化 上海 1953",
    "弗莱明 青霉素 中国 投产 哪一年",
]

for q in queries:
    print(f"\n{'='*60}")
    print(f"Query: {q}")
    results = search(q, top_k=5)
    for r in results:
        title = r.get("title", "")[:80]
        snippet = r.get("snippet", "")[:200]
        url = r.get("url", "")[:80]
        print(f"  [{title}]")
        print(f"   {snippet}")
        print(f"   {url}")
        print()
    time.sleep(0.5)
