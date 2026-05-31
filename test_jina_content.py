"""Check what Jina actually returned for the Moegirl page."""
import requests
import time

JINA_ENDPOINT = "https://r.jina.ai"

# The Moegirl.tw page that returned 38K chars
url = "https://zh.moegirl.tw/%E6%96%B0%E5%B2%9B%E5%A4%95"

print(f"Fetching: {url}")
t0 = time.time()
resp = requests.get(f"{JINA_ENDPOINT}/{url}", headers={"Accept": "text/markdown"}, timeout=60)
print(f"Done in {time.time()-t0:.1f}s, {len(resp.text)} chars\n")

text = resp.text

# Search for relevant keywords
keywords = ["anemoi", "Anemoi", "路线", "线路", "シナリオ", "剧本", "ED", "作词", "作曲",
            "朱比華", "爱乃", "愛乃", "辻倉", "担当", "负责", "SAGA"]

print("=== 关键词搜索 ===")
for kw in keywords:
    count = text.count(kw)
    if count > 0:
        print(f"  '{kw}': {count} 次")

# Find lines containing key terms
print("\n=== 包含 'anemoi' 或 'Anemoi' 的上下文 (前后各 200 字符) ===")
for kw in ["anemoi", "Anemoi"]:
    idx = 0
    while True:
        idx = text.find(kw, idx)
        if idx == -1:
            break
        start = max(0, idx - 100)
        end = min(len(text), idx + 300)
        print(f"  [...{text[start:end]}...]")
        print("  ---")
        idx += len(kw)

print("\n=== 作品表附近内容 (前 5000 字符) ===")
print(text[:5000])
