"""Check Facebook post content about anemoi character routes."""
import requests
import time

url = "https://www.facebook.com/NineASMR/posts/key%E7%A4%BE%E6%96%B0%E4%BD%9Canemoi%E7%9A%84%E6%96%B0%E6%83%85%E5%A0%B1%E9%80%99%E5%B9%BE%E5%A4%A9visual-arts%E5%AE%98%E7%B6%B2%E9%87%8B%E5%87%BA%E4%BA%86c103%E6%99%82%E7%99%BC%E5%94%AE%E7%9A%84%E5%B0%8F%E5%86%8A%E5%AD%90%E5%AE%8C%E6%95%B4%E7%89%88%E9%9B%BB%E5%AD%90%E6%AA%94%E6%9C%AC%E7%AF%87%E5%85%A7%E5%AE%B9%E6%98%AF%E5%8F%83%E8%80%83%E5%85%A7%E6%96%87%E5%AE%8C%E6%95%B4%E9%9B%BB%E5%AD%90%E6%AA%94%E4%BB%A5%E5%8F%8A%E6%96%87%E7%AB%A0%E8%BD%89%E8%BC%89%E4%BE%86%E6%BA%90%E9%99%84%E6%96%BC%E9%80%A3%E7%B5%90-%E6%9C%AC/215346518317208/"

print("Fetching Facebook post via Jina...")
t0 = time.time()
resp = requests.get(
    f"https://r.jina.ai/{url}",
    headers={"Accept": "text/markdown"},
    timeout=30,
)
print(f"Done in {time.time()-t0:.1f}s, {len(resp.text)} chars\n")

text = resp.text

# Search for key terms about routes
for kw in ["新岛", "路线", "线路", "脚本", "担当", "负责", "キャラ", "ルート",
           "朱比華", "爱乃", "愛乃", "辻倉", "ED", "作词", "シナリオ",
           "纯真", "天然", "不良", "红莲", "A", "B", "無垢"]:
    idxes = []
    start = 0
    while True:
        idx = text.find(kw, start)
        if idx == -1:
            break
        idxes.append(idx)
        start = idx + 1
    if idxes:
        print(f"'{kw}': {len(idxes)} 次匹配 at {idxes[:5]}")

# Show the most relevant sections
print("\n=== 正文内容（前 5000 字符）===")
print(text[:5000])

# If there's a section about routes, try to find it
print("\n=== 搜索 'ルート' 或 'route' 上下文 ===")
for kw in ["ルート", "route", "脚本", "シナリオ"]:
    idx = text.find(kw)
    if idx != -1:
        start = max(0, idx - 100)
        end = min(len(text), idx + 300)
        print(f"  '{kw}' at {idx}:")
        print(f"  {text[start:end]}")
        print()
