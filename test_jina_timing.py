"""Measure Jina Reader latency per call."""
import time
import requests

JINA_ENDPOINT = "https://r.jina.ai"

TEST_URLS = [
    "https://en.wikipedia.org/wiki/Rick_Riordan",
    "https://www.texasmonthly.com/articles/rick-riordan-2/",
    "https://zh.wikipedia.org/wiki/%E5%88%9B%E4%BD%9C%E5%BD%BC%E5%A5%B3%E7%9A%84%E6%81%8B%E7%88%B1%E5%85%AC%E5%BC%8F",
    "https://www.bilibili.com/read/cv11908783/",
    "https://h-ero-game.com/archives/5264/",
]

print("=== Jina Reader 延迟测试 ===\n")

total = 0
for url in TEST_URLS:
    domain = url.split("/")[2]
    t0 = time.time()
    try:
        resp = requests.get(
            f"{JINA_ENDPOINT}/{url}",
            headers={"Accept": "text/markdown"},
            timeout=30,
        )
        elapsed = time.time() - t0
        status = resp.status_code
        chars = len(resp.text) if status == 200 else 0
        print(f"{domain:<30} {elapsed:5.1f}s  status={status}  chars={chars:,}")
        total += elapsed
    except Exception as e:
        elapsed = time.time() - t0
        print(f"{domain:<30} {elapsed:5.1f}s  ERROR: {e}")
        total += elapsed

print(f"\n总耗时: {total:.1f}s, 平均: {total/len(TEST_URLS):.1f}s/条")
print(f"预估 600 题 (5 条/题 = 3000 次调用): {total/len(TEST_URLS)*3000/60:.0f} 分钟")
