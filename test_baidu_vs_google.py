"""Compare Baidu vs Google SerpAPI results."""
import json, time
from src.bing_client import search
from src.config import SERP_API_KEY, SEARCH_TOP_K, SERPAPI_ENDPOINT
import requests
import urllib3
urllib3.disable_warnings()

JINA_ENDPOINT = "https://r.jina.ai"
q = "20世纪二十年代中在上海成立的刊物成为了我国知名学生运动的先导"

# ── Baidu ──
print("=" * 60)
print("BAIDU")
print("=" * 60)
t0 = time.time()
baidu_params = {
    "engine": "baidu", "q": q, "api_key": SERP_API_KEY, "num": 5,
}
baidu_resp = requests.get(SERPAPI_ENDPOINT, params=baidu_params, timeout=60, verify=False)
baidu_data = baidu_resp.json()
bt = time.time() - t0

baidu_results = baidu_data.get("organic_results", [])
print(f"Results: {len(baidu_results)}, time: {bt:.1f}s")
print(f"Top-level keys: {list(baidu_data.keys())[:10]}")
for i, r in enumerate(baidu_results[:5]):
    print(f"\n  [{i+1}] {r.get('title', '')[:80]}")
    print(f"  snippet: {len(r.get('snippet', ''))} chars")
    print(f"  link: {r.get('link', '')[:80]}")

# ── Google ──
print("\n" + "=" * 60)
print("GOOGLE")
print("=" * 60)
t0 = time.time()
google_params = {
    "engine": "google", "q": q, "api_key": SERP_API_KEY, "num": 5,
}
google_resp = requests.get(SERPAPI_ENDPOINT, params=google_params, timeout=60, verify=False)
google_data = google_resp.json()
gt = time.time() - t0

google_results = google_data.get("organic_results", [])
print(f"Results: {len(google_results)}, time: {gt:.1f}s")
print(f"Top-level keys: {list(google_data.keys())[:10]}")
for i, r in enumerate(google_results[:5]):
    print(f"\n  [{i+1}] {r.get('title', '')[:80]}")
    print(f"  snippet: {len(r.get('snippet', ''))} chars")
    print(f"  link: {r.get('link', '')[:80]}")

# ── Summary ──
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Baidu:  {len(baidu_results)} results, {bt:.1f}s")
print(f"Google: {len(google_results)} results, {gt:.1f}s")

if baidu_results:
    print(f"\nBaidu snippet avg: {sum(len(r.get('snippet','')) for r in baidu_results[:5])//max(1,len(baidu_results[:5]))} chars")
if google_results:
    print(f"Google snippet avg: {sum(len(r.get('snippet','')) for r in google_results[:5])//max(1,len(google_results[:5]))} chars")
