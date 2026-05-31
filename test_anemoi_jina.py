"""SEVE on CUSTOM_ANEMOI with Jina Reader full-text enrichment."""
import json
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.seve import extract_claims, generate_answer, verify_claims, apply_verification
from src.utils import format_search_results

QID = "CUSTOM_ANEMOI"
QUESTION = "anemoi中新岛夕负责哪些部分？"
JINA_ENDPOINT = "https://r.jina.ai"

# --- Load cached search results ---
cache = json.loads((ROOT / "data/search_cache.json").read_text("utf-8"))
questions = json.loads((ROOT / "data/freshqa_questions.json").read_text("utf-8"))
entry = cache.get(QID, {})
results = entry.get("results", [])
answer_ref = ""
for q in questions:
    if q.get("question_id") == QID:
        answer_ref = q.get("reference_answer", "")

print(f"问题: {QUESTION}")
print(f"参考答案: {answer_ref}")
print(f"搜索结果数: {len(results)}")
print()

# --- Show original snippets ---
print("=" * 60)
print("原始 snippet vs Jina 全文对比")
print("=" * 60)

enriched_results = []
for r in results:
    url = r["url"]
    snippet_len = len(r.get("snippet", ""))

    # Fetch from Jina
    t0 = time.time()
    try:
        resp = requests.get(
            f"{JINA_ENDPOINT}/{url}",
            headers={"Accept": "text/markdown"},
            timeout=30,
        )
        elapsed = time.time() - t0
        if resp.status_code == 200 and resp.text.strip():
            jina_text = resp.text.strip()
        else:
            jina_text = r.get("snippet", "")
            print(f"  [!] Jina status={resp.status_code}, fallback to snippet")
    except Exception as e:
        elapsed = time.time() - t0
        jina_text = r.get("snippet", "")
        print(f"  [!] Jina error: {e}")

    jina_len = len(jina_text)
    print(f"[{r['rank']}] {r['title'][:60]}")
    print(f"    URL: {url[:100]}")
    print(f"    snippet: {snippet_len} 字符 -> Jina: {jina_len} 字符 ({jina_len/max(1,snippet_len):.0f}x)  {elapsed:.1f}s")
    print(f"    Jina 前 200: {jina_text[:200]}")
    print()

    enriched_results.append({
        **r,
        "full_text": jina_text,
        "snippet": r.get("snippet", ""),
    })
    time.sleep(0.3)

# --- SEVE pipeline with enriched results ---
print()
print("=" * 60)
print("STEP 1: 结构化抽取 (基于 Jina 全文)")
print("=" * 60)
claims = extract_claims(enriched_results, QUESTION)
print(f"抽取到 {len(claims)} 条声明:\n")
for c in claims:
    conflict = " [冲突]" if c.get("has_conflict") else ""
    print(f"  [{c['id']}] {c['claim'][:200]}{conflict}")
    print(f"       原文: {c['snippet'][:200]}")
    print(f"       来源: {c['url'][:100]}")
    print()

if claims:
    print("=" * 60)
    print("STEP 2: 校准生成")
    print("=" * 60)
    answer, citation_map = generate_answer(QUESTION, claims)
    print(answer)
    print()

    if enriched_results:
        print("=" * 60)
        print("STEP 3: 反向验证")
        print("=" * 60)
        verifications = verify_claims(answer, claims, enriched_results)
        for v in verifications:
            flag = "YES" if v["verdict"] == "YES" else ("PARTIAL" if v["verdict"] == "PARTIAL" else "NO")
            print(f"  [{v['claim_id']}] {flag:>8} | {v['claim_text'][:120]}")
        print()

        print("=" * 60)
        print("STEP 4: 最终输出")
        print("=" * 60)
        final = apply_verification(answer, verifications)
        print(final)
else:
    print("无声明，管道终止")

# --- Summary comparison ---
print()
print("=" * 60)
print("对比总结")
print("=" * 60)
print(f"  snippet 模式: 1 条声明 (\\\"新岛夕负责anemoi的剧本\\\")")
print(f"  Jina 全文模式: {len(claims)} 条声明")
