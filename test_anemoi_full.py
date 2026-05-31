"""Full SEVE pipeline test on CUSTOM_ANEMOI."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.seve import extract_claims, generate_answer, verify_claims, apply_verification
from src.utils import load_search_cache, save_json, format_search_results

QID = "CUSTOM_ANEMOI"
QUESTION = "anemoi中新岛夕负责哪些部分？"

cache = load_search_cache()
entry = cache.get(QID, {})
results = entry.get("results", [])
print(f"问题: {QUESTION}")
print(f"搜索结果数: {len(results)}")
print()

print("--- 搜索结果 snippet ---")
for r in results:
    print(f"[{r['rank']}] {r['title']}")
    print(f"    URL: {r['url']}")
    print(f"    Snippet ({len(r.get('snippet',''))} 字符): {r.get('snippet','')[:200]}")
    print()

print("=" * 60)
print("STEP 1: 结构化抽取")
print("=" * 60)
try:
    claims = extract_claims(results, QUESTION)
    print(f"抽取成功: {len(claims)} 条声明")
except Exception as e:
    print(f"抽取失败: {e}")
    claims = []

if claims:
    print()
    for c in claims:
        conflict = " [冲突]" if c.get("has_conflict") else ""
        print(f"  [{c['id']}] {c['claim'][:150]}{conflict}")

    print()
    print("=" * 60)
    print("STEP 2: 校准生成")
    print("=" * 60)
    try:
        answer, citation_map = generate_answer(QUESTION, claims)
        print(f"生成成功: {len(answer)} 字符")
        print()
        print(answer)
    except Exception as e:
        print(f"生成失败: {e}")
        answer = ""

    if answer and results:
        print()
        print("=" * 60)
        print("STEP 3: 反向验证")
        print("=" * 60)
        try:
            verifications = verify_claims(answer, claims, results)
            print(f"验证了 {len(verifications)} 条引用:")
            for v in verifications:
                flag = "✅" if v["verdict"] == "YES" else ("⚠️" if v["verdict"] == "PARTIAL" else "❌")
                print(f"  {flag} [{v['claim_id']}] {v['verdict']}")

            print()
            print("=" * 60)
            print("STEP 4: 最终输出（验证后）")
            print("=" * 60)
            final = apply_verification(answer, verifications)
            print(final)
        except Exception as e:
            print(f"验证失败: {e}")
else:
    print("无声明，管道终止。")
