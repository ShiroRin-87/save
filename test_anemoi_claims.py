"""Show CUSTOM_ANEMOI: claims + search snippets side by side."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.seve import extract_claims

QID = "CUSTOM_ANEMOI"
QUESTION = "anemoi中新岛夕负责哪些部分？"

# Load cache and questions
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
print()

print("=" * 60)
print("搜索结果 (snippet 当前只有 ~80-100 字符)")
print("=" * 60)
for r in results:
    print(f"[{r['rank']}] {r['title']}")
    print(f"    snippet: {r.get('snippet', '')[:300]}")
    print()

print("=" * 60)
print("结构化抽取结果")
print("=" * 60)
claims = extract_claims(results, QUESTION)
print(f"抽取到 {len(claims)} 条声明:\n")
for c in claims:
    print(f"  ID: {c['id']}")
    print(f"  声明: {c['claim']}")
    print(f"  原文: {c['snippet'][:200]}")
    print(f"  来源: {c['url']}")
    print(f"  冲突: {c['has_conflict']}")
    print()
