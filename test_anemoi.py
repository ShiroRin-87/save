"""Test all 3 methods on custom question: 新岛夕 in anemoi."""
import json, sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.bing_client import search
from src.model_client import generate
from src.seve import extract_claims, generate_answer, verify_claims, apply_verification
from src.utils import load_search_cache, format_search_results_numbered, save_json

QID = "CUSTOM_ANEMOI2"
QUESTION = "新岛夕在anemoi中负责了哪些线路的内容？"

# Step 1: Search & cache
print(f"Searching: {QUESTION}")
results = search(QUESTION)
print(f"  Got {len(results)} results")
for i, r in enumerate(results, 1):
    print(f"  [{i}] {r['title'][:80]}")

cache = load_search_cache()
cache[QID] = {"query": QUESTION, "timestamp": __import__("time").strftime("%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime()), "results": results}
save_json(cache, "data/search_cache.json")
context = format_search_results_numbered(results)

# === SEVE ===
print("\n" + "=" * 60)
print("SEVE — Structured Extraction")
print("=" * 60)
try:
    claims = extract_claims(results)
    print(f"Extracted {len(claims)} claims:")
    for c in claims:
        print(f"  [{c['id']}] {c['claim'][:120]}")
except Exception as e:
    print(f"Extraction FAILED: {e}")
    claims = []

if claims:
    print("\n" + "=" * 60)
    print("SEVE — Calibrated Generation")
    print("=" * 60)
    seve_answer, citation_map = generate_answer(QUESTION, claims)
    print(seve_answer)

    print("\n" + "=" * 60)
    print("SEVE — Verification")
    print("=" * 60)
    verifications = verify_claims(seve_answer, claims, results)
    for v in verifications:
        print(f"  [{v['claim_id']}] {v['verdict']}")

    final = apply_verification(seve_answer, verifications)
    save_json({QID: {"question": QUESTION, "claims": claims, "num_claims": len(claims)}}, "outputs/seve_custom/structured_tables.json")
    save_json({QID: {"question": QUESTION, "answer": seve_answer, "citation_map": citation_map}}, "outputs/seve_custom/generated_answers.json")
    save_json({QID: {"question": QUESTION, "verifications": verifications}}, "outputs/seve_custom/verification_results.json")
    save_json({QID: {"question": QUESTION, "answer": final, "method": "seve", "search_available": bool(results)}}, "outputs/seve_custom/final_answers.json")
else:
    print("\nSEVE SKIPPED — extraction failed")
    seve_answer = "[SEVE extraction failed]"
    final = seve_answer

# === Vanilla RAG ===
print("\n" + "=" * 60)
print("VANILLA RAG")
print("=" * 60)
v_answer = generate(f"""你是一个问答助手。请基于以下搜索结果为用户问题提供准确、简洁的回答。
每个事实声明后标注引用编号，如 [1][2]，对应搜索结果的序号。

搜索结果：
{context}

用户问题：{QUESTION}

请回答：""")
print(v_answer)

# === Self-RAG ===
print("\n" + "=" * 60)
print("SELF-RAG")
print("=" * 60)
r1 = generate(f"""基于以下搜索结果回答问题。在回答中，为每个事实片段标注 [Relevant] 或 [Irrelevant]。

搜索结果：
{context}

问题：{QUESTION}""")
r2 = generate(f"""以下是一个回答及其标注。请检查每个声明是否被原文搜索结果支撑。
标注每个声明为 [Supported]、[Partially]、[Unsupported]。

回答：
{r1}

搜索结果：
{context}""")
r3 = generate(f"""基于反思结果重新生成最终答案。只保留 [Supported] 和 [Partially] 的声明。

原始回答：
{r1}

反思结果：
{r2}

用户问题：{QUESTION}""")
print(r3)

# Save baselines
try: existing_v = json.loads((ROOT / "outputs/vanilla_rag/answers.json").read_text("utf-8"))
except: existing_v = {}
existing_v[QID] = {"question": QUESTION, "answer": v_answer, "method": "vanilla_rag", "search_available": bool(results)}
save_json(existing_v, "outputs/vanilla_rag/answers.json")

try: existing_s = json.loads((ROOT / "outputs/self_rag/answers.json").read_text("utf-8"))
except: existing_s = {}
existing_s[QID] = {"question": QUESTION, "answer": r3, "method": "self_rag", "round1": r1, "round2": r2, "search_available": bool(results)}
save_json(existing_s, "outputs/self_rag/answers.json")

print("\nDone.")
