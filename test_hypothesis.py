"""Test hypothesis-driven search vs direct search on BC070 (青城山)."""
import json, sys, time
from pathlib import Path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.bing_client import search
from src.model_client import generate
from src.seve import (
    hypothesis_driven_search, extract_claims, generate_answer,
    verify_claims, apply_verification, extract_and_generate
)
from src.utils import load_json, save_json, format_search_results_numbered

# BC070 = 青城山 (geography riddle, previously 0/4)
# BC022 = 王安石 (history riddle, Vanilla got right after iterative search)
# BC080 = 梁博 (music riddle, previously all failed)
TEST_IDX = 80

all_qs = load_json("data/browsecomp-zh-decrypted.json")
cache = load_json("data/search_cache.json")

q = all_qs[TEST_IDX]
qid = f"BC{TEST_IDX:03d}"
question = q["Question"]
answer_ref = q["Answer"]
topic = q["Topic"]

print(f"[{qid}] [{topic}] {question}")
print(f"Ref: {answer_ref}")
print()

# ── A) Hypothesis-driven search ──
print("=" * 60)
print("A) HYPOTHESIS-DRIVEN SEARCH")
print("=" * 60)
t0_total = time.time()
results_h = hypothesis_driven_search(question, n_hypotheses=5)
print(f"\nTotal: {len(results_h)} results, {sum(r.get('full_text_len',0) for r in results_h):,} chars ({time.time()-t0_total:.1f}s)")

# Show titles
print("\nResult titles:")
for r in results_h[:8]:
    print(f"  [{r['rank']}] {r['title'][:120]}")

# Vanilla RAG on hypothesis-driven results
print(f"\n--- Vanilla RAG (hypothesis results) ---")
t0 = time.time()
ctx = format_search_results_numbered(results_h)
v = generate(f"你是一个问答助手。请基于以下搜索结果为用户问题提供准确、简洁的回答。回答要尽量简洁。\n\n搜索结果：\n{ctx}\n\n用户问题：{question}\n\n请回答：")
print(f"({time.time()-t0:.1f}s) {v[:500]}")

# SEVE on hypothesis-driven results
print(f"\n--- SEVE (hypothesis results) ---")
t0 = time.time()
claims_h = extract_claims(results_h, question)
print(f"Extract: {len(claims_h)} claims ({time.time()-t0:.1f}s)")
for c in claims_h:
    print(f"  [{c['id']}] {c['claim'][:150]}")
answer_h, cmap_h = generate_answer(question, claims_h)
print(f"Generate: {answer_h[:400]}")
verifs_h = verify_claims(answer_h, claims_h, results_h)
for v in verifs_h:
    print(f"  V[{v['claim_id']}] {v['verdict']}: {v['claim_text'][:80]}")
final_h = apply_verification(answer_h, verifs_h)
print(f">>> SEVE Final ({time.time()-t0:.1f}s): {final_h[:500]}")

# ── B) Direct search (baseline for comparison) ──
print()
print("=" * 60)
print("B) DIRECT SEARCH (baseline)")
print("=" * 60)
cache_key = qid
if cache_key in cache and cache[cache_key].get("results"):
    results_d = cache[cache_key]["results"]
    print(f"Cached: {len(results_d)} results")
else:
    t0 = time.time()
    results_d = search(question)
    cache[cache_key] = {"query": question, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"), "results": results_d}
    save_json(cache, "data/search_cache.json")
    print(f"{len(results_d)} results ({time.time()-t0:.1f}s)")

print(f"\n--- Vanilla RAG (direct results) ---")
t0 = time.time()
ctx_d = format_search_results_numbered(results_d)
v_d = generate(f"你是一个问答助手。请基于以下搜索结果为用户问题提供准确、简洁的回答。回答要尽量简洁。\n\n搜索结果：\n{ctx_d}\n\n用户问题：{question}\n\n请回答：")
print(f"({time.time()-t0:.1f}s) {v_d[:500]}")

# Summary
print(f"\n{'='*60}")
print(f"COMPARISON: BC{TEST_IDX:03d} | Ref={answer_ref}")
print(f"  Hypothesis+Vanilla: {(v[:120])}...")
print(f"  Direct+Vanilla    : {(v_d[:120])}...")
print(f"  Hypothesis+SEVE   : {(final_h[:120])}...")
