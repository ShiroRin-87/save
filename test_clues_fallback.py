"""Test _extract_clues results-based fallback on one question."""
import json, sys, time
from pathlib import Path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.bing_client import search
from src.seve import extract_claims, iterative_search, _extract_clues, _has_enough_info
from src.utils import load_json, save_json

all_qs = load_json("data/browsecomp-zh-decrypted.json")
cache = load_json("data/search_cache.json")

# BC022 = 王安石
q = all_qs[22]
qid = "BC022"
question = q["Question"]
answer_ref = q["Answer"]

print(f"[{qid}] {question}")
print(f"Ref: {answer_ref}")
print()

# Load or do initial search
cache_key = qid
if cache_key in cache and cache[cache_key].get("results"):
    results = cache[cache_key]["results"]
    print(f"Initial search: cached {len(results)} results")
else:
    print("Initial search...", end=" ", flush=True); t0 = time.time()
    results = search(question)
    cache[cache_key] = {"query": question, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"), "results": results}
    save_json(cache, "data/search_cache.json")
    print(f"{len(results)} results ({time.time()-t0:.1f}s)")

# Show what titles/snippets we have
print("\n--- Titles & Snippets ---")
for r in results[:5]:
    print(f"  [{r['rank']}] {r['title'][:100]}")
    print(f"      {r['snippet'][:150]}")

# Test _has_enough_info
print(f"\n_has_enough_info: {_has_enough_info(results, question)}")

# Test claim extraction
print("\n--- Extracting claims ---")
claims = extract_claims(results, question)
print(f"Got {len(claims)} claims")
for c in claims:
    print(f"  [{c['id']}] {c['claim'][:120]}")

# Test _extract_clues with BOTH paths
print("\n--- _extract_clues from claims ---")
clues_from_claims = _extract_clues(claims, question, None)
print(f"  -> {clues_from_claims[:200]}")

print("\n--- _extract_clues from results (simulating empty claims) ---")
clues_from_results = _extract_clues([], question, results)
print(f"  -> {clues_from_results[:200]}")

# Test iterative search
print("\n--- Iterative search ---")
n_before = len(results)
expanded = iterative_search(question, results, max_rounds=2)
print(f"Results: {n_before} -> {len(expanded)}")
