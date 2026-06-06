"""Quick: re-run SEVE+gap on BC005 using cached search results, print full output."""
import json, sys, time
from pathlib import Path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.model_client import generate, get_usage, reset_usage
from src.seve import (
    gap_fill_search, extract_claims, generate_answer,
    verify_claims, apply_verification,
)
from src.utils import load_json, format_search_results_numbered

all_qs = load_json("data/browsecomp-zh-decrypted.json")
q = all_qs[5]
qid = "BC005"
question = q["Question"]
answer_ref = q["Answer"]

print(f"[{qid}] {question}")
print(f"Ref: {answer_ref}")

# Use the search cache from the earlier run + hypothesis search
cache = load_json("data/search_cache.json")

# Re-do hypothesis-driven search to get full results
from src.seve import hypothesis_driven_search
reset_usage()
t0 = time.time()

print("\n--- Hypothesis-driven search ---")
results = hypothesis_driven_search(question, n_hypotheses=5)

# SEVE
print("\n--- SEVE extraction ---")
claims = extract_claims(results, question)
print(f"Claims: {len(claims)}")
for c in claims:
    print(f"  [{c['id']}] {c['claim'][:150]}")

answer, cmap = generate_answer(question, claims)
print(f"\n--- Initial Answer ---\n{answer}")

# Gap-fill
print("\n--- Gap-fill ---")
gap = gap_fill_search(question, answer, results, max_rounds=3, top_k=5)
if gap:
    results = results + gap
    claims = extract_claims(results, question)
    print(f"\nClaims after gap-fill: {len(claims)}")
    for c in claims:
        print(f"  [{c['id']}] {c['claim'][:150]}")
    answer, cmap = generate_answer(question, claims)

print(f"\n--- Final Answer ---\n{answer}")

# Verify
print("\n--- Verification ---")
verifs = verify_claims(answer, claims, results)
for v in verifs:
    print(f"  [{v['claim_id']}] {v['verdict']:>8}: {v['claim_text'][:100]}")

final = apply_verification(answer, verifs)
print(f"\n--- Final (after verification) ---\n{final}")
