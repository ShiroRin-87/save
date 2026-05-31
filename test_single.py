"""Quick test: run SEVE on a single question (FQ010)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.bing_client import search
from src.seve import extract_claims, generate_answer, verify_claims, apply_verification
from src.utils import load_search_cache, save_json

QID = "FQ010"
QUESTION = "In what year did Hikaru Nakamura become the World Chess Champion?"

# Step 1: Ensure search cache has this question
cache = load_search_cache()
if QID not in cache or not cache[QID].get("results"):
    print(f"Searching for: {QUESTION}")
    results = search(QUESTION)
    cache[QID] = {
        "query": QUESTION,
        "timestamp": __import__("time").strftime("%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime()),
        "results": results,
    }
    save_json(cache, "data/search_cache.json")
    print(f"  Got {len(results)} results, cache updated.")
else:
    results = cache[QID]["results"]
    print(f"  Using cached {len(results)} results.")

# Step 2: SEVE pipeline
print("\n=== STEP 1: Structured Extraction ===")
claims = extract_claims(results)
print(f"Extracted {len(claims)} claims:")
for c in claims:
    print(f"  [{c['id']}] {c['claim'][:100]}...")

print("\n=== STEP 2: Calibrated Generation ===")
answer, citation_map = generate_answer(QUESTION, claims)
print(f"Answer:\n{answer}")

print("\n=== STEP 3: Reverse Verification ===")
verifications = verify_claims(answer, claims, results)
for v in verifications:
    print(f"  [{v['claim_id']}] {v['verdict']}")

print("\n=== STEP 4: Final Answer (after verification) ===")
final = apply_verification(answer, verifications)
print(final)

# Save single outputs
save_json({
    QID: {"question": QUESTION, "claims": claims, "num_claims": len(claims)}
}, "outputs/seve/structured_tables.json")

save_json({
    QID: {"question": QUESTION, "answer": answer, "citation_map": citation_map}
}, "outputs/seve/generated_answers.json")

save_json({
    QID: {"question": QUESTION, "verifications": verifications}
}, "outputs/seve/verification_results.json")

save_json({
    QID: {"question": QUESTION, "answer": final, "method": "seve", "search_available": bool(results)}
}, "outputs/seve/final_answers.json")

print("\nDone. Check outputs/seve/*.json")
