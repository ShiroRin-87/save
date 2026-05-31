"""Test SEVE on custom question: Saga Planets leaving Visual Arts."""
import json, sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.bing_client import search
from src.seve import extract_claims, generate_answer, verify_claims, apply_verification
from src.utils import load_search_cache, save_json

QID = "CUSTOM_SAGA"
QUESTION = "SAGA PLANETS是在哪一部作品后离开Visual Arts的？"

# Step 1: Search
print(f"Searching: {QUESTION}")
results = search(QUESTION)
print(f"  Got {len(results)} results")

cache = load_search_cache()
cache[QID] = {
    "query": QUESTION,
    "timestamp": __import__("time").strftime("%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime()),
    "results": results,
}
save_json(cache, "data/search_cache.json")

# Step 2: SEVE pipeline
print("\n=== STEP 1: Structured Extraction ===")
claims = extract_claims(results)
print(f"Extracted {len(claims)} claims:")
for c in claims:
    print(f"  [{c['id']}] {c['claim'][:120]}...")

print("\n=== STEP 2: Calibrated Generation ===")
answer, citation_map = generate_answer(QUESTION, claims)
print(f"Answer:\n{answer}")

print("\n=== STEP 3: Reverse Verification ===")
verifications = verify_claims(answer, claims, results)
for v in verifications:
    print(f"  [{v['claim_id']}] {v['verdict']}")

print("\n=== STEP 4: Final Answer ===")
final = apply_verification(answer, verifications)
print(final)

# Save
save_json({QID: {"question": QUESTION, "claims": claims, "num_claims": len(claims)}}, "outputs/seve_custom/structured_tables.json")
save_json({QID: {"question": QUESTION, "answer": answer, "citation_map": citation_map}}, "outputs/seve_custom/generated_answers.json")
save_json({QID: {"question": QUESTION, "verifications": verifications}}, "outputs/seve_custom/verification_results.json")
save_json({QID: {"question": QUESTION, "answer": final, "method": "seve", "search_available": bool(results)}}, "outputs/seve_custom/final_answers.json")

print("\nDone.")
