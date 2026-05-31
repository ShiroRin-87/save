"""Test SEVE on a custom question about AB2 / Aino+Links."""
import json, sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.bing_client import search
from src.seve import extract_claims, generate_answer, verify_claims, apply_verification
from src.utils import load_search_cache, save_json

QID = "CUSTOM_AB2"
QUESTION = "AB2和哪个社团一起组建了新品牌并制作了创作彼女的恋爱公式这个游戏？"

# Step 1: Search
print(f"Searching: {QUESTION}")
results = search(QUESTION)
print(f"  Got {len(results)} results")

# Save to cache
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

# Save outputs
extracted_data = {QID: {"question": QUESTION, "claims": claims, "num_claims": len(claims)}}
gen_data = {QID: {"question": QUESTION, "answer": answer, "citation_map": citation_map}}
ver_data = {QID: {"question": QUESTION, "verifications": verifications}}
final_data = {QID: {"question": QUESTION, "answer": final, "method": "seve", "search_available": bool(results)}}

save_json(extracted_data, "outputs/seve_custom/structured_tables.json")
save_json(gen_data, "outputs/seve_custom/generated_answers.json")
save_json(ver_data, "outputs/seve_custom/verification_results.json")
save_json(final_data, "outputs/seve_custom/final_answers.json")

print("\nDone. Saved to outputs/seve_custom/")
