"""Quick test: JSON extraction on the anemoi question that previously crashed."""
import json, sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.seve import extract_claims, generate_answer, verify_claims
from src.utils import load_search_cache

QID = "CUSTOM_ANEMOI2"
QUESTION = "新岛夕在anemoi中负责了哪些线路的内容？"

cache = load_search_cache()
results = cache.get(QID, {}).get("results", [])

print("=== JSON Extraction ===")
claims = extract_claims(results, QUESTION)
print(f"Extracted {len(claims)} claims:")
for c in claims:
    print(f"  [{c['id']}] {c['claim'][:100]}...")

print("\n=== Generation ===")
answer, cmap = generate_answer(QUESTION, claims)
print(answer)

print("\n=== Verification (batch) ===")
verifications = verify_claims(answer, claims, results)
for v in verifications:
    print(f"  [{v['claim_id']}] {v['verdict']}")

print(f"\nVerification calls: 1 batch (was {len(verifications)} serial calls)")
