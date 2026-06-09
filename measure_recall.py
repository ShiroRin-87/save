"""Measure per-method search recall from cached search results.

Reads data/search_recall_cache.json (populated by test_5q_compare.py).
For each method A/B/C/D, checks if the reference answer string appears
in any search result's title/snippet/full_text.
"""
import json
from pathlib import Path

ROOT = Path(__file__).parent

# Load recall cache
recall_path = ROOT / "data" / "search_recall_cache.json"
if not recall_path.exists():
    print("recall_cache.json not found. Run test_5q_compare.py first.")
    exit(1)

recall = json.load(open(recall_path, encoding="utf-8"))

# Load questions for answer_ref
try:
    results_data = json.load(open(ROOT / "outputs" / "5q_compare_results.json", encoding="utf-8"))
    questions = {r["qid"]: r["ref"] for r in results_data.get("results", [])}
except Exception:
    questions = {}

# Also try loading from browsecomp data
if not questions:
    try:
        all_qs = json.load(open(ROOT / "data" / "browsecomp-zh-decrypted.json", encoding="utf-8"))
        # map question_id -> answer
        for q in all_qs:
            questions[q.get("question_id", "")] = q.get("Answer", "")
    except Exception:
        pass

def check_results(results, ref):
    """Check if reference answer appears in any result."""
    for r in results:
        full = (
            r.get("full_text", "") + " " +
            r.get("snippet", "") + " " +
            r.get("title", "")
        ).lower()
        if ref in full:
            return True
    return False

METHODS = ["A", "B", "C", "D", "E", "F"]

print(f"{'QID':<8} {'Ref':<20} | {'A':^5} {'B':^5} {'C':^5} {'D':^5} {'E':^5} {'F':^5}")
print(f"{'':8} {'':20} | {'':5} {'':5} {'':5} {'':5} {'':5} {'':5}")
print("-" * 50)

hits = {m: 0 for m in METHODS}
total = 0

for qid, method_results in sorted(recall.items()):
    ref = questions.get(qid, "").strip().lower()
    if not ref or len(ref) < 2:
        continue
    total += 1

    statuses = {}
    for m in METHODS:
        results = method_results.get(m, [])
        ok = check_results(results, ref)
        statuses[m] = "HIT" if ok else "MISS"
        if ok:
            hits[m] += 1

    print(f"{qid:<8} {ref[:18]:<20} | {statuses['A']:^5} {statuses['B']:^5} {statuses['C']:^5} {statuses['D']:^5} {statuses['E']:^5} {statuses['F']:^5}")

print("-" * 50)
print(f"\nPer-method recall ({total} questions):")
print(f"{'Method':<12} {'Recall':<12} {'Description'}")
print(f"{'':12} {'':12} {'':}")
for m in METHODS:
    names = {"A": "A: Direct", "B": "B: +gap", "C": "C: Hypo+gap", "D": "D: SEVE+gap", "E": "E: +Fallback", "F": "F: Chain"}
    pct = f"{hits[m]}/{total} = {hits[m]/total*100:.0f}%" if total else "N/A"
    print(f"{names[m]:<12} {pct:<12}")
