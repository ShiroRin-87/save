"""Extract metrics from all_results.json for the report."""
import json
from pathlib import Path

data = json.load(open("outputs/browsecomp_10/all_results.json", encoding="utf-8"))

print(f"{'QID':<8} {'Topic':<8} {'Ref':<20} | {'Vanilla':^12} | {'Self-RAG':^12} | {'SEVE':^12}")
print(f"{'':8} {'':8} {'':20} | {'match':>6} {'claims':>6} | {'match':>6} {'verify':>6} | {'match':>6} {'claims':>6} {'verify':>6}")

vanilla_wins = 0
self_rag_wins = 0
seve_wins = 0
total = 0

for qid, q in data.items():
    ref = q.get("answer_ref", "")
    topic = q.get("topic", "")

    # Vanilla
    v_answer = q.get("vanilla", "")
    v_match = "Y" if ref.strip().lower() in v_answer.lower() else "N"

    # Self-RAG
    sr = q.get("self_rag", {})
    sr_answer = sr.get("answer", "") if isinstance(sr, dict) else ""
    sr_match = "Y" if ref.strip().lower() in sr_answer.lower() else "N"
    sr_r1 = sr.get("round1", "") if isinstance(sr, dict) else ""
    sr_r2 = sr.get("round2", "") if isinstance(sr, dict) else ""

    # SEVE
    se = q.get("seve", {})
    se_answer = se.get("final", se.get("answer", "")) if isinstance(se, dict) else ""
    se_match = "Y" if ref.strip().lower() in se_answer.lower() else "N"
    se_claims = se.get("num_claims", len(se.get("claims", []))) if isinstance(se, dict) else 0
    se_verifs = se.get("verifications", []) if isinstance(se, dict) else []

    yes_n = sum(1 for v in se_verifs if v.get("verdict") == "YES")
    no_n = sum(1 for v in se_verifs if v.get("verdict") == "NO")

    print(f"{qid:<8} {topic:<8} {ref[:18]:<20} | {v_match:>6} {'':>6} | {sr_match:>6} {'':>6} | {se_match:>6} {se_claims:>6} {f'{yes_n}Y/{no_n}N':>6}")

    if v_match == "Y": vanilla_wins += 1
    if sr_match == "Y": self_rag_wins += 1
    if se_match == "Y": seve_wins += 1
    total += 1

print(f"\n{'':8} {'':8} {'':20} | {'':6} {'':6} | {'':6} {'':6} | {'':6} {'':6} {'':6}")
print(f"{'WINS':<8} {'':8} {'':20} | {vanilla_wins:>6}/{total} {'':>6} | {self_rag_wins:>6}/{total} {'':>6} | {seve_wins:>6}/{total} {'':>6} {'':>6}")
