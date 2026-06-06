"""4 pipeline methods × 5 questions comparison."""
import json, sys, time
from pathlib import Path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.bing_client import search
from src.model_client import generate, get_usage, reset_usage
from src.seve import (
    hypothesis_driven_search, gap_fill_search, needs_hypothesis_search,
    extract_claims, generate_answer, verify_claims, apply_verification,
    fallback_reasoning,
)
from src.utils import load_json, save_json, format_search_results_numbered

all_qs = load_json("data/browsecomp-zh-decrypted.json")
cache = load_json("data/search_cache.json")

# Per-method search recall cache: {qid: {A: [results], B: [results], ...}}
RECALL_CACHE_PATH = "data/search_recall_cache.json"
recall_cache = load_json(RECALL_CACHE_PATH)

QUESTIONS = [
    (9,  "BC009", "音乐"),
    (13, "BC013", "艺术"),
    (15, "BC015", "电子游戏"),
    (25, "BC025", "医学"),
    (30, "BC030", "影视"),
]

RESULTS = []

for idx, qid, topic in QUESTIONS:
    q = all_qs[idx]
    question = q["Question"]
    answer_ref = q["Answer"]

    print(f"\n{'='*70}")
    print(f"[{qid}] [{topic}] {question[:100]}...")
    print(f"Ref: {answer_ref}")

    row = {"qid": qid, "ref": answer_ref}

    # ── A: Direct+Vanilla ──
    print(f"\n  A) Direct+Vanilla...", end=" ", flush=True)
    reset_usage(); t0 = time.time()
    results_a = search(question)
    recall_cache.setdefault(qid, {})["A"] = [
        {"title": r["title"], "url": r["url"], "snippet": r["snippet"]}
        for r in results_a
    ]
    ctx = format_search_results_numbered(results_a)
    answer = generate(
        f"你是一个问答助手。请基于以下搜索结果为用户问题提供准确、简洁的回答。\n\n"
        f"搜索结果：\n{ctx}\n\n用户问题：{question}\n\n请回答："
    )
    u = get_usage()
    row["A_time"] = time.time()-t0; row["A_tok_in"] = u["input_tokens"]; row["A_tok_out"] = u["output_tokens"]
    row["A_match"] = "Y" if answer_ref.strip().lower() in answer.lower() else "N"
    print(f"{row['A_time']:.0f}s match={row['A_match']}")

    # ── B: Direct+Vanilla+gap ──
    print(f"  B) Direct+gap...", end=" ", flush=True)
    reset_usage(); t0 = time.time()
    results_b = list(results_a)
    ctx = format_search_results_numbered(results_b)
    answer = generate(
        f"你是一个问答助手。请基于以下搜索结果为用户问题提供准确、简洁的回答。\n\n"
        f"搜索结果：\n{ctx}\n\n用户问题：{question}\n\n请回答："
    )
    gap = gap_fill_search(question, answer, results_b, max_rounds=2, top_k=5)
    if gap:
        results_b = results_b + gap
        recall_cache.setdefault(qid, {})["B"] = [
            {"title": r["title"], "url": r["url"], "snippet": r["snippet"]}
            for r in results_b
        ]
        ctx = format_search_results_numbered(results_b)
        answer = generate(
            f"你是一个问答助手。请基于以下搜索结果为用户问题提供准确、简洁的回答。\n\n"
            f"搜索结果：\n{ctx}\n\n用户问题：{question}\n\n请回答："
        )
    u = get_usage()
    row["B_time"] = time.time()-t0; row["B_tok_in"] = u["input_tokens"]; row["B_tok_out"] = u["output_tokens"]
    row["B_match"] = "Y" if answer_ref.strip().lower() in answer.lower() else "N"
    row["B_gap"] = len(gap)
    print(f"{row['B_time']:.0f}s gap={row['B_gap']} match={row['B_match']}")

    # ── C: Hypothesis+Vanilla+gap ──
    print(f"  C) Hypothesis+gap...", end=" ", flush=True)
    reset_usage(); t0 = time.time()
    need = needs_hypothesis_search(results_a, question)
    if need:
        results_c = hypothesis_driven_search(question, n_hypotheses=5)
    else:
        print("skip-hypo", end=" ")
        results_c = list(results_a)
    recall_cache.setdefault(qid, {})["C"] = [
        {"title": r["title"], "url": r["url"], "snippet": r["snippet"]}
        for r in results_c
    ]
    ctx = format_search_results_numbered(results_c)
    answer = generate(
        f"你是一个问答助手。请基于以下搜索结果为用户问题提供准确、简洁的回答。\n\n"
        f"搜索结果：\n{ctx}\n\n用户问题：{question}\n\n请回答："
    )
    gap = gap_fill_search(question, answer, results_c, max_rounds=2, top_k=5)
    if gap:
        results_c = results_c + gap
        ctx = format_search_results_numbered(results_c)
        answer = generate(
            f"你是一个问答助手。请基于以下搜索结果为用户问题提供准确、简洁的回答。\n\n"
            f"搜索结果：\n{ctx}\n\n用户问题：{question}\n\n请回答："
        )
    u = get_usage()
    row["C_time"] = time.time()-t0; row["C_tok_in"] = u["input_tokens"]; row["C_tok_out"] = u["output_tokens"]
    row["C_match"] = "Y" if answer_ref.strip().lower() in answer.lower() else "N"
    row["C_hypo"] = "Y" if need else "N"
    print(f"hypo={row['C_hypo']} gap={len(gap)} match={row['C_match']}")

    # ── D: Hypothesis+SEVE+gap ──
    print(f"  D) Hypothesis+SEVE+gap...", end=" ", flush=True)
    reset_usage(); t0 = time.time()
    claims = extract_claims(results_c, question)
    answer, cmap = generate_answer(question, claims)
    gap = gap_fill_search(question, answer, results_c, max_rounds=2, top_k=5)
    if gap:
        results_c = results_c + gap
        claims = extract_claims(results_c, question)
        answer, cmap = generate_answer(question, claims)
    recall_cache.setdefault(qid, {})["D"] = [
        {"title": r["title"], "url": r["url"], "snippet": r["snippet"]}
        for r in results_c
    ]
    verifs = verify_claims(answer, claims, results_c)
    answer_final = apply_verification(answer, verifs)
    u = get_usage()
    yes_n = sum(1 for v in verifs if v['verdict'] == 'YES')
    no_n = sum(1 for v in verifs if v['verdict'] == 'NO')
    row["D_time"] = time.time()-t0; row["D_tok_in"] = u["input_tokens"]; row["D_tok_out"] = u["output_tokens"]
    row["D_match"] = "Y" if answer_ref.strip().lower() in answer_final.lower() else "N"
    row["D_claims"] = len(claims); row["D_verify"] = f"{yes_n}Y/{no_n}N"
    print(f"claims={row['D_claims']} verify={row['D_verify']} match={row['D_match']}")

    # ── E: Hypothesis+SEVE+gap+fallback ──
    print(f"  E) +Fallback...", end=" ", flush=True)
    t0 = time.time()
    answer_final_e, fb_trig = fallback_reasoning(question, answer_final, verifs)
    row["E_time"] = time.time()-t0
    row["E_match"] = "Y" if answer_ref.strip().lower() in answer_final_e.lower() else "N"
    row["E_fb"] = "Y" if fb_trig else "N"
    # E uses same search results as D (no new searches)
    recall_cache.setdefault(qid, {})["E"] = recall_cache.setdefault(qid, {}).get("D", [])
    print(f"fb={row['E_fb']} match={row['E_match']}")

    RESULTS.append(row)
    # Incremental save — preserve progress on crash
    save_json(RESULTS, "outputs/5q_compare_results.json")
    save_json(recall_cache, RECALL_CACHE_PATH)

# ── Summary Table ──
print(f"\n{'='*70}")
print(f"SUMMARY: 4 Methods × 5 Questions")
print(f"{'='*70}")
print(f"{'QID':<8} {'REF':<10} | {'A:Direct':^10} | {'B:+gap':^12} | {'C:Hypo+gap':^12} | {'D:SEVE+gap':^12} | {'E:+Fallback':^10} |")
print(f"{'':8} {'':10} | {'match':>4} {'time':>5} | {'match':>4} {'time':>5} {'gap':>4} | {'hypo':>4} {'match':>4} {'time':>5} | {'claims':>4} {'verif':>6} {'match':>4} | {'fb':>4} {'match':>4} |")

wins = {m: 0 for m in "ABCDE"}
for r in RESULTS:
    print(f"{r['qid']:<8} {r['ref']:<10} | "
          f"{r['A_match']:>4} {r['A_time']:>5.0f}s | "
          f"{r['B_match']:>4} {r['B_time']:>5.0f}s {r['B_gap']:>4} | "
          f"{r['C_hypo']:>4} {r['C_match']:>4} {r['C_time']:>5.0f}s | "
          f"{r['D_claims']:>4} {r['D_verify']:>6} {r['D_match']:>4} | "
          f"{r.get('E_fb','?'):>4} {r.get('E_match','?'):>4} |")
    for m in "ABCDE":
        if r.get(f"{m}_match") == "Y":
            wins[m] += 1

total = len(QUESTIONS)
print(f"\n{'':18} | {'A='+str(wins['A'])+'/'+str(total):^10} | {'B='+str(wins['B'])+'/'+str(total):^12} | {'C='+str(wins['C'])+'/'+str(total):^12} | {'D='+str(wins['D'])+'/'+str(total):^12} | {'E='+str(wins['E'])+'/'+str(total):^10} |")

# Final save with summary
save_json({"results": RESULTS, "wins": wins, "total": total}, "outputs/5q_compare_results.json")
save_json(recall_cache, RECALL_CACHE_PATH)
print(f"\nResults saved to outputs/5q_compare_results.json")
print(f"Recall cache saved to {RECALL_CACHE_PATH}")
