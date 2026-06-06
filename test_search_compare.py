"""Compare 3 search preprocessing approaches × 5 questions.
All use Direct+Vanilla+gap (cost-effective baseline).
Approaches: direct / keyword / decompose
"""
import json, sys, time
from pathlib import Path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.bing_client import search
from src.model_client import generate, get_usage, reset_usage
from src.seve import (
    gap_fill_search, _extract_constraints, decompose_question,
)
from src.utils import load_json, save_json, format_search_results_numbered

all_qs = load_json("data/browsecomp-zh-decrypted.json")
cache = load_json("data/search_cache.json")

# 5 questions across domains
QUESTIONS = [
    (6,  "BC006", "艺术"),
    (8,  "BC008", "音乐"),
    (11, "BC011", "地理"),
    (22, "BC022", "历史"),
    (70, "BC070", "地理"),
]

def recall_score(results, answer):
    ans = answer.strip().lower()
    hits = 0
    for r in results:
        if ans in r.get("title", "").lower() or ans in r.get("snippet", "").lower():
            hits += 1
    return hits > 0, hits, len(results)

def check_match(answer, ref):
    return "YES" if ref.strip().lower() in answer.lower() else "?"

def search_direct(question, top_k=8):
    return search(question, top_k=top_k)

def search_keyword(question, top_k=8):
    kw = _extract_constraints(question)
    if not kw:
        return search(question, top_k=top_k)
    print(f"      Keywords: {kw[:100]}")
    return search(kw, top_k=top_k)

def search_decompose(question, top_k=8):
    sub_qs = decompose_question(question)
    if not sub_qs:
        return search(question, top_k=top_k)
    all_results = []
    seen = set()
    for sq in sub_qs:
        print(f"      Sub-query: {sq[:80]}")
        results = search(sq, top_k=max(3, top_k // len(sub_qs)))
        for r in results:
            url = r.get("url", "")
            if url and url not in seen:
                seen.add(url)
                all_results.append(r)
    return all_results

def run_with_gap(question, results, label):
    """Direct+Vanilla+gap-fill, return answer and metrics."""
    ctx = format_search_results_numbered(results)
    answer = generate(
        f"你是一个问答助手。请基于以下搜索结果为用户问题提供准确、简洁的回答。\n\n"
        f"搜索结果：\n{ctx}\n\n用户问题：{question}\n\n请回答："
    )
    gap = gap_fill_search(question, answer, results, max_rounds=2, top_k=5)
    if gap:
        results = results + gap
        ctx = format_search_results_numbered(results)
        answer = generate(
            f"你是一个问答助手。请基于以下搜索结果为用户问题提供准确、简洁的回答。\n\n"
            f"搜索结果：\n{ctx}\n\n用户问题：{question}\n\n请回答："
        )
    usage = get_usage()
    return answer, results, usage

# ── Run ──
results_table = []

for idx, qid, topic in QUESTIONS:
    q = all_qs[idx]
    question = q["Question"]
    answer_ref = q["Answer"]

    print(f"\n{'='*70}")
    print(f"[{qid}] [{topic}] {question[:100]}...")
    print(f"Ref: {answer_ref}")

    row = {"qid": qid, "ref": answer_ref}

    for approach, search_fn in [
        ("direct", search_direct),
        ("keyword", search_keyword),
        ("decompose", search_decompose),
    ]:
        print(f"\n  -- {approach} --")
        reset_usage()
        t0 = time.time()

        print(f"      Searching...", end=" ", flush=True)
        results = search_fn(question, top_k=8)
        srch_t = time.time() - t0
        rec_ok, hits, total = recall_score(results, answer_ref)
        print(f"{len(results)} results, recall={'YES' if rec_ok else 'NO'} ({srch_t:.1f}s)")

        answer, results, usage = run_with_gap(question, results, approach)
        elapsed = time.time() - t0
        match = check_match(answer, answer_ref)
        gap_added = len(results) - (total if total else len(results))

        print(f"      → {elapsed:.1f}s | {usage['input_tokens']}t in / {usage['output_tokens']}t out | match={match}")
        print(f"      Answer: {answer[:120]}")

        row[approach] = {
            "recall": rec_ok, "match": match,
            "time": elapsed, "tokens_in": usage["input_tokens"],
            "answer": answer[:200],
        }

    results_table.append(row)

# ── Summary ──
print(f"\n{'='*70}")
print(f"SUMMARY")
print(f"{'='*70}")
header = f"{'QID':<8} {'REF':<14} {'直接搜索':^20} {'关键词搜索':^20} {'问题拆解搜索':^20}"
print(header)
print(f"{'':8} {'':14} {'recall':>6} {'match':>6} {'time':>6}  {'recall':>6} {'match':>6} {'time':>6}  {'recall':>6} {'match':>6} {'time':>6}")

for row in results_table:
    d = row.get("direct", {})
    k = row.get("keyword", {})
    q = row.get("decompose", {})
    print(f"{row['qid']:<8} {row['ref']:<14} "
          f"{'Y' if d.get('recall') else 'N':>6} {d.get('match','?'):>6} {d.get('time',0):>5.0f}s  "
          f"{'Y' if k.get('recall') else 'N':>6} {k.get('match','?'):>6} {k.get('time',0):>5.0f}s  "
          f"{'Y' if q.get('recall') else 'N':>6} {q.get('match','?'):>6} {q.get('time',0):>5.0f}s")

# Win count
for app in ["direct", "keyword", "decompose"]:
    wins = sum(1 for r in results_table if r.get(app, {}).get("match") == "YES")
    recall_wins = sum(1 for r in results_table if r.get(app, {}).get("recall"))
    print(f"\n{app}: {wins}/{len(results_table)} match correct, {recall_wins}/{len(results_table)} recall hit")
