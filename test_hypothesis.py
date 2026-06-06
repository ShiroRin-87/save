"""Full-pipeline comparison with metrics: time, recall, tokens, cost.
Uses needs_hypothesis_search to decide when hypothesis search is worth it.
Uses iterative gap-fill for multi-hop questions.
"""
import json, sys, time
from pathlib import Path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.bing_client import search
from src.model_client import generate, get_usage, reset_usage
from src.seve import (
    hypothesis_driven_search, gap_fill_search, needs_hypothesis_search,
    extract_claims, generate_answer, verify_claims, apply_verification,
)
from src.utils import load_json, save_json, format_search_results_numbered

# ── Config ──
TEST_IDX = 5  # BC005 = 宁波奉化
GEMINI_INPUT_COST = 0.10 / 1_000_000
GEMINI_OUTPUT_COST = 0.40 / 1_000_000

all_qs = load_json("data/browsecomp-zh-decrypted.json")
q = all_qs[TEST_IDX]
qid = f"BC{TEST_IDX:03d}"
question = q["Question"]
answer_ref = q["Answer"]
topic = q["Topic"]

print(f"[{qid}] [{topic}] {question}")
print(f"Ref: {answer_ref}")


def recall_score(results, answer):
    ans = answer.strip().lower()
    hits = 0
    for r in results:
        if ans in r.get("title", "").lower() or ans in r.get("snippet", "").lower():
            hits += 1
    return hits > 0, hits, len(results)


def check_match(answer, ref):
    return "YES" if ref.strip().lower() in answer.lower() else "?"


def format_row(label, t, usage, cost, recall_ok, match):
    return f"{label:<35} {t:>7.1f}s {usage['input_tokens']:>10}/{usage['output_tokens']:<9} ${cost:>7.4f} {'YES' if recall_ok else 'NO':>8} {match:>8}"

rows = {}


# ====================================================================
# A) Direct Search + Vanilla RAG
# ====================================================================
print(f"\n{'='*60}")
print("A) DIRECT SEARCH + VANILLA RAG")
print(f"{'='*60}")
reset_usage()
t0 = time.time()

results_a = search(question)
ctx_a = format_search_results_numbered(results_a)
answer_a = generate(
    f"你是一个问答助手。请基于以下搜索结果为用户问题提供准确、简洁的回答。回答要尽量简洁。\n\n"
    f"搜索结果：\n{ctx_a}\n\n用户问题：{question}\n\n请回答："
)
elapsed_a = time.time() - t0
usage_a = get_usage()
cost_a = usage_a["input_tokens"] * GEMINI_INPUT_COST + usage_a["output_tokens"] * GEMINI_OUTPUT_COST
recall_ok_a, hits_a, total_a = recall_score(results_a, answer_ref)
match_a = check_match(answer_a, answer_ref)

print(f"Time: {elapsed_a:.1f}s | Tokens: {usage_a['input_tokens']}/{usage_a['output_tokens']} | Cost: ${cost_a:.4f}")
print(f"Recall: {hits_a}/{total_a} pages ({'YES' if recall_ok_a else 'NO'}) | Match: {match_a}")
print(f"Answer: {answer_a[:250]}")
rows["A) Direct+Vanilla"] = (elapsed_a, usage_a, cost_a, recall_ok_a, match_a)


# ====================================================================
# B) Direct + Vanilla + Gap-fill
# ====================================================================
print(f"\n{'='*60}")
print("B) DIRECT + VANILLA + GAP-FILL")
print(f"{'='*60}")
reset_usage()
t0 = time.time()

results_b = list(results_a)  # reuse direct search results
ctx_b = format_search_results_numbered(results_b)
answer_b0 = generate(
    f"你是一个问答助手。请基于以下搜索结果为用户问题提供准确、简洁的回答。回答要尽量简洁。\n\n"
    f"搜索结果：\n{ctx_b}\n\n用户问题：{question}\n\n请回答："
)
# Iterative gap-fill
gap_b = gap_fill_search(question, answer_b0, results_b, max_rounds=3, top_k=5)
if gap_b:
    results_b = results_b + gap_b
    ctx_b = format_search_results_numbered(results_b)
    answer_b = generate(
        f"你是一个问答助手。请基于以下搜索结果为用户问题提供准确、简洁的回答。\n\n"
        f"搜索结果：\n{ctx_b}\n\n用户问题：{question}\n\n请回答："
    )
else:
    answer_b = answer_b0

elapsed_b = time.time() - t0
usage_b = get_usage()
cost_b = usage_b["input_tokens"] * GEMINI_INPUT_COST + usage_b["output_tokens"] * GEMINI_OUTPUT_COST
recall_ok_b, hits_b, total_b = recall_score(results_b, answer_ref)
match_b = check_match(answer_b, answer_ref)

print(f"Time: {elapsed_b:.1f}s | Tokens: {usage_b['input_tokens']}/{usage_b['output_tokens']} | Cost: ${cost_b:.4f}")
print(f"Gap-fill added: {len(gap_b)} pages | Recall: {'YES' if recall_ok_b else 'NO'} | Match: {match_b}")
print(f"Answer: {answer_b[:250]}")
rows["B) Direct+Vanilla+gap"] = (elapsed_b, usage_b, cost_b, recall_ok_b, match_b)


# ====================================================================
# C) Hypothesis-Driven Search + Vanilla + Gap-fill
# ====================================================================
print(f"\n{'='*60}")
print("C) HYPOTHESIS-DRIVEN + VANILLA + GAP-FILL")
print(f"{'='*60}")
reset_usage()
t0 = time.time()

# Check if hypothesis search is actually needed
need_hyp = needs_hypothesis_search(results_a, question)
print(f"needs_hypothesis_search: {'YES' if need_hyp else 'NO'}")

if need_hyp:
    results_h = hypothesis_driven_search(question, n_hypotheses=5)
else:
    print("  Direct results look sufficient, skipping hypothesis search")
    results_h = list(results_a)

ctx_h = format_search_results_numbered(results_h)
answer_h0 = generate(
    f"你是一个问答助手。请基于以下搜索结果为用户问题提供准确、简洁的回答。回答要尽量简洁。\n\n"
    f"搜索结果：\n{ctx_h}\n\n用户问题：{question}\n\n请回答："
)
gap_h = gap_fill_search(question, answer_h0, results_h, max_rounds=3, top_k=5)
if gap_h:
    results_h = results_h + gap_h
    ctx_h = format_search_results_numbered(results_h)
    answer_h = generate(
        f"你是一个问答助手。请基于以下搜索结果为用户问题提供准确、简洁的回答。\n\n"
        f"搜索结果：\n{ctx_h}\n\n用户问题：{question}\n\n请回答："
    )
else:
    answer_h = answer_h0

elapsed_h = time.time() - t0
usage_h = get_usage()
cost_h = usage_h["input_tokens"] * GEMINI_INPUT_COST + usage_h["output_tokens"] * GEMINI_OUTPUT_COST
recall_ok_h, hits_h, total_h = recall_score(results_h, answer_ref)
match_h = check_match(answer_h, answer_ref)

print(f"Time: {elapsed_h:.1f}s | Tokens: {usage_h['input_tokens']}/{usage_h['output_tokens']} | Cost: ${cost_h:.4f}")
print(f"Hypothesis: {'YES' if need_hyp else 'NO'} | Gap-fill added: {len(gap_h)} | Recall: {'YES' if recall_ok_h else 'NO'} | Match: {match_h}")
print(f"Answer: {answer_h[:250]}")
rows["C) Hypothesis+Vanilla+gap"] = (elapsed_h, usage_h, cost_h, recall_ok_h, match_h)


# ====================================================================
# D) Hypothesis + SEVE + Gap-fill
# ====================================================================
print(f"\n{'='*60}")
print("D) HYPOTHESIS + SEVE + GAP-FILL")
print(f"{'='*60}")
reset_usage()
t0 = time.time()

claims_h = extract_claims(results_h, question)
answer_s0, cmap = generate_answer(question, claims_h)

gap_s = gap_fill_search(question, answer_s0, results_h, max_rounds=3, top_k=5)
if gap_s:
    results_h = results_h + gap_s
    claims_h = extract_claims(results_h, question)
    answer_seve, cmap = generate_answer(question, claims_h)
else:
    answer_seve = answer_s0

verifs = verify_claims(answer_seve, claims_h, results_h)
answer_seve_final = apply_verification(answer_seve, verifs)

elapsed_s = time.time() - t0
usage_s = get_usage()
cost_s = usage_s["input_tokens"] * GEMINI_INPUT_COST + usage_s["output_tokens"] * GEMINI_OUTPUT_COST
yes_n = sum(1 for v in verifs if v['verdict'] == 'YES')
partial_n = sum(1 for v in verifs if v['verdict'] == 'PARTIAL')
no_n = sum(1 for v in verifs if v['verdict'] == 'NO')
match_s = check_match(answer_seve_final, answer_ref)

print(f"Time: {elapsed_s:.1f}s | Tokens: {usage_s['input_tokens']}/{usage_s['output_tokens']} | Cost: ${cost_s:.4f}")
print(f"Claims: {len(claims_h)} | Verdicts: {yes_n}Y/{partial_n}P/{no_n}N | Match: {match_s}")
print(f"Answer: {answer_seve_final[:250]}")
rows["D) Hypothesis+SEVE+gap"] = (elapsed_s, usage_s, cost_s, True, match_s)


# ====================================================================
# SUMMARY
# ====================================================================
print(f"\n{'='*70}")
print(f"SUMMARY: {qid} | Ref={answer_ref}")
print(f"{'='*70}")
print(f"{'Method':<35} {'Time':>8} {'Tokens(in/out)':>22} {'Cost':>8} {'Recall':>8} {'Match':>8}")
for label, (t, u, c, r, m) in rows.items():
    print(format_row(label, t, u, c, r, m))
print()
print(f"Hypothesis search was: {'NEEDED' if need_hyp else 'NOT NEEDED (skipped)'}")
print(f"Cost estimated at $0.10/M input, $0.40/M output")
