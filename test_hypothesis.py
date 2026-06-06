"""Full-pipeline comparison with metrics: time, recall, tokens, cost.
Includes gap-fill search for multi-hop questions.
"""
import json, sys, time
from pathlib import Path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.bing_client import search
from src.model_client import generate, get_usage, reset_usage
from src.seve import (
    hypothesis_driven_search, gap_fill_search,
    extract_claims, generate_answer, verify_claims, apply_verification
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
    """Check if answer appears in titles or snippets."""
    ans = answer.strip().lower()
    hits = 0
    for r in results:
        title = r.get("title", "").lower()
        snippet = r.get("snippet", "").lower()
        if ans in title or ans in snippet:
            hits += 1
    return hits > 0, hits, len(results)


def check_match(answer, ref):
    return "YES" if ref.strip().lower() in answer.lower() else "?"


# ── Storage for summary ──
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
    f"你是一个问答助手。请基于以下搜索结果为用户问题提供准确、简洁的回答。回答要尽量简洁，如果无法确定请直接说。\n\n"
    f"搜索结果：\n{ctx_a}\n\n用户问题：{question}\n\n请回答："
)
elapsed_a = time.time() - t0
usage_a = get_usage()
cost_a = usage_a["input_tokens"] * GEMINI_INPUT_COST + usage_a["output_tokens"] * GEMINI_OUTPUT_COST
recall_ok_a, hits_a, total_a = recall_score(results_a, answer_ref)

print(f"Time: {elapsed_a:.1f}s | Tokens: {usage_a['input_tokens']}/{usage_a['output_tokens']} | Cost: ${cost_a:.4f}")
print(f"Recall: {hits_a}/{total_a} pages contain answer ({'YES' if recall_ok_a else 'NO'})")
print(f"Answer: {answer_a[:250]}")

rows["A) Direct+Vanilla"] = (elapsed_a, usage_a, cost_a, recall_ok_a, check_match(answer_a, answer_ref))


# ====================================================================
# B) Hypothesis-Driven Search + Vanilla RAG
# ====================================================================
print(f"\n{'='*60}")
print("B) HYPOTHESIS-DRIVEN SEARCH + VANILLA RAG")
print(f"{'='*60}")
reset_usage()
t0 = time.time()

results_h = hypothesis_driven_search(question, n_hypotheses=5)
ctx_h = format_search_results_numbered(results_h)
answer_h = generate(
    f"你是一个问答助手。请基于以下搜索结果为用户问题提供准确、简洁的回答。回答要尽量简洁，如果无法确定请直说。\n\n"
    f"搜索结果：\n{ctx_h}\n\n用户问题：{question}\n\n请回答："
)
elapsed_h1 = time.time() - t0
usage_b = get_usage()

# ── Gap-fill: if answer doesn't contain reference, do follow-up search ──
if answer_ref.strip().lower() not in answer_h.lower():
    print(f"\n  Answer incomplete, running gap-fill search...")
    gap_results = gap_fill_search(question, answer_h, results_h, top_k=5)
    if gap_results:
        results_h = results_h + gap_results
        ctx_h = format_search_results_numbered(results_h)
        answer_h = generate(
            f"你是一个问答助手。请基于以下搜索结果为用户问题提供准确、简洁的回答。\n\n"
            f"搜索结果：\n{ctx_h}\n\n用户问题：{question}\n\n请回答："
        )
elapsed_h = time.time() - t0
usage_h = get_usage()
cost_h = usage_h["input_tokens"] * GEMINI_INPUT_COST + usage_h["output_tokens"] * GEMINI_OUTPUT_COST
recall_ok_h, hits_h, total_h = recall_score(results_h, answer_ref)

print(f"Time: {elapsed_h:.1f}s | Tokens: {usage_h['input_tokens']}/{usage_h['output_tokens']} | Cost: ${cost_h:.4f}")
print(f"Recall: {hits_h}/{total_h} pages contain answer ({'YES' if recall_ok_h else 'NO'})")
print(f"Answer: {answer_h[:250]}")

rows["B) Hypothesis+Vanilla"] = (elapsed_h, usage_h, cost_h, recall_ok_h, check_match(answer_h, answer_ref))


# ====================================================================
# C) Hypothesis-Driven Search + SEVE
# ====================================================================
print(f"\n{'='*60}")
print("C) HYPOTHESIS-DRIVEN + SEVE")
print(f"{'='*60}")
reset_usage()
t0 = time.time()

claims_h = extract_claims(results_h, question)
answer_seve, cmap = generate_answer(question, claims_h)

# Gap-fill for SEVE too
if answer_ref.strip().lower() not in answer_seve.lower():
    print(f"\n  Answer incomplete, running gap-fill for SEVE...")
    gap_results = gap_fill_search(question, answer_seve, results_h, top_k=5)
    if gap_results:
        results_h = results_h + gap_results
        claims_h = extract_claims(results_h, question)
        answer_seve, cmap = generate_answer(question, claims_h)

verifs = verify_claims(answer_seve, claims_h, results_h)
answer_seve_final = apply_verification(answer_seve, verifs)

elapsed_s = time.time() - t0
usage_s = get_usage()
cost_s = usage_s["input_tokens"] * GEMINI_INPUT_COST + usage_s["output_tokens"] * GEMINI_OUTPUT_COST
yes_n = sum(1 for v in verifs if v['verdict'] == 'YES')
partial_n = sum(1 for v in verifs if v['verdict'] == 'PARTIAL')
no_n = sum(1 for v in verifs if v['verdict'] == 'NO')

print(f"Time: {elapsed_s:.1f}s | Tokens: {usage_s['input_tokens']}/{usage_s['output_tokens']} | Cost: ${cost_s:.4f}")
print(f"Claims: {len(claims_h)} | Verdicts: {yes_n} YES / {partial_n} PARTIAL / {no_n} NO")
print(f"Answer: {answer_seve_final[:250]}")

rows["C) Hypothesis+SEVE"] = (elapsed_s, usage_s, cost_s, True, check_match(answer_seve_final, answer_ref))


# ====================================================================
# SUMMARY
# ====================================================================
print(f"\n{'='*60}")
print(f"SUMMARY: {qid} | Ref={answer_ref}")
print(f"{'='*60}")
print(f"{'Method':<35} {'Time':>8} {'Tokens(in/out)':>22} {'Cost':>8} {'Recall':>8} {'Match':>8}")
for label, (t, u, c, r, m) in rows.items():
    tokens_str = f"{u['input_tokens']}/{u['output_tokens']}"
    print(f"{label:<35} {t:>7.1f}s {tokens_str:>22} ${c:>7.4f} {'YES' if r else 'NO':>8} {m:>8}")
print()
print("Cost estimated at $0.10/M input, $0.40/M output (Gemini Flash ~approx)")
