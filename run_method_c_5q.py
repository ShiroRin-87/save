"""Run Method C (Hypothesis + Vanilla + Gap-fill) on 5 comparison questions.
Uses D's cached search results (which already include hypothesis + gap-fill expansion).
Method C = vanilla generation on those results (no SEVE extraction/verification).
"""
import json, sys, time
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.model_client import generate, get_usage, reset_usage
from src.utils import format_search_results_numbered

QUESTIONS = ["BC016", "BC021", "BC031", "BC050", "BC074"]
CACHE_DIR = ROOT / "data" / "adf_test_cache"

GEMINI_INPUT_COST  = 0.10 / 1_000_000
GEMINI_OUTPUT_COST = 0.40 / 1_000_000


def normalize_results(results):
    for i, r in enumerate(results, 1):
        r.setdefault("rank", i)
        r.setdefault("full_text", r.get("snippet", ""))
        r.setdefault("full_text_source", r.get("full_text_src", "snippet"))
        r.setdefault("full_text_len", r.get("full_text_len", len(r.get("full_text", ""))))
        r.setdefault("jina_quality", "ok")
    return results


def recall_score(results, answer_ref):
    ans = answer_ref.strip().lower()
    for r in results:
        body = r.get("full_text", r.get("snippet", ""))
        if ans in body.lower() or ans in r.get("snippet", "").lower() or ans in r.get("title", "").lower():
            return True
    return False


def check_accuracy(answer, ref):
    return ref.strip().lower() in answer.lower()


def run_method_c(qid, question, ref, search_results):
    trace = []
    reset_usage()

    # ── Vanilla generation from hypothesis-driven search results ──
    t_start = time.time()
    ctx = format_search_results_numbered(search_results)
    prompt = (
        "你是一个问答助手。请基于以下搜索结果为用户问题提供准确、简洁的回答。回答要尽量简洁。\n\n"
        f"搜索结果：\n{ctx}\n\n用户问题：{question}\n\n请回答："
    )
    answer = generate(prompt)
    t_gen = time.time() - t_start
    usage = get_usage()

    trace.append({
        "step": "C_generate",
        "wall_s": round(t_gen, 1),
        "tokens_in": usage["input_tokens"],
        "tokens_out": usage["output_tokens"],
    })

    rec = recall_score(search_results, ref)
    acc = check_accuracy(answer, ref)
    cost = usage["input_tokens"] * GEMINI_INPUT_COST + usage["output_tokens"] * GEMINI_OUTPUT_COST

    return {
        "method": "C",
        "qid": qid,
        "question": question,
        "ref": ref,
        "answer": answer,
        "accuracy": "Y" if acc else "N",
        "recall": rec,
        "total_wall_s": round(t_gen, 1),
        "llm_calls": usage["calls"],
        "search_calls": 0,  # reused cached results
        "tokens_in": usage["input_tokens"],
        "tokens_out": usage["output_tokens"],
        "cost_usd": round(cost, 5),
        "n_search_results": len(search_results),
        "trace": trace,
        "_note": "gap-fill skipped (SerpAPI rate limited); using D's already-expanded search results",
    }


if __name__ == "__main__":
    results = []
    for qid in QUESTIONS:
        cache_path = CACHE_DIR / f"{qid}_cache.json"
        if not cache_path.exists():
            print(f"SKIP {qid}: cache not found")
            continue

        cache = json.load(open(cache_path, encoding="utf-8"))
        question = cache["question"]
        ref = cache["ref"]
        topic = cache.get("topic", "?")

        d_data = cache.get("methods", {}).get("D", {})
        search_results = d_data.get("search_results", [])
        if not search_results:
            a_data = cache.get("methods", {}).get("A", {})
            search_results = a_data.get("search_results", [])

        search_results = normalize_results(search_results)

        print(f"\n{'='*70}")
        print(f"[{qid}] [{topic}] {question[:100]}...")
        print(f"Ref: {ref} | Cached search results: {len(search_results)} pages")
        print(f"{'='*70}")

        if not search_results:
            print("  No search results, skip")
            continue

        try:
            result = run_method_c(qid, question, ref, search_results)
            results.append(result)
            print(f"  Answer: {result['answer'][:250]}")
            print(f"  >> Accuracy: {result['accuracy']} | Recall: {result['recall']} | "
                  f"Time: {result['total_wall_s']:.1f}s | "
                  f"Tokens: {result['tokens_in']}/{result['tokens_out']} | "
                  f"Cost: ${result['cost_usd']:.4f}")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # ── Save ──
    out_path = ROOT / "outputs" / "method_c_5q_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {out_path}")

    # ── Summary ──
    print(f"\n{'QID':<8} {'Ref':<14} {'Acc':>5} {'Recall':>7} {'Time':>7} {'LLM':>5} {'TokIn':>8} {'TokOut':>8} {'Cost':>8}")
    for r in results:
        print(f"{r['qid']:<8} {r['ref']:<14} {r['accuracy']:>5} {str(r['recall']):>7} {r['total_wall_s']:>6.0f}s {r['llm_calls']:>5} {r['tokens_in']:>8} {r['tokens_out']:>8} ${r['cost_usd']:>7.4f}")

    acc = sum(1 for r in results if r['accuracy'] == 'Y')
    rec = sum(1 for r in results if r['recall'])
    print(f"\nAccuracy: {acc}/{len(results)} | Recall: {rec}/{len(results)}")
