"""Test 4 questions across domains with all 4 methods, using multi-round iterative search."""
import json, sys, time
from pathlib import Path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.bing_client import search
from src.model_client import generate
from src.seve import (extract_claims, generate_answer, verify_claims,
                      apply_verification, iterative_search, extract_and_generate)
from src.utils import load_json, save_json, format_search_results_numbered

# BC011=地理/回车巷, BC022=历史/王安石, BC070=地理/青城山, BC080=音乐/梁博
SELECTED = [11, 22, 70, 80]
all_qs = load_json("data/browsecomp-zh-decrypted.json")
cache = load_json("data/search_cache.json") if (ROOT / "data/search_cache.json").exists() else {}

for idx in SELECTED:
    q = all_qs[idx]
    qid = f"BC{idx:03d}"
    question = q["Question"]
    answer_ref = q["Answer"]
    topic = q["Topic"]

    print(f"\n{'='*60}")
    print(f"[{qid}] [{topic}] {question[:120]}...")
    print(f"Ref: {answer_ref}")
    print(f"{'='*60}")

    # Phase 1: Initial search
    cache_key = qid
    if cache_key in cache and cache[cache_key].get("results"):
        results = cache[cache_key]["results"]
        print(f"Initial search: cached {len(results)} results")
    else:
        print(f"Initial search...", end=" ", flush=True); t0 = time.time()
        results = search(question)
        cache[cache_key] = {"query": question, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"), "results": results}
        save_json(cache, "data/search_cache.json")
        print(f"{len(results)} results ({time.time()-t0:.1f}s)")

    if not results:
        print("No results, skip")
        continue

    # Phase 2: Multi-round iterative search
    print(f"\n--- Multi-round Iterative Search (max 2 rounds) ---")
    n_before = len(results)
    results = iterative_search(question, results, max_rounds=2)
    if len(results) > n_before:
        cache[cache_key]["results"] = results
        cache[cache_key]["iterative_expanded"] = True
        save_json(cache, "data/search_cache.json")
        print(f"Expanded: {n_before} -> {len(results)} results")

    total_chars = sum(r.get("full_text_len", 0) for r in results)
    print(f"\nTotal context: {total_chars:,} chars across {len(results)} pages")

    # 1) Vanilla RAG
    print(f"\n--- Vanilla RAG ---"); t0 = time.time()
    ctx_raw = format_search_results_numbered(results)
    v = generate(f"你是一个问答助手。请基于以下搜索结果为用户问题提供准确、简洁的回答。\n\n搜索结果：\n{ctx_raw}\n\n用户问题：{question}\n\n请回答：")
    print(f"({time.time()-t0:.1f}s) {v[:300]}")

    # 2) Self-RAG
    print(f"\n--- Self-RAG ---"); t0 = time.time()
    ctx = format_search_results_numbered(results)
    r1 = generate(f"基于以下搜索结果回答问题，为每个事实标注[Relevant]或[Irrelevant]。\n搜索结果：\n{ctx}\n问题：{question}")
    r2 = generate(f"检查每个声明是否被原文支撑，标注[Supported]/[Partially]/[Unsupported]。\n回答：\n{r1}\n搜索结果：\n{ctx}")
    r3 = generate(f"基于反思重新生成答案，只保留[Supported]和[Partially]。\n原始回答：\n{r1}\n反思：\n{r2}\n问题：{question}")
    print(f"({time.time()-t0:.1f}s) {r3[:300]}")

    # 3) SEVE (original)
    print(f"\n--- SEVE (original) ---"); t0 = time.time()
    claims_o = extract_claims(results, question)
    print(f"Extract: {len(claims_o)} claims")
    for c in claims_o:
        print(f"  [{c['id']}] {c['claim'][:100]}")
    answer_o, cmap_o = generate_answer(question, claims_o)
    print(f"Generate: {len(answer_o)} chars -> {answer_o[:200]}")
    verifs_o = verify_claims(answer_o, claims_o, results)
    verdicts_o = [(v['claim_id'], v['verdict'], v['claim_text'][:60]) for v in verifs_o]
    print(f"Verify: {verdicts_o}")
    final_o = apply_verification(answer_o, verifs_o)
    print(f"Final ({time.time()-t0:.1f}s): {final_o[:300]}")

    # 4) SEVE-Merged (extract+generate)
    print(f"\n--- SEVE-Merged ---"); t0 = time.time()
    answer_m, claims_m, cmap_m = extract_and_generate(results, question)
    print(f"Extract+Generate: {len(claims_m)} claims, {len(answer_m)} chars -> {answer_m[:200]}")
    if claims_m:
        for c in claims_m:
            print(f"  [{c['id']}] {c['claim'][:100]}")
    verifs_m = verify_claims(answer_m, claims_m, results)
    verdicts_m = [(v['claim_id'], v['verdict'], v['claim_text'][:60]) for v in verifs_m]
    print(f"Verify: {verdicts_m}")
    final_m = apply_verification(answer_m, verifs_m)
    print(f"Final ({time.time()-t0:.1f}s): {final_m[:300]}")

    # Summary line
    print(f"\n  >> BC{idx:03d} Ref={answer_ref} | Vanilla={v[:60]}... | SEVE={final_o[:60]}... | Merged={final_m[:60]}...")
