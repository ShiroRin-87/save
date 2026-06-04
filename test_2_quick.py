"""Quick test: 2 BrowseComp-ZH questions with all 3 methods, using multi-round iterative search."""
import json, sys, time
from pathlib import Path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.bing_client import search
from src.model_client import generate
from src.seve import extract_claims, generate_answer, verify_claims, apply_verification, iterative_search, extract_and_generate
from src.utils import load_json, save_json, format_search_results_numbered

# BC008 = 音乐/欢颜, BC006 = 艺术/2004
SELECTED = [8, 6]
all_qs = load_json("data/browsecomp-zh-decrypted.json")
cache = load_json("data/search_cache.json") if (ROOT / "data/search_cache.json").exists() else {}

for idx in SELECTED:
    q = all_qs[idx]
    qid = f"BC{idx:03d}"
    question = q["Question"]
    answer_ref = q["Answer"]
    topic = q["Topic"]

    print(f"\n{'='*60}")
    print(f"[{qid}] [{topic}] {question[:100]}...")
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
    print(f"\n--- Multi-round Iterative Search ---")
    n_before = len(results)
    results = iterative_search(question, results, max_rounds=2)
    if len(results) > n_before:
        cache[cache_key]["results"] = results
        cache[cache_key]["iterative_expanded"] = True
        save_json(cache, "data/search_cache.json")
        print(f"Expanded: {n_before} → {len(results)} results")

    # Show what we got
    total_chars = sum(r.get("full_text_len", 0) for r in results)
    print(f"\nTotal context: {total_chars:,} chars across {len(results)} pages")

    # Vanilla RAG
    print(f"\n--- Vanilla RAG ---"); t0 = time.time()
    v = generate(f"""你是一个问答助手。请基于以下搜索结果为用户问题提供准确、简洁的回答。

搜索结果：
{format_search_results_numbered(results)}

用户问题：{question}

请回答：""")
    print(f"({time.time()-t0:.1f}s) {v[:300]}")

    # Self-RAG
    print(f"\n--- Self-RAG ---"); t0 = time.time()
    ctx = format_search_results_numbered(results)
    r1 = generate(f"基于以下搜索结果回答问题，为每个事实标注[Relevant]或[Irrelevant]。\n搜索结果：\n{ctx}\n问题：{question}")
    r2 = generate(f"检查每个声明是否被原文支撑，标注[Supported]/[Partially]/[Unsupported]。\n回答：\n{r1}\n搜索结果：\n{ctx}")
    r3 = generate(f"基于反思重新生成答案，只保留[Supported]和[Partially]。\n原始回答：\n{r1}\n反思：\n{r2}\n问题：{question}")
    print(f"({time.time()-t0:.1f}s) {r3[:300]}")

    # SEVE
    print(f"\n--- SEVE ---"); t0 = time.time()
    claims = extract_claims(results, question)
    print(f"Extract: {len(claims)} claims")
    for c in claims:
        print(f"  [{c['id']}] {c['claim'][:100]}")
    answer, cmap = generate_answer(question, claims)
    print(f"Generate: {len(answer)} chars → {answer[:200]}")
    verifs = verify_claims(answer, claims, results)
    verdicts = [(v['claim_id'], v['verdict'], v['claim_text'][:60]) for v in verifs]
    print(f"Verify: {verdicts}")
    final = apply_verification(answer, verifs)
    print(f"Final ({time.time()-t0:.1f}s): {final[:300]}")

    # SEVE-Merged (extract + generate combined, 1 API call instead of 2)
    print(f"\n--- SEVE-Merged (extract+generate) ---"); t0 = time.time()
    answer_m, claims_m, cmap_m = extract_and_generate(results, question)
    print(f"Extract+Generate: {len(claims_m)} claims, {len(answer_m)} chars → {answer_m[:200]}")
    if claims_m:
        for c in claims_m:
            print(f"  [{c['id']}] {c['claim'][:100]}")
    verifs_m = verify_claims(answer_m, claims_m, results)
    verdicts_m = [(v['claim_id'], v['verdict'], v['claim_text'][:60]) for v in verifs_m]
    print(f"Verify: {verdicts_m}")
    final_m = apply_verification(answer_m, verifs_m)
    print(f"Final ({time.time()-t0:.1f}s): {final_m[:300]}")
