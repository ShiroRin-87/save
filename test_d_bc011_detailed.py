"""Method D detailed per-step timing on BC011 (回车巷)."""
import json, sys, time, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.bing_client import search, _jina_fetch, _truncate, SERP_API_KEY, SEARCH_TOP_K, SERPAPI_ENDPOINT, JINA_ENDPOINT
from src.model_client import generate, reset_usage, get_usage
from src.seve import (
    needs_hypothesis_search, generate_hypotheses, _extract_constraints,
    _filter_relevant_results, extract_claims, generate_answer,
    verify_claims, apply_verification, fallback_reasoning,
    gap_fill_search, _parse_json_strings, _parse_json_claims,
)
from src.utils import load_json, format_search_results

# Load BC011
all_qs = load_json("data/browsecomp-zh-decrypted.json")
q = all_qs[11]  # BC011 (回车巷)
question = q["Question"]
answer_ref = q["Answer"]
print(f"BC011: 回车巷")
print(f"Question: {question}")
print(f"Reference: {answer_ref}")

TIMINGS = {}  # label -> seconds
TOKENS = {}   # label -> (in, out)

def timed(label):
    """Context manager/decorator for timing."""
    class _Ctx:
        def __enter__(self2):
            self2.t0 = time.time()
            reset_usage()
            return self2
        def __exit__(self2, *args):
            elapsed = time.time() - self2.t0
            u = get_usage()
            TIMINGS[label] = elapsed
            TOKENS[label] = (u["input_tokens"], u["output_tokens"])
            print(f"  [{label}] {elapsed:.1f}s  tok: in={u['input_tokens']} out={u['output_tokens']}")
    return _Ctx()

print(f"\n{'='*70}")
print("PHASE 1: SEARCH")
print(f"{'='*70}")

# ═══ Step 1a: Direct search ═══
print("\n--- 1a: Direct search ---")
with timed("1a_serpapi_direct"):
    # Manually time the SerpAPI part separately
    import requests, urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    params = {"engine": "google", "q": question, "api_key": SERP_API_KEY, "num": SEARCH_TOP_K}
    t_serp = time.time()
    resp = requests.get(SERPAPI_ENDPOINT, params=params, timeout=30, verify=False)
    resp.raise_for_status()
    data = resp.json()
    serp_time = time.time() - t_serp
    print(f"  SerpAPI call: {serp_time:.1f}s")
    TIMINGS["1a_serpapi"] = serp_time

    items = list(enumerate(data.get("organic_results", [])[:SEARCH_TOP_K], start=1))
    results_a = []
    for i, item in items:
        results_a.append({
            "rank": i, "title": item.get("title", ""),
            "url": item.get("link", ""), "snippet": item.get("snippet", ""),
            "full_text": item.get("snippet", ""),
            "full_text_source": "snippet",
            "full_text_len": len(item.get("snippet", "")),
        })

print(f"  SerpAPI returned {len(results_a)} organic results")

# Jina fetches (parallel, timed individually)
print(f"\n  Jina full-text fetches (parallel, 8 workers):")
jina_tasks = [(i, r["url"]) for i, r in enumerate(results_a) if r["url"]]
jina_times = {}
with ThreadPoolExecutor(max_workers=8) as ex:
    def _fetch_one(idx, url):
        t0 = time.time()
        text, is_bad = _jina_fetch(url)
        elapsed = time.time() - t0
        return idx, text, is_bad, elapsed, url[:60]
    futures = {ex.submit(_fetch_one, idx, url): idx for idx, url in jina_tasks}
    for f in as_completed(futures):
        idx, text, is_bad, elapsed, url_short = f.result()
        jina_times[idx] = elapsed
        if text is not None:
            results_a[idx]["full_text"] = _truncate(text)
            results_a[idx]["full_text_source"] = "jina"
            results_a[idx]["full_text_len"] = len(results_a[idx]["full_text"])
            status = "LOW" if is_bad else "OK"
        else:
            status = "FAIL"
        print(f"    [{idx}] {url_short}... {elapsed:.1f}s [{status}]")
TIMINGS["1a_jina_total"] = sum(jina_times.values())
TIMINGS["1a_jina_max"] = max(jina_times.values()) if jina_times else 0
TIMINGS["1a_total"] = TIMINGS["1a_serpapi"] + TIMINGS["1a_jina_total"]
print(f"  Jina total: {TIMINGS['1a_jina_total']:.1f}s, max: {TIMINGS['1a_jina_max']:.1f}s")
print(f"  1a total (direct search): {TIMINGS['1a_total']:.1f}s")

# ═══ Step 1b: needs_hypothesis_search ═══
print(f"\n--- 1b: Check if hypothesis search needed ---")
with timed("1b_needs_hypothesis"):
    need_hypo = needs_hypothesis_search(results_a, question)
print(f"  Hypothesis search needed: {need_hypo}")

# ═══ Step 1c: Generate hypotheses ═══
print(f"\n--- 1c: Generate hypotheses ---")
with timed("1c_generate_hypotheses"):
    hypotheses = generate_hypotheses(question, n=5)
for i, h in enumerate(hypotheses, 1):
    print(f"    H{i}: {h}")

# ═══ Step 1d: Extract constraints (once) ═══
print(f"\n--- 1d: Extract constraints ---")
with timed("1d_extract_constraints"):
    constraints = _extract_constraints(question)
print(f"  Constraints: {constraints[:120]}...")

# ═══ Step 1e: Parallel hypothesis searches ═══
print(f"\n--- 1e: Parallel hypothesis searches (5 threads) ---")
hypo_search_times = {}
all_results = list(results_a)
seen_urls = {r.get("url", "") for r in all_results}
seen_lock = threading.Lock()

def _search_one(h):
    query = f"{h} {constraints}"
    t0 = time.time()
    results = search(query, top_k=5)
    elapsed = time.time() - t0
    return h, results, elapsed

t_search_start = time.time()
with ThreadPoolExecutor(max_workers=5) as ex:
    futures = {ex.submit(_search_one, h): h for h in hypotheses}
    for f in as_completed(futures):
        h, results, elapsed = f.result()
        hypo_search_times[h] = elapsed
        with seen_lock:
            added = 0
            for r in results:
                url = r.get("url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    all_results.append(r)
                    added += 1
        print(f"    {h[:40]}... {elapsed:.1f}s, {added} new results")
TIMINGS["1e_hypo_searches_total"] = time.time() - t_search_start
TIMINGS["1e_hypo_searches_sum"] = sum(hypo_search_times.values())
TIMINGS["1e_hypo_searches_max"] = max(hypo_search_times.values()) if hypo_search_times else 0
print(f"  Parallel wall time: {TIMINGS['1e_hypo_searches_total']:.1f}s")
print(f"  Sum of individual searches: {TIMINGS['1e_hypo_searches_sum']:.1f}s")
print(f"  Longest individual: {TIMINGS['1e_hypo_searches_max']:.1f}s")

# ═══ Step 1f: Direct fallback search ═══
print(f"\n--- 1f: Direct fallback search ---")
with timed("1f_direct_fallback"):
    direct_results = search(question, top_k=3)
for r in direct_results:
    url = r.get("url", "")
    if url and url not in seen_urls:
        seen_urls.add(url)
        all_results.append(r)
print(f"  Added {len([r for r in direct_results if r.get('url','') not in seen_urls])} new, total: {len(all_results)} results")

TIMINGS["phase1_total"] = TIMINGS["1a_total"] + TOKENS.get("1b_needs_hypothesis", (0,0))[0] / 1000 + TIMINGS["1c_generate_hypotheses"] + TIMINGS["1d_extract_constraints"] + TIMINGS["1e_hypo_searches_total"] + TIMINGS["1f_direct_fallback"]
# Better: just sum the actual timed sections
phase1_keys = ["1a_total", "1b_needs_hypothesis", "1c_generate_hypotheses", "1d_extract_constraints", "1e_hypo_searches_total", "1f_direct_fallback"]
TIMINGS["phase1_sum"] = sum(TIMINGS.get(k, 0) for k in phase1_keys)
print(f"\n  >>> Phase 1 total: {TIMINGS['phase1_sum']:.1f}s")

print(f"\n{'='*70}")
print("PHASE 2: EXTRACTION")
print(f"{'='*70}")

# ═══ Step 2a: Relevance filter ═══
print(f"\n--- 2a: Relevance filter ---")
with timed("2a_relevance_filter"):
    relevant_results = _filter_relevant_results(all_results, question, top_n=10)
print(f"  {len(all_results)} → {len(relevant_results)} results")

# ═══ Step 2b: Batch extraction ═══
print(f"\n--- 2b: Batch extraction ---")
results_c = relevant_results  # use filtered results
BATCH_SIZE = 3
batches = []
for i in range(0, len(results_c), BATCH_SIZE):
    batch = results_c[i:i + BATCH_SIZE]
    if batch:
        batches.append(batch)

from src.seve import _extract_from_batch
from src.config import JINA_MAX_CHARS

batch_times = []
all_claims = []
print(f"  {len(batches)} batches from {len(results_c)} pages (batch_size={BATCH_SIZE}):")
with ThreadPoolExecutor(max_workers=4) as ex:
    def _extract_batch_timed(batch, idx):
        t0 = time.time()
        reset_usage()  # per-batch token tracking not possible easily, skip
        claims = _extract_from_batch(batch, question)
        elapsed = time.time() - t0
        return idx, claims, elapsed
    futures = {ex.submit(_extract_batch_timed, batch, i): i for i, batch in enumerate(batches)}
    for f in as_completed(futures):
        idx, claims, elapsed = f.result()
        batch_times.append(elapsed)
        all_claims.extend(claims)
        batch_urls = [b.get("url","")[:50] for b in batches[idx]]
        print(f"    Batch {idx+1}/{len(batches)}: {len(claims)} claims, {elapsed:.1f}s  [{', '.join(batch_urls)}]")

TIMINGS["2b_batch_total_wall"] = sum(batch_times)  # sum since batches are serial in practice
TIMINGS["2b_batch_max"] = max(batch_times) if batch_times else 0
print(f"  Batch extraction wall sum: {TIMINGS['2b_batch_total_wall']:.1f}s")
print(f"  Longest batch: {TIMINGS['2b_batch_max']:.1f}s")

# ═══ Step 2c: Dedup ═══
print(f"\n--- 2c: Dedup ---")
t_dedup = time.time()
seen = {}
for c in all_claims:
    key = " ".join(c["claim"].strip().lower().split())
    if not key:
        continue
    if key in seen:
        if len(c["snippet"]) > len(seen[key]["snippet"]):
            seen[key]["snippet"] = c["snippet"]
    else:
        seen[key] = c
claims = list(seen.values())
for i, c in enumerate(claims, 1):
    c["id"] = i
TIMINGS["2c_dedup"] = time.time() - t_dedup
print(f"  {len(all_claims)} raw → {len(claims)} unique ({TIMINGS['2c_dedup']:.2f}s)")

TIMINGS["phase2_total"] = TIMINGS["2a_relevance_filter"] + TIMINGS["2b_batch_total_wall"] + TIMINGS["2c_dedup"]
print(f"\n  >>> Phase 2 total: {TIMINGS['phase2_total']:.1f}s")

print(f"\n{'='*70}")
print("PHASE 3: GENERATION + GAP-FILL")
print(f"{'='*70}")

# ═══ Step 3a: Generate answer from claims ═══
print(f"\n--- 3a: Generate answer ---")
with timed("3a_generate_answer"):
    answer, cmap = generate_answer(question, claims)
print(f"  Answer: {answer[:200]}...")

# ═══ Step 3b: Gap-fill (manual timing per round) ═══
print(f"\n--- 3b: Gap-fill (max 2 rounds) ---")
all_new = []
current_answer = answer
seen_gap_urls = {r.get("url", "") for r in results_c}

for rnd in range(2):
    print(f"\n  Gap-fill round {rnd+1}:")
    prev = current_answer if rnd > 0 else ""

    with timed(f"3b_gap_keywords_r{rnd+1}"):
        kw_prompt = f"""以下回答可能不完整，只识别了中间步骤但没有给出最终答案。请提取需要补充搜索的关键词。

问题：{question}
当前回答：{current_answer}

输出一个JSON字符串数组，包含2-5个补充搜索关键词。只输出JSON数组。

JSON数组："""
        keywords_raw = generate(kw_prompt).strip()
        keywords = _parse_json_strings(keywords_raw)
    print(f"    Keywords: {keywords}")

    if not keywords:
        print(f"    No keywords, stop")
        break

    query = " ".join(keywords)
    print(f"    Query: {query[:100]}...")
    t_search = time.time()
    gap_results = search(query, top_k=5)
    gap_search_time = time.time() - t_search
    TIMINGS[f"3b_gap_search_r{rnd+1}"] = gap_search_time

    new = [r for r in gap_results if r.get("url", "") not in seen_gap_urls]
    for r in new:
        seen_gap_urls.add(r.get("url", ""))
    all_new.extend(new)
    print(f"    {len(new)} new results ({gap_search_time:.1f}s)")

    if not new:
        print(f"    No new results, stop")
        break

    # Regenerate answer
    accumulated = results_c + all_new
    ctx = format_search_results(accumulated)
    with timed(f"3b_gap_regenerate_r{rnd+1}"):
        current_answer = generate(
            f"你是一个问答助手。请基于以下搜索结果为用户问题提供准确、简洁的回答。\n\n"
            f"搜索结果：\n{ctx}\n\n用户问题：{question}\n\n请回答："
        )
    print(f"    New answer: {current_answer[:200]}...")

# If gap-fill added results, re-extract claims from expanded set
if all_new:
    print(f"\n--- 3c: Re-extract claims with gap-fill results ---")
    with timed("3c_reextract_claims"):
        expanded_results = results_c + all_new
        claims = extract_claims(expanded_results, question)
    print(f"  {len(claims)} claims after re-extraction")

    with timed("3c_regenerate_answer"):
        answer, cmap = generate_answer(question, claims)
else:
    print(f"\n  Gap-fill added 0 results, keeping original claims")

TIMINGS["3b_gap_total"] = sum(TIMINGS.get(k, 0) for k in ["3b_gap_keywords_r1", "3b_gap_search_r1", "3b_gap_regenerate_r1", "3b_gap_keywords_r2", "3b_gap_search_r2", "3b_gap_regenerate_r2"])
TIMINGS["phase3_total"] = TIMINGS["3a_generate_answer"] + TIMINGS["3b_gap_total"] + TIMINGS.get("3c_reextract_claims", 0) + TIMINGS.get("3c_regenerate_answer", 0)
print(f"\n  >>> Phase 3 total: {TIMINGS['phase3_total']:.1f}s")

print(f"\n{'='*70}")
print("PHASE 4: VERIFICATION")
print(f"{'='*70}")

# ═══ Step 4a: Verify claims ═══
print(f"\n--- 4a: Batch verification ---")

# Extract cited IDs
import re
cited_ids = set()
for m in re.finditer(r"\[(\d+)\]", answer):
    cited_ids.add(m.group(1))

claim_by_id = {str(c["id"]): c for c in claims}
url_to_result = {r["url"]: r for r in (results_c + all_new)}

items = []
ordered_ids = []
for cid in cited_ids:
    claim = claim_by_id.get(cid)
    if not claim:
        continue
    original_result = url_to_result.get(claim["url"])
    original_text = (original_result.get("full_text") or original_result.get("snippet", "")) if original_result else claim["snippet"]
    items.append({"id": cid, "claim": claim["claim"], "original": original_text})
    ordered_ids.append(cid)

print(f"  {len(items)} cited claims to verify")

if items:
    with timed("4a_batch_verify"):
        items_json = json.dumps(items, ensure_ascii=False, indent=2)
        batch_prompt = f"""对以下 JSON 数组中的每组声明和原文，判断原文是否包含声明中声称的信息。
为每个元素添加 "verdict" 字段，值为 YES/PARTIAL/NO。
只输出 JSON 数组，不要输出任何其他文字。

输入：
{items_json}

输出 JSON："""
        result_text = generate(batch_prompt)
        batch_results = _parse_json_claims(result_text)
        verdict_map = {}
        for r in batch_results:
            cid_key = str(r.get("id", ""))
            raw_verdict = str(r.get("verdict", "")).upper()
            if raw_verdict in ("YES", "PARTIAL", "NO"):
                verdict_map[cid_key] = raw_verdict
    yes_n = sum(1 for v in verdict_map.values() if v == "YES")
    no_n = sum(1 for v in verdict_map.values() if v == "NO")
    partial_n = sum(1 for v in verdict_map.values() if v == "PARTIAL")
    unknown_n = len(items) - len(verdict_map)
    print(f"  Batch result: {yes_n}Y/{partial_n}P/{no_n}N/{unknown_n}UNK")

    # ═══ Step 4b: Parallel fallback for UNKNOWN ═══
    if unknown_n > 0:
        print(f"\n--- 4b: Parallel fallback for {unknown_n} UNKNOWN claims ---")
        unknown_list = []
        for cid in ordered_ids:
            if verdict_map.get(cid, "UNKNOWN") == "UNKNOWN":
                claim = claim_by_id.get(cid)
                if claim:
                    original_result = url_to_result.get(claim["url"])
                    original_text = (original_result.get("full_text") or original_result.get("snippet", "")) if original_result else (claim.get("snippet") or claim.get("context", ""))
                    claim_text = claim.get("claim") or claim.get("sentence", "")
                    unknown_list.append((cid, claim_text, original_text))

        fb_times = {}
        def _verify_one(cid, claim_text, original_text):
            try:
                t0 = time.time()
                fallback_prompt = f"""请判断以下引用原文是否包含生成声明中声称的信息。

生成声明：{claim_text}
引用原文：{original_text}

仅回答一个词：YES / PARTIAL / NO

你的判断："""
                fb = generate(fallback_prompt).strip().upper()
                elapsed = time.time() - t0
                if fb in ("YES", "PARTIAL", "NO"):
                    return cid, fb, elapsed
            except Exception:
                pass
            return cid, "UNKNOWN", 0

        with ThreadPoolExecutor(max_workers=8) as ex:
            futures = {ex.submit(_verify_one, cid, ct, ot): cid for cid, ct, ot in unknown_list}
            for f in as_completed(futures):
                cid, verdict, elapsed = f.result()
                verdict_map[cid] = verdict
                fb_times[cid] = elapsed
                print(f"    Claim [{cid}]: {verdict} ({elapsed:.1f}s)")

        TIMINGS["4b_fallback_sum"] = sum(fb_times.values())
        TIMINGS["4b_fallback_max"] = max(fb_times.values()) if fb_times else 0
        print(f"  Fallback sum: {TIMINGS['4b_fallback_sum']:.1f}s, max: {TIMINGS['4b_fallback_max']:.1f}s")

    # Build final verification results
    verifs = []
    for cid in ordered_ids:
        claim = claim_by_id.get(cid)
        original_result = url_to_result.get(claim["url"]) if claim else None
        original_text = (original_result.get("full_text") or original_result.get("snippet", "")) if original_result else (claim.get("snippet") or claim.get("context", "") if claim else "")
        claim_text = (claim.get("claim") or claim.get("sentence", "")) if claim else ""
        url = claim["url"] if claim else ""
        verdict = verdict_map.get(cid, "UNKNOWN")
        verifs.append({
            "claim_id": cid, "claim_text": claim_text,
            "original_text": original_text, "url": url, "verdict": verdict,
        })
else:
    verifs = []

final_yes = sum(1 for v in verifs if v["verdict"] == "YES")
final_no = sum(1 for v in verifs if v["verdict"] == "NO")
final_partial = sum(1 for v in verifs if v["verdict"] == "PARTIAL")
print(f"\n  Final: {final_yes}Y/{final_partial}P/{final_no}N")

TIMINGS["phase4_total"] = TIMINGS.get("4a_batch_verify", 0) + TIMINGS.get("4b_fallback_sum", 0)
print(f"  >>> Phase 4 total: {TIMINGS['phase4_total']:.1f}s")

print(f"\n{'='*70}")
print("PHASE 5: CORRECTION")
print(f"{'='*70}")

# ═══ Step 5: Apply verification ═══
t_correct = time.time()
answer_final = apply_verification(answer, verifs)
TIMINGS["5_apply_verification"] = time.time() - t_correct
print(f"  Apply verification: {TIMINGS['5_apply_verification']:.3f}s")

print(f"\n{'='*70}")
print("PHASE 6: FALLBACK (Method E)")
print(f"{'='*70}")

# ═══ Step 6: Fallback reasoning ═══
print(f"\n--- 6: Fallback reasoning check ---")
with timed("6_fallback"):
    answer_final_e, fb_trig = fallback_reasoning(question, answer_final, verifs)
print(f"  Fallback triggered: {fb_trig}")
if fb_trig:
    print(f"  Fallback answer: {answer_final_e[:200]}...")

print(f"\n{'='*70}")
print("RESULTS SUMMARY")
print(f"{'='*70}")

print(f"\nQuestion: {question[:150]}...")
print(f"Reference answer: {answer_ref}")
print(f"Method D answer: {answer_final[:300]}")
print(f"Method E answer: {answer_final_e[:300]}")

match_d = "Y" if answer_ref.strip().lower() in answer_final.lower() else "N"
match_e = "Y" if answer_ref.strip().lower() in answer_final_e.lower() else "N"
print(f"\nD match: {match_d}")
print(f"E match: {match_e}")
print(f"Claims: {len(claims)}, Verification: {final_yes}Y/{final_partial}P/{final_no}N")

print(f"\n{'='*70}")
print("DETAILED TIMING BREAKDOWN")
print(f"{'='*70}")

def print_timing_tree():
    total = 0
    sections = [
        ("PHASE 1: SEARCH", [
            ("1a_serpapi", "  SerpAPI call"),
            ("1a_jina_total", "  Jina fetches (parallel sum)"),
            ("1a_jina_max", "    longest single Jina"),
            ("1a_total", "  1a Direct search total"),
            ("1b_needs_hypothesis", "1b Check hypothesis need (LLM)"),
            ("1c_generate_hypotheses", "1c Generate hypotheses (LLM)"),
            ("1d_extract_constraints", "1d Extract constraints (LLM)"),
            ("1e_hypo_searches_total", "1e Hypothesis searches (parallel wall)"),
            ("1e_hypo_searches_max", "    longest single hypo search"),
            ("1f_direct_fallback", "1f Direct fallback search"),
        ]),
        ("PHASE 2: EXTRACTION", [
            ("2a_relevance_filter", "2a Relevance filter (LLM)"),
            ("2b_batch_total_wall", "2b Batch extraction (total wall)"),
            ("2b_batch_max", "    longest single batch"),
            ("2c_dedup", "2c Deduplication"),
        ]),
        ("PHASE 3: GENERATION + GAP-FILL", [
            ("3a_generate_answer", "3a Generate answer (LLM)"),
            ("3b_gap_keywords_r1", "3b Gap keywords R1 (LLM)"),
            ("3b_gap_search_r1", "   Gap search R1 (SerpAPI+Jina)"),
            ("3b_gap_regenerate_r1", "   Gap regenerate R1 (LLM)"),
            ("3b_gap_keywords_r2", "3b Gap keywords R2 (LLM)"),
            ("3b_gap_search_r2", "   Gap search R2 (SerpAPI+Jina)"),
            ("3b_gap_regenerate_r2", "   Gap regenerate R2 (LLM)"),
            ("3c_reextract_claims", "3c Re-extract claims (LLM batches)"),
            ("3c_regenerate_answer", "3c Re-generate answer (LLM)"),
        ]),
        ("PHASE 4: VERIFICATION", [
            ("4a_batch_verify", "4a Batch verify (LLM)"),
            ("4b_fallback_sum", "4b UNKNOWN fallback (parallel sum)"),
            ("4b_fallback_max", "    longest single fallback"),
        ]),
        ("PHASE 5: CORRECTION", [
            ("5_apply_verification", "5 Apply verification (no LLM)"),
        ]),
        ("PHASE 6: FALLBACK (Method E)", [
            ("6_fallback", "6 Fallback reasoning check"),
        ]),
    ]

    grand_total = 0
    for section, keys in sections:
        print(f"\n  {section}:")
        section_total = 0
        for key, label in keys:
            val = TIMINGS.get(key, 0)
            if val > 0:
                print(f"    {label:<45} {val:>7.1f}s")
                if not key.startswith("1a_jina_max") and not key.startswith("1e_hypo_searches_max") and not key.startswith("2b_batch_max") and not key.startswith("4b_fallback_max"):
                    section_total += val
        print(f"    {'─'*52}")
        print(f"    {'Section subtotal':<45} {section_total:>7.1f}s")
        grand_total += section_total

    print(f"\n  {'='*52}")
    print(f"  {'GRAND TOTAL':<45} {grand_total:>7.1f}s")

print_timing_tree()

# Save detailed results
output = {
    "qid": "BC011",
    "question": question,
    "reference": answer_ref,
    "answer_d": answer_final,
    "answer_e": answer_final_e,
    "match_d": match_d,
    "match_e": match_e,
    "claims": len(claims),
    "verification": f"{final_yes}Y/{final_partial}P/{final_no}N",
    "fallback_triggered": fb_trig,
    "timings": {k: round(v, 1) for k, v in TIMINGS.items()},
    "tokens": {k: list(v) for k, v in TOKENS.items()},
}
save_path = ROOT / "outputs" / "bc011_d_timing.json"
import json as _json
with open(save_path, "w", encoding="utf-8") as f:
    _json.dump(output, f, ensure_ascii=False, indent=2)
print(f"\nDetailed results saved to {save_path}")
