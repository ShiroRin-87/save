"""Comprehensive A/D/F comparison on 5 new questions.

Saves EVERY intermediate step: LLM prompts/responses, search queries/results,
claims, verifications, decomposition output. Uses proper wall clock for parallel ops.
"""
import json, sys, time, threading
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ══════════════════════════════════════════════════════════════════
# Global trace log: records every LLM call and search call with timing
# ══════════════════════════════════════════════════════════════════
_TRACE = []  # list of {step, type, wall_ts, prompt_snippet, response_snippet, ...}
_TRACE_LOCK = threading.Lock()
_T0 = time.time()

def _trace_llm(step_label, prompt, response, tok_in, tok_out, wall_s):
    with _TRACE_LOCK:
        _TRACE.append({
            "step": step_label, "type": "llm",
            "wall_ts": round(time.time() - _T0, 1),
            "wall_s": round(wall_s, 1),
            "prompt_snippet": prompt[:500].replace("\n", "\\n"),
            "response_snippet": response[:500].replace("\n", "\\n"),
            "prompt_full": prompt,
            "response_full": response,
            "tok_in": tok_in, "tok_out": tok_out,
        })

def _trace_search(step_label, query, n_results, wall_s):
    with _TRACE_LOCK:
        _TRACE.append({
            "step": step_label, "type": "search",
            "wall_ts": round(time.time() - _T0, 1),
            "wall_s": round(wall_s, 1),
            "query": query,
            "n_results": n_results,
        })

# ── Monkey-patch generate() ──
import src.model_client as mc
_orig_generate = mc.generate
_LLM_CALLS = [0]
_CURRENT_STEP = ["init"]

def _counting_generate(prompt, **kw):
    _LLM_CALLS[0] += 1
    t0 = time.time()
    response = _orig_generate(prompt, **kw)
    wall = time.time() - t0
    # Estimate tokens (prompt ~4 chars/tok, response ~2 chars/tok for Chinese)
    tok_in = len(prompt) // 4
    tok_out = len(response) // 2
    _trace_llm(_CURRENT_STEP[0], prompt, response, tok_in, tok_out, wall)
    return response

mc.generate = _counting_generate

# ── Monkey-patch search() ──
import src.bing_client as bc
_orig_search = bc.search
_SEARCH_CALLS = [0]

def _counting_search(query, **kw):
    _SEARCH_CALLS[0] += 1
    t0 = time.time()
    results = _orig_search(query, **kw)
    wall = time.time() - t0
    _trace_search(_CURRENT_STEP[0], query, len(results), wall)
    return results

bc.search = _counting_search

from src.model_client import reset_usage, get_usage
from src.utils import load_json, save_json, format_search_results
from src.seve import (
    needs_hypothesis_search, generate_hypotheses, _extract_constraints,
    hypothesis_driven_search, gap_fill_search,
    extract_claims, generate_answer,
    verify_claims, apply_verification, fallback_reasoning,
    chain_reasoning, _parse_json_strings, _parse_json_array, _parse_json_claims,
)
from src.config import SEARCH_TOP_K, JINA_MAX_CHARS

# ── Questions ──
all_qs = load_json("data/browsecomp-zh-decrypted.json")
INDICES = [16, 21, 31, 50, 74]  # BC016, BC021, BC031, BC050, BC074

CACHE_DIR = Path("data/adf_test_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

ALL_RESULTS = []


def check_recall(search_results, ref_answer):
    ref = ref_answer.strip().lower()
    if not ref or len(ref) < 2:
        return False
    for r in (search_results or []):
        full = (r.get("full_text", "") + " " + r.get("snippet", "") + " " + r.get("title", "")).lower()
        if ref in full:
            return True
    return False


def check_accuracy(predicted, ref_answer):
    return "Y" if ref_answer.strip().lower() in predicted.lower() else "N"


# ══════════════════════════════════════════════════════════════════
def run_method_A(question, answer_ref, qid):
    """A: Direct search + generate. Save everything."""
    global _TRACE, _LLM_CALLS, _SEARCH_CALLS, _CURRENT_STEP
    _TRACE = []
    _LLM_CALLS[0] = 0; _SEARCH_CALLS[0] = 0
    t0 = time.time()
    steps_log = []

    # A1: Search
    _CURRENT_STEP[0] = "A1_search"
    st = time.time()
    results = _orig_search(question, top_k=SEARCH_TOP_K)
    _SEARCH_CALLS[0] += 1
    _trace_search("A1_search", question, len(results), time.time() - st)
    steps_log.append({"step": "A1_direct_search", "wall_s": round(time.time() - st, 1),
                      "n_results": len(results)})

    # A2: Generate
    _CURRENT_STEP[0] = "A2_generate"
    st = time.time()
    ctx = format_search_results(results)
    prompt = f"你是一个问答助手。请基于以下搜索结果为用户问题提供准确、简洁的回答。\n\n搜索结果：\n{ctx}\n\n用户问题：{question}\n\n请回答："
    answer = _orig_generate(prompt)
    _LLM_CALLS[0] += 1
    _trace_llm("A2_generate", prompt, answer, len(prompt)//4, len(answer)//2, time.time() - st)
    steps_log.append({"step": "A2_generate", "wall_s": round(time.time() - st, 1)})

    total = time.time() - t0
    recall = check_recall(results, answer_ref)
    accuracy = check_accuracy(answer, answer_ref)
    print(f"    A: match={accuracy} recall={'HIT' if recall else 'MISS'} {total:.0f}s llm={_LLM_CALLS[0]} search={_SEARCH_CALLS[0]}")

    return {
        "method": "A", "qid": qid,
        "answer": answer, "accuracy": accuracy, "recall": recall,
        "total_wall_s": round(total, 1),
        "llm_calls": _LLM_CALLS[0], "search_calls": _SEARCH_CALLS[0],
        "hallucination_rate": None,
        "search_results": [{"title": r["title"], "url": r["url"], "snippet": r["snippet"],
                            "full_text_src": r.get("full_text_source","?"),
                            "full_text_len": r.get("full_text_len", 0)}
                          for r in results],
        "steps": steps_log, "trace": list(_TRACE),
    }


def run_method_D(question, answer_ref, qid):
    """D: Hypothesis + SEVE + gap. Save every single LLM/search step."""
    global _TRACE, _LLM_CALLS, _SEARCH_CALLS, _CURRENT_STEP
    _TRACE = []
    _LLM_CALLS[0] = 0; _SEARCH_CALLS[0] = 0
    t0 = time.time()
    steps_log = []

    # ── D1: Direct search ──
    _CURRENT_STEP[0] = "D1_direct_search"
    st = time.time()
    results_a = _orig_search(question, top_k=SEARCH_TOP_K)
    _SEARCH_CALLS[0] += 1
    _trace_search("D1_direct_search", question, len(results_a), time.time() - st)
    steps_log.append({"step": "D1_direct_search", "wall_s": round(time.time() - st, 1),
                      "n_results": len(results_a)})

    # ── D2: Check hypothesis need ──
    _CURRENT_STEP[0] = "D2_needs_hypothesis"
    st = time.time()
    need = needs_hypothesis_search(results_a, question)
    _LLM_CALLS[0] += 1
    steps_log.append({"step": "D2_needs_hypothesis", "wall_s": round(time.time() - st, 1),
                      "need": need})

    # ── D3: Hypothesis search ──
    _CURRENT_STEP[0] = "D3_hypotheses"
    st_hypo = time.time()
    if need:
        # Generate hypotheses (tracked via monkey-patch)
        _CURRENT_STEP[0] = "D3a_generate_hypotheses"
        st = time.time()
        hypotheses = generate_hypotheses(question, n=5)
        _LLM_CALLS[0] += 1
        steps_log.append({"step": "D3a_hypotheses", "wall_s": round(time.time() - st, 1),
                          "hypotheses": hypotheses})

        # Extract constraints
        _CURRENT_STEP[0] = "D3b_constraints"
        st = time.time()
        constraints = _extract_constraints(question)
        _LLM_CALLS[0] += 1
        steps_log.append({"step": "D3b_constraints", "wall_s": round(time.time() - st, 1),
                          "constraints": constraints[:200]})

        # Parallel hypothesis searches
        _CURRENT_STEP[0] = "D3c_hypo_search"
        all_hypo_results = []
        seen_urls = {r.get("url", "") for r in results_a}
        seen_lock = threading.Lock()
        hypo_detail = []

        def _search_one(h):
            query = f"{h} {constraints}"
            t0_s = time.time()
            res = _orig_search(query, top_k=5)
            _SEARCH_CALLS[0] += 1
            _trace_search("D3c_hypo_search", query, len(res), time.time() - t0_s)
            return h, query, res, time.time() - t0_s

        t_search_wall = time.time()
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=5) as ex:
            futures = {ex.submit(_search_one, h): h for h in hypotheses}
            for f in as_completed(futures):
                h, query, res, elapsed = f.result()
                with seen_lock:
                    added = 0
                    for r in res:
                        url = r.get("url", "")
                        if url and url not in seen_urls:
                            seen_urls.add(url)
                            all_hypo_results.append(r)
                            added += 1
                hypo_detail.append({"hypothesis": h, "query": query[:200],
                                    "n_results": len(res), "n_new": added, "wall_s": round(elapsed, 1)})
        hypo_wall = time.time() - t_search_wall
        steps_log.append({"step": "D3c_hypo_search", "wall_s": round(hypo_wall, 1),
                          "parallel_wall": round(hypo_wall, 1), "per_hypo": hypo_detail,
                          "total_new": len(all_hypo_results)})

        # Direct fallback
        _CURRENT_STEP[0] = "D3d_fallback_search"
        st = time.time()
        fb_results = _orig_search(question, top_k=3)
        _SEARCH_CALLS[0] += 1
        _trace_search("D3d_fallback_search", question, len(fb_results), time.time() - st)
        for r in fb_results:
            url = r.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                all_hypo_results.append(r)

        results_c = results_a + all_hypo_results
        steps_log.append({"step": "D3d_fallback", "wall_s": round(time.time() - st, 1),
                          "total_results": len(results_c)})
    else:
        results_c = list(results_a)
        steps_log.append({"step": "D3_skip_hypothesis", "total_results": len(results_c)})

    # ── D4: Extract claims ──
    _CURRENT_STEP[0] = "D4_extract_claims"
    st = time.time()
    claims = extract_claims(results_c, question)
    claims_wall = time.time() - st
    steps_log.append({"step": "D4_extract_claims", "wall_s": round(claims_wall, 1),
                      "n_raw_claims": len(claims),
                      "claims_snippet": [c["claim"][:100] for c in claims[:10]]})

    # ── D5: Generate answer ──
    _CURRENT_STEP[0] = "D5_generate_answer"
    st = time.time()
    answer, cmap = generate_answer(question, claims)
    _LLM_CALLS[0] += 1
    steps_log.append({"step": "D5_generate_answer", "wall_s": round(time.time() - st, 1),
                      "answer_snippet": answer[:200]})

    # ── D6: Gap-fill (manual for detailed tracking) ──
    _CURRENT_STEP[0] = "D6_gapfill"
    gap_log = []
    all_gap_new = []
    current_answer = answer
    seen_gap_urls = {r.get("url", "") for r in results_c}

    for rnd in range(2):
        # Keywords
        _CURRENT_STEP[0] = f"D6_gap_r{rnd+1}_keywords"
        st = time.time()
        kw_prompt = f"""以下回答可能不完整，只识别了中间步骤但没有给出最终答案。请提取需要补充搜索的关键词。

问题：{question}
当前回答：{current_answer}

输出一个JSON字符串数组，包含2-5个补充搜索关键词。只输出JSON数组。

JSON数组："""
        kw_raw = _orig_generate(kw_prompt).strip()
        _LLM_CALLS[0] += 1
        keywords = _parse_json_strings(kw_raw)
        kw_wall = time.time() - st
        _trace_llm(f"D6_gap_r{rnd+1}_keywords", kw_prompt, kw_raw, len(kw_prompt)//4, len(kw_raw)//2, kw_wall)

        if not keywords:
            gap_log.append({"round": rnd+1, "status": "no_keywords", "wall_s": round(kw_wall, 1)})
            break

        query = " ".join(keywords)
        _CURRENT_STEP[0] = f"D6_gap_r{rnd+1}_search"
        st = time.time()
        gap_res = _orig_search(query, top_k=5)
        _SEARCH_CALLS[0] += 1
        _trace_search(f"D6_gap_r{rnd+1}_search", query, len(gap_res), time.time() - st)
        search_wall = time.time() - st

        new = [r for r in gap_res if r.get("url", "") not in seen_gap_urls]
        for r in new:
            seen_gap_urls.add(r.get("url", ""))
        all_gap_new.extend(new)

        if not new:
            gap_log.append({"round": rnd+1, "status": "no_new_results",
                            "keywords": keywords, "query": query[:200],
                            "kw_wall_s": round(kw_wall, 1), "search_wall_s": round(search_wall, 1)})
            break

        # Regenerate
        _CURRENT_STEP[0] = f"D6_gap_r{rnd+1}_regenerate"
        st = time.time()
        accumulated = results_c + all_gap_new
        ctx = format_search_results(accumulated)
        regen_prompt = f"你是一个问答助手。请基于以下搜索结果为用户问题提供准确、简洁的回答。\n\n搜索结果：\n{ctx}\n\n用户问题：{question}\n\n请回答："
        current_answer = _orig_generate(regen_prompt)
        _LLM_CALLS[0] += 1
        regen_wall = time.time() - st
        _trace_llm(f"D6_gap_r{rnd+1}_regenerate", regen_prompt, current_answer, len(regen_prompt)//4, len(current_answer)//2, regen_wall)

        gap_log.append({"round": rnd+1, "status": "done",
                        "keywords": keywords, "query": query[:200],
                        "n_new": len(new), "n_total_gap": len(all_gap_new),
                        "kw_wall_s": round(kw_wall, 1),
                        "search_wall_s": round(search_wall, 1),
                        "regen_wall_s": round(regen_wall, 1),
                        "answer_snippet": current_answer[:200]})

        # Stop if answer stopped changing (simple check)
        if rnd > 0 and current_answer[:100] == (gap_log[rnd-1].get("answer_snippet", ""))[:100]:
            break

    steps_log.append({"step": "D6_gapfill", "rounds": gap_log, "total_new": len(all_gap_new)})

    # ── D7: Re-extract + regenerate if gap added results ──
    if all_gap_new:
        _CURRENT_STEP[0] = "D7_reextract"
        st = time.time()
        results_c = results_c + all_gap_new
        claims = extract_claims(results_c, question)
        steps_log.append({"step": "D7_reextract", "wall_s": round(time.time() - st, 1),
                          "n_claims": len(claims),
                          "claims_snippet": [c["claim"][:100] for c in claims[:10]]})

        _CURRENT_STEP[0] = "D7_regenerate"
        st = time.time()
        answer, cmap = generate_answer(question, claims)
        _LLM_CALLS[0] += 1
        steps_log.append({"step": "D7_regenerate", "wall_s": round(time.time() - st, 1),
                          "answer_snippet": answer[:200]})

    # ── D8: Verify claims ──
    _CURRENT_STEP[0] = "D8_verify"
    st = time.time()
    verifs = verify_claims(answer, claims, results_c)
    verif_wall = time.time() - st
    yes_n = sum(1 for v in verifs if v['verdict'] == 'YES')
    no_n = sum(1 for v in verifs if v['verdict'] == 'NO')
    partial_n = sum(1 for v in verifs if v['verdict'] == 'PARTIAL')
    steps_log.append({"step": "D8_verify", "wall_s": round(verif_wall, 1),
                      "verdict": f"{yes_n}Y/{partial_n}P/{no_n}N",
                      "verifications": [{"cid": v["claim_id"], "verdict": v["verdict"],
                                         "claim": v["claim_text"][:100]}
                                       for v in verifs]})

    # ── D9: Apply verification ──
    _CURRENT_STEP[0] = "D9_correct"
    st = time.time()
    answer_final = apply_verification(answer, verifs)
    steps_log.append({"step": "D9_correct", "wall_s": round(time.time() - st, 3)})

    # ── D10: Fallback ──
    _CURRENT_STEP[0] = "D10_fallback"
    st = time.time()
    answer_final_e, fb_trig = fallback_reasoning(question, answer_final, verifs)
    steps_log.append({"step": "D10_fallback", "wall_s": round(time.time() - st, 1),
                      "triggered": fb_trig})

    total = time.time() - t0
    recall = check_recall(results_c, answer_ref)
    accuracy = check_accuracy(answer_final_e, answer_ref)
    total_v = yes_n + no_n + partial_n
    hall_rate = round(no_n / total_v, 3) if total_v > 0 else None

    print(f"    D: match={accuracy} recall={'HIT' if recall else 'MISS'} {total:.0f}s "
          f"vf={yes_n}Y/{partial_n}P/{no_n}N llm={_LLM_CALLS[0]} search={_SEARCH_CALLS[0]}")

    return {
        "method": "D", "qid": qid,
        "answer": answer_final_e, "accuracy": accuracy, "recall": recall,
        "total_wall_s": round(total, 1),
        "llm_calls": _LLM_CALLS[0], "search_calls": _SEARCH_CALLS[0],
        "hallucination_rate": hall_rate,
        "verification": f"{yes_n}Y/{partial_n}P/{no_n}N",
        "n_claims": len(claims),
        "fallback_triggered": fb_trig,
        "search_results": [{"title": r["title"], "url": r["url"], "snippet": r["snippet"],
                            "full_text_src": r.get("full_text_source","?"),
                            "full_text_len": r.get("full_text_len", 0)}
                          for r in results_c],
        "steps": steps_log, "trace": list(_TRACE),
    }


def run_method_F(question, answer_ref, qid):
    """F: Chain + SEVE. Fully instrumented — decompose, each hop, SEVE phases."""
    global _TRACE, _LLM_CALLS, _SEARCH_CALLS, _CURRENT_STEP
    _TRACE = []
    _LLM_CALLS[0] = 0; _SEARCH_CALLS[0] = 0
    t0 = time.time()
    steps_log = []
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # ── F1: Decompose ──
    _CURRENT_STEP[0] = "F1_decompose"
    st = time.time()
    decomp_prompt = f"""将以下复杂问题拆解为2-5个需要按顺序解决的子问题。
每个子问题应是独立的、可搜索的事实查询。解决前一个才能进入下一个。

输出JSON数组，每个元素含 "step"(序号) 和 "question"(子问题)。
只输出JSON数组，不要其他文字。

问题：{question}

JSON数组："""
    decomp_raw = _orig_generate(decomp_prompt).strip()
    _LLM_CALLS[0] += 1
    _trace_llm("F1_decompose", decomp_prompt, decomp_raw, len(decomp_prompt)//4, len(decomp_raw)//2, time.time() - st)
    steps = _parse_json_array(decomp_raw)
    steps_log.append({
        "step": "F1_decompose", "wall_s": round(time.time() - st, 1),
        "raw_response": decomp_raw,
        "parsed": steps if isinstance(steps, list) else str(steps),
        "n_steps": len(steps) if isinstance(steps, list) else 0,
        "success": isinstance(steps, list) and len(steps) > 0,
    })

    if not isinstance(steps, list) or len(steps) < 1:
        print(f"    F: decompose FAILED (raw: {decomp_raw[:150]})")
        total = time.time() - t0
        return {
            "method": "F", "qid": qid,
            "answer": "", "accuracy": "N", "recall": False,
            "total_wall_s": round(total, 1),
            "llm_calls": _LLM_CALLS[0], "search_calls": _SEARCH_CALLS[0],
            "hallucination_rate": None, "verification": "0Y/0P/0N",
            "n_claims": 0, "chain_success": False,
            "search_results": [],
            "steps": steps_log, "trace": list(_TRACE),
        }

    print(f"    F: decomposed into {len(steps)} hops")

    # ── F2: Sequential hop-by-hop search ──
    all_results = []
    confirmed_facts = []
    seen_urls = set()
    hop_log = []

    for i, step in enumerate(steps):
        if isinstance(step, dict):
            step_q = step.get("question", str(step))
            step_n = step.get("step", i + 1)
        else:
            step_q = str(step)
            step_n = i + 1

        hop_entry = {"hop": step_n, "question": step_q[:200]}

        # Search this hop
        _CURRENT_STEP[0] = f"F2_hop{step_n}_search"
        st = time.time()
        hop_results = _orig_search(step_q)
        _SEARCH_CALLS[0] += 1
        _trace_search(f"F2_hop{step_n}_search", step_q, len(hop_results), time.time() - st)
        hop_entry["search_wall_s"] = round(time.time() - st, 1)
        hop_entry["n_results"] = len(hop_results)
        for r in hop_results:
            url = r.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                all_results.append(r)
        hop_entry["total_accumulated"] = len(all_results)

        # Answer this hop
        _CURRENT_STEP[0] = f"F2_hop{step_n}_answer"
        st = time.time()
        prev_facts = "\n".join(f"- {f}" for f in confirmed_facts) if confirmed_facts else "(第一个步骤)"
        ctx = format_search_results(hop_results)
        hop_prompt = f"""基于搜索结果回答子问题。注意：下面列出了前面步骤已确认的事实，请基于这些事实继续推理。

已确认事实：
{prev_facts}

当前搜索结果：
{ctx}

当前子问题：{step_q}

如果搜索结果足以回答，请以"确认："开头给出答案。如果不足，以"不确定："开头说明缺少什么。"""
        hop_answer = _orig_generate(hop_prompt).strip()
        _LLM_CALLS[0] += 1
        _trace_llm(f"F2_hop{step_n}_answer", hop_prompt, hop_answer, len(hop_prompt)//4, len(hop_answer)//2, time.time() - st)
        is_confirmed = hop_answer.startswith("确认")
        hop_entry["answer_wall_s"] = round(time.time() - st, 1)
        hop_entry["answer"] = hop_answer[:300]
        hop_entry["confirmed"] = is_confirmed

        # Retry if uncertain
        retry_log = []
        if not is_confirmed:
            _CURRENT_STEP[0] = f"F2_hop{step_n}_retry_query"
            st = time.time()
            alt_prompt = f"""以下搜索未找到答案，请提供替代搜索词。
子问题：{step_q}
缺失信息：{hop_answer}
输出JSON字符串数组。只输出JSON数组。
JSON数组："""
            alt_raw = _orig_generate(alt_prompt).strip()
            _LLM_CALLS[0] += 1
            _trace_llm(f"F2_hop{step_n}_retry_query", alt_prompt, alt_raw, len(alt_prompt)//4, len(alt_raw)//2, time.time() - st)
            alt_queries = _parse_json_strings(alt_raw)

            for j, alt_q in enumerate(alt_queries[:2]):
                _CURRENT_STEP[0] = f"F2_hop{step_n}_retry{j}_search"
                st_s = time.time()
                alt_results = _orig_search(alt_q)
                _SEARCH_CALLS[0] += 1
                _trace_search(f"F2_hop{step_n}_retry{j}_search", alt_q, len(alt_results), time.time() - st_s)
                for r in alt_results:
                    url = r.get("url", "")
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        all_results.append(r)

                _CURRENT_STEP[0] = f"F2_hop{step_n}_retry{j}_answer"
                st_r = time.time()
                alt_ctx = format_search_results(alt_results)
                retry_prompt = f"""基于新搜索结果重新回答。

已确认事实：
{prev_facts}

新搜索结果：
{alt_ctx}

子问题：{step_q}

如果搜索结果足以回答，请以"确认："开头。不足则以"不确定："开头。"""
                retry_answer = _orig_generate(retry_prompt).strip()
                _LLM_CALLS[0] += 1
                _trace_llm(f"F2_hop{step_n}_retry{j}_answer", retry_prompt, retry_answer, len(retry_prompt)//4, len(retry_answer)//2, time.time() - st_r)
                retry_ok = retry_answer.startswith("确认")
                retry_log.append({"retry": j, "alt_query": alt_q[:200],
                                  "n_results": len(alt_results), "confirmed": retry_ok,
                                  "answer": retry_answer[:200]})
                if retry_ok:
                    hop_answer = retry_answer
                    is_confirmed = True
                    break
            hop_entry["retries"] = retry_log

        confirmed_facts.append(f"[Step {step_n}] Q: {step_q} → {hop_answer}")
        hop_log.append(hop_entry)
        print(f"    Hop {step_n}/{len(steps)}: {'OK' if is_confirmed else '?'} {hop_answer[:80]}")

    steps_log.append({"step": "F2_hops", "hops": hop_log, "total_results": len(all_results)})

    if not all_results:
        total = time.time() - t0
        return {
            "method": "F", "qid": qid,
            "answer": "", "accuracy": "N", "recall": False,
            "total_wall_s": round(total, 1),
            "llm_calls": _LLM_CALLS[0], "search_calls": _SEARCH_CALLS[0],
            "hallucination_rate": None, "verification": "0Y/0P/0N",
            "n_claims": 0, "chain_success": False,
            "search_results": [],
            "steps": steps_log, "trace": list(_TRACE),
        }

    # ── F3-7: SEVE pipeline ──
    _CURRENT_STEP[0] = "F3_extract_claims"
    st = time.time()
    claims = extract_claims(all_results, question)
    steps_log.append({"step": "F3_extract", "wall_s": round(time.time() - st, 1),
                      "n_claims": len(claims),
                      "claims_snippet": [c["claim"][:100] for c in claims[:10]]})

    _CURRENT_STEP[0] = "F4_generate"
    st = time.time()
    answer, cmap = generate_answer(question, claims)
    _LLM_CALLS[0] += 1
    steps_log.append({"step": "F4_generate", "wall_s": round(time.time() - st, 1),
                      "answer_snippet": answer[:200] if answer else ""})

    _CURRENT_STEP[0] = "F5_verify"
    st = time.time()
    verifs = verify_claims(answer, claims, all_results)
    yes_n = sum(1 for v in verifs if v.get("verdict") == "YES") if verifs else 0
    no_n = sum(1 for v in verifs if v.get("verdict") == "NO") if verifs else 0
    partial_n = sum(1 for v in verifs if v.get("verdict") == "PARTIAL") if verifs else 0
    steps_log.append({"step": "F5_verify", "wall_s": round(time.time() - st, 1),
                      "verdict": f"{yes_n}Y/{partial_n}P/{no_n}N",
                      "verifications": [{"cid": v.get("claim_id",""), "verdict": v.get("verdict",""),
                                         "claim": v.get("claim_text","")[:100]}
                                       for v in (verifs or [])]})

    _CURRENT_STEP[0] = "F6_correct"
    st = time.time()
    answer_final = apply_verification(answer, verifs or [])
    steps_log.append({"step": "F6_correct", "wall_s": round(time.time() - st, 3)})

    _CURRENT_STEP[0] = "F7_fallback"
    st = time.time()
    answer_final, fb_trig = fallback_reasoning(question, answer_final, verifs or [])
    steps_log.append({"step": "F7_fallback", "wall_s": round(time.time() - st, 1),
                      "triggered": fb_trig})

    total = time.time() - t0
    recall = check_recall(all_results, answer_ref)
    accuracy = check_accuracy(answer_final, answer_ref)
    total_v = yes_n + no_n + partial_n
    hall_rate = round(no_n / total_v, 3) if total_v > 0 else None

    print(f"    F: match={accuracy} recall={'HIT' if recall else 'MISS'} {total:.0f}s "
          f"chain_ok=True vf={yes_n}Y/{partial_n}P/{no_n}N llm={_LLM_CALLS[0]} search={_SEARCH_CALLS[0]}")

    return {
        "method": "F", "qid": qid,
        "answer": answer_final, "accuracy": accuracy, "recall": recall,
        "total_wall_s": round(total, 1),
        "llm_calls": _LLM_CALLS[0], "search_calls": _SEARCH_CALLS[0],
        "hallucination_rate": hall_rate,
        "verification": f"{yes_n}Y/{partial_n}P/{no_n}N",
        "n_claims": len(claims) if claims else 0,
        "chain_success": True,
        "search_results": [{"title": r["title"], "url": r["url"], "snippet": r["snippet"],
                            "full_text_src": r.get("full_text_source","?"),
                            "full_text_len": r.get("full_text_len", 0)}
                          for r in all_results],
        "steps": steps_log, "trace": list(_TRACE),
    }


# ══════════════════════════════════════════════════════════════════
print("=" * 70)
print("A/D/F COMPARISON: 5 NEW QUESTIONS (fully instrumented)")
print(f"BC016, BC021, BC031, BC050, BC074")
print("=" * 70)

for idx in INDICES:
    q = all_qs[idx]
    question = q["Question"]
    answer_ref = q["Answer"].strip()
    qid = f"BC{str(idx).zfill(3)}"
    topic = q.get("Topic", "?")

    print(f"\n{'='*70}")
    print(f"[{qid}] [{topic}] {question[:120]}...")
    print(f"Ref: {answer_ref}")

    q_cache = {"qid": qid, "topic": topic, "question": question, "ref": answer_ref,
               "run_time": datetime.now().isoformat(), "methods": {}}

    for method_name, run_fn in [("A", run_method_A), ("D", run_method_D), ("F", run_method_F)]:
        print(f"  [{method_name}]...", end=" ", flush=True)
        try:
            result = run_fn(question, answer_ref, qid)
            ALL_RESULTS.append(result)
            q_cache["methods"][method_name] = {
                "answer": result["answer"],
                "accuracy": result["accuracy"],
                "recall": result["recall"],
                "total_wall_s": result["total_wall_s"],
                "llm_calls": result["llm_calls"],
                "search_calls": result["search_calls"],
                "hallucination_rate": result.get("hallucination_rate"),
                "verification": result.get("verification"),
                "n_claims": result.get("n_claims", 0),
                "chain_success": result.get("chain_success"),
                "search_results": result["search_results"],
                "steps": result["steps"],
                "trace": result["trace"],
            }
        except Exception as e:
            import traceback
            traceback.print_exc()
            ALL_RESULTS.append({"method": method_name, "qid": qid, "accuracy": "ERR", "error": str(e)})
            q_cache["methods"][method_name] = {"error": str(e), "traceback": traceback.format_exc()}

    # Save per-question FULL cache (all steps, all traces)
    cache_path = CACHE_DIR / f"{qid}_cache.json"
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(q_cache, f, ensure_ascii=False, indent=2)
    print(f"  [cache] {cache_path} ({cache_path.stat().st_size} bytes)")

    # Incremental results summary
    save_json({"results": [{k: v for k, v in r.items() if k not in ("steps", "trace", "search_results")}
                           for r in ALL_RESULTS]},
              "outputs/adf_compare_results.json")

# ══════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"SUMMARY: 3 Methods x {len(INDICES)} Questions")
print(f"{'='*70}")

print(f"\n{'QID':<8} {'REF':<15} | {'A:Direct':>8} | {'D:SEVE':>14} | {'F:Chain':>14} |")
print(f"{'':8} {'':15} | {'m':>4} {'s':>4} | {'m':>4} {'s':>4} {'vf':>8} | {'m':>4} {'s':>4} {'vf':>8} |")

wins = {"A": 0, "D": 0, "F": 0}
recalls = {"A": 0, "D": 0, "F": 0}

for idx in INDICES:
    qid = f"BC{str(idx).zfill(3)}"
    ref = all_qs[idx]["Answer"].strip()[:12]
    parts = []
    for m in ["A", "D", "F"]:
        r = next((x for x in ALL_RESULTS if x.get("qid") == qid and x.get("method") == m), None)
        if r:
            acc = r.get("accuracy", "?")
            t = r.get("total_wall_s", 0)
            vf = r.get("verification", "") if m != "A" else ""
            parts.append(f"{acc:>4} {t:>4.0f}s {vf:>8}")
            if acc == "Y":
                wins[m] += 1
            if r.get("recall"):
                recalls[m] += 1
        else:
            parts.append(f"{'ERR':>4} {'?':>4}")
    print(f"{qid:<8} {ref:<15} | {parts[0]} | {parts[1]} | {parts[2]} |")

n = len(INDICES)
print(f"\n{'':24} | A={wins['A']}/{n} rec={recalls['A']}/{n} | D={wins['D']}/{n} rec={recalls['D']}/{n} | F={wins['F']}/{n} rec={recalls['F']}/{n} |")

print(f"\n{'='*70}")
print("AGGREGATE METRICS")
print(f"{'='*70}")

for m in ["A", "D", "F"]:
    m_results = [r for r in ALL_RESULTS if r.get("method") == m and r.get("accuracy") != "ERR"]
    if not m_results:
        print(f"\n  Method {m}: ALL ERRORS")
        continue
    n_m = len(m_results)
    acc_n = sum(1 for r in m_results if r["accuracy"] == "Y")
    rec_n = sum(1 for r in m_results if r.get("recall"))
    avg_time = sum(r.get("total_wall_s", 0) for r in m_results) / n_m
    avg_llm = sum(r.get("llm_calls", 0) for r in m_results) / n_m
    avg_search = sum(r.get("search_calls", 0) for r in m_results) / n_m
    hall_rates = [r.get("hallucination_rate") for r in m_results if r.get("hallucination_rate") is not None]
    avg_hall = sum(hall_rates) / len(hall_rates) if hall_rates else None

    print(f"\n  Method {m} ({n_m} completed):")
    print(f"    Accuracy:    {acc_n}/{n_m} = {acc_n/n_m*100:.0f}%")
    print(f"    Recall:      {rec_n}/{n_m} = {rec_n/n_m*100:.0f}%")
    print(f"    Avg Time:    {avg_time:.0f}s")
    print(f"    Avg LLM:     {avg_llm:.0f} calls")
    print(f"    Avg Search:  {avg_search:.0f} calls")
    if avg_hall is not None:
        print(f"    Avg Hall:    {avg_hall*100:.0f}%")

summary = {
    "questions": [{"index": i, "qid": f"BC{str(i).zfill(3)}",
                    "topic": all_qs[i]["Topic"], "ref": all_qs[i]["Answer"]}
                   for i in INDICES],
    "wins": wins, "recalls": recalls, "total": n,
    "results": [{k: v for k, v in r.items() if k not in ("steps", "trace", "search_results")}
                for r in ALL_RESULTS],
}
save_json(summary, "outputs/adf_compare_results.json")
print(f"\nFull results → outputs/adf_compare_results.json")
print(f"Per-question full caches (with steps/traces) → {CACHE_DIR}/")
