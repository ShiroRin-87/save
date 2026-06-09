"""Compare two prompt fixes on BC021, saving all intermediate outputs.

V0: Original (hypotheses = final answers)
V1: Fix generate_hypotheses only (output searchable entities)
V2: Fix both (hypotheses = entities + LLM generates search queries)

Saves EVERYTHING: raw LLM responses, search queries, search results.
"""
import json, sys, time, re
from pathlib import Path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.model_client import generate, get_last_reasoning
from src.seve import _parse_json_strings
from src.bing_client import search as do_search
from src.utils import load_json, save_json

# BC021
qid = int(sys.argv[1]) if len(sys.argv) > 1 else 2
q = load_json("data/browsecomp-zh-decrypted.json")[qid]
question = q["Question"]
answer_ref = q["Answer"].strip()
label = f"BC{qid:03d}"
print(f"{label}: {question[:150]}...")
print(f"Ref: {answer_ref}")

SAVE = {"question": question, "ref": answer_ref, "versions": {}}

# ── Helpers ──
def _parse_json_any(text):
    """Parse JSON from model output, tolerant of markdown fences."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:] if lines[0].startswith("```") else lines
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\[[\s\S]*\]", text)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return None

def check_recall(results, ref):
    ref = ref.strip().lower()
    for r in results:
        full = (r.get("full_text", "") + " " + r.get("snippet", "") + " " + r.get("title", "")).lower()
        if ref in full:
            return True
    return False

def search_and_collect(pairs_list, label):
    """Search with each query, collect all results."""
    seen_urls = set()
    all_results = []
    detail = []
    for item in pairs_list:
        if isinstance(item, dict):
            entity = item.get("entity", "")
            query = item.get("search_query", item.get("query", ""))
        else:
            entity = str(item)
            query = str(item)
        if not query:
            continue

        print(f"  Search: {query[:100]}...", end=" ", flush=True)
        t0 = time.time()
        results = do_search(query, top_k=5)
        elapsed = time.time() - t0

        added = 0
        for r in results:
            url = r.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                all_results.append(r)
                added += 1

        top_snippets = [{"title": r["title"][:100], "snippet": r["snippet"][:200], "url": r["url"]}
                        for r in results[:3]]
        detail.append({
            "entity": entity, "query": query,
            "n_results": len(results), "n_new": added,
            "wall_s": round(elapsed, 1),
            "top3": top_snippets,
        })
        print(f"{len(results)} results, {added} new ({elapsed:.1f}s)")

    return all_results, detail


# ═══════════════════════════════════════════════════════
# V0: Original
# ═══════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("V0: ORIGINAL (hypotheses = final answers)")
print(f"{'='*60}")

# Generate hypotheses
v0_prompt = f"""你是一个知识渊博的推理助手。请根据以下谜题的约束条件，依靠你自己的知识推测可能的答案。

谜题：{question}

输出一个JSON字符串数组，包含5个方向各异的最终候选答案。每个元素直接是答案名称，覆盖不同方向。只输出JSON数组，不要其他文字。

示例：["青城山","都江堰","乐山大佛","峨眉山","莫高窟"]

JSON数组："""
v0_hypo_raw = generate(v0_prompt).strip()
v0_hypo_reasoning = get_last_reasoning()
v0_hypotheses = _parse_json_strings(v0_hypo_raw)[:5]
print(f"Hypotheses: {v0_hypotheses}")

# Extract constraints
v0_con_prompt = f"""从以下谜题中提取3-5个可用于搜索验证的关键词或短语。

谜题：{question}

输出一个JSON字符串数组。只输出JSON数组。

JSON数组："""
v0_con_raw = generate(v0_con_prompt).strip()
v0_con_reasoning = get_last_reasoning()
v0_constraints = _parse_json_strings(v0_con_raw)
v0_con_str = " ".join(v0_constraints) if v0_constraints else ""
print(f"Constraints: {v0_constraints}")

# Build queries
v0_pairs = [f"{h} {v0_con_str}" for h in v0_hypotheses]
print(f"Queries:")
for qry in v0_pairs:
    print(f"  {qry[:120]}")

# Search
v0_results, v0_detail = search_and_collect(v0_pairs, "V0")
v0_recall = check_recall(v0_results, answer_ref)
print(f"Recall: {'HIT' if v0_recall else 'MISS'} ({len(v0_results)} total results)")

SAVE["versions"]["V0_original"] = {
    "hypotheses_prompt": v0_prompt,
    "hypotheses_raw": v0_hypo_raw,
    "hypotheses_reasoning": v0_hypo_reasoning,
    "hypotheses": v0_hypotheses,
    "constraints_prompt": v0_con_prompt,
    "constraints_raw": v0_con_raw,
    "constraints_reasoning": v0_con_reasoning,
    "constraints": v0_constraints,
    "queries": v0_pairs,
    "search_detail": v0_detail,
    "total_results": len(v0_results),
    "recall": v0_recall,
    "all_results": [{"title": r["title"], "url": r["url"], "snippet": r["snippet"]} for r in v0_results],
}

# ═══════════════════════════════════════════════════════
# V1: Fix generate_hypotheses only
# ═══════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("V1: FIX HYPOTHESES ONLY (entities, not final answers)")
print(f"{'='*60}")

v1_prompt = f"""你是一个知识渊博的推理助手。请根据以下谜题的约束条件，推测谜题中涉及的可搜索实体名称。

重要：输出的是谜题的中间实体（物质名、人名、地名、作品名等具体名词），而不是最终答案。
- 如果问题问"哪个年份"，应输出该事件涉及的具体物质/人物名称，而非年份
- 如果问题问"哪个朝代"，应输出该事物/人物的具体名称，而非朝代名
- 如果问题问"哪家公司"，应输出创始人的名字或相关产品名，而非公司名

谜题：{question}

输出一个JSON字符串数组，包含5个可搜索的中间实体名称，覆盖不同方向。只输出JSON数组，不要其他文字。

示例：["青霉素","链霉素","四环素","磺胺","红霉素"]

JSON数组："""
v1_hypo_raw = generate(v1_prompt).strip()
v1_hypo_reasoning = get_last_reasoning()
v1_hypotheses = _parse_json_strings(v1_hypo_raw)[:5]
print(f"Hypotheses: {v1_hypotheses}")

# Same constraints as V0
print(f"Constraints (same as V0): {v0_constraints}")
v1_pairs = [f"{h} {v0_con_str}" for h in v1_hypotheses]
print(f"Queries:")
for qry in v1_pairs:
    print(f"  {qry[:120]}")

v1_results, v1_detail = search_and_collect(v1_pairs, "V1")
v1_recall = check_recall(v1_results, answer_ref)
print(f"Recall: {'HIT' if v1_recall else 'MISS'} ({len(v1_results)} total results)")

SAVE["versions"]["V1_fix_hypotheses"] = {
    "hypotheses_prompt": v1_prompt,
    "hypotheses_raw": v1_hypo_raw,
    "hypotheses_reasoning": v1_hypo_reasoning,
    "hypotheses": v1_hypotheses,
    "constraints": v0_constraints,
    "constraints_str": v0_con_str,
    "queries": v1_pairs,
    "search_detail": v1_detail,
    "total_results": len(v1_results),
    "recall": v1_recall,
    "all_results": [{"title": r["title"], "url": r["url"], "snippet": r["snippet"]} for r in v1_results],
}

# ═══════════════════════════════════════════════════════
# V2: Fix both — LLM generates queries directly
# ═══════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("V2: FIX BOTH (entities + LLM-crafted search queries)")
print(f"{'='*60}")

# Same entity hypotheses as V1
print(f"Hypotheses (same as V1): {v1_hypotheses}")

hlist = "\n".join(f"- {h}" for h in v1_hypotheses)
v2_query_prompt = f"""对于以下谜题和候选实体，为每个实体生成一个用于Google搜索验证的关键词组合。关键词应：
1. 包含实体名本身
2. 加入谜题中对验证该实体最关键的1-3个约束词（只选事实性约束如"中国""投产""年份"，不要谜语修辞如"绿色山脉""小人物"）
3. 简短精准，不超过8个词

谜题：{question}

候选实体：
{hlist}

输出JSON数组，每个元素含 "entity" 和 "search_query" 两字段。只输出JSON数组。

JSON数组："""
v2_query_raw = generate(v2_query_prompt).strip()
v2_query_reasoning = get_last_reasoning()
v2_pairs = _parse_json_any(v2_query_raw) or []
print(f"Queries:")
for item in v2_pairs:
    if isinstance(item, dict):
        print(f"  {item.get('entity','?')}: {item.get('search_query','?')[:120]}")

v2_results, v2_detail = search_and_collect(v2_pairs, "V2")
v2_recall = check_recall(v2_results, answer_ref)
print(f"Recall: {'HIT' if v2_recall else 'MISS'} ({len(v2_results)} total results)")

SAVE["versions"]["V2_fix_both"] = {
    "hypotheses": v1_hypotheses,
    "query_prompt": v2_query_prompt,
    "query_raw": v2_query_raw,
    "query_reasoning": v2_query_reasoning,
    "queries_parsed": v2_pairs,
    "search_detail": v2_detail,
    "total_results": len(v2_results),
    "recall": v2_recall,
    "all_results": [{"title": r["title"], "url": r["url"], "snippet": r["snippet"]} for r in v2_results],
}

# ═══════════════════════════════════════════════════════
# SUMMARY + SAVE
# ═══════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
for name, v in SAVE["versions"].items():
    print(f"  {name:<30} recall={'HIT' if v['recall'] else 'MISS'}  results={v['total_results']}")

save_json(SAVE, "outputs/prompt_ab_test.json")
print(f"\nFull results saved to outputs/prompt_ab_test.json")
print(f"Also saved to data/adf_test_cache/{label}_prompt_ab.json")
save_json(SAVE, f"data/adf_test_cache/{label}_prompt_ab.json")
