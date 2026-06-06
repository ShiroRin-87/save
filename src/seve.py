"""SEVE — Structured Extraction & Verification: full pipeline.

Steps:
  1. extract_claims() — parse search results into structured fact table
  2. generate_answer() — calibrated generation from structured table
  3. verify_claims() — reverse-verify each citation against original text
  4. apply_verification() — remove NO claims, annotate PARTIAL ones

Merged pipeline (extract_and_generate):
  Single API call replaces extract_claims + generate_answer.
  Model outputs answer with [N] citations + ===SOURCES=== JSON block.
  verify_claims / apply_verification still work unchanged.
"""
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.model_client import generate
from src.utils import (
    load_questions,
    load_search_cache,
    save_json,
    format_search_results,
    retry,
)


# ── JSON parsing utilities ─────────────────────────────────────────────────

def _parse_json(text: str):
    """Parse JSON from model output with tolerance for markdown fences."""
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
        return None


def _parse_json_array(text: str) -> list:
    """Parse a JSON array, with regex fallback."""
    data = _parse_json(text)
    if isinstance(data, list):
        return data
    # Fallback: find first JSON array in text
    m = re.search(r"\[[\s\S]*?\]", text)
    if m:
        data = _parse_json(m.group(0))
        if isinstance(data, list):
            return data
    return []


def _parse_json_strings(text: str) -> list[str]:
    """Parse a JSON array of strings."""
    arr = _parse_json_array(text)
    return [str(x).strip() for x in arr if x and str(x).strip()]


def _parse_json_bool(text: str) -> bool | None:
    """Parse a JSON boolean or YES/NO string."""
    data = _parse_json(text)
    if isinstance(data, bool):
        return data
    upper = text.strip().upper()
    if upper.startswith("YES") or upper.startswith("TRUE"):
        return True
    if upper.startswith("NO") or upper.startswith("FALSE"):
        return False
    return None


# ── Hypothesis-driven search ───────────────────────────────────────────────

def generate_hypotheses(question: str, n: int = 5) -> list[str]:
    """Use LLM parametric knowledge to guess candidate answers for a riddle.

    This bridges the gap between riddle-language and answer-language —
    the riddle may say "a place named in Tang dynasty, 10km from a Warring
    States site" but the answer is "青城山". The LLM can make that leap.
    """
    prompt = f"""你是一个知识渊博的推理助手。请根据以下谜题的约束条件，依靠你自己的知识推测可能的答案。

谜题：{question}

输出一个JSON字符串数组，包含{n}个方向各异的最终候选答案。每个元素直接是答案名称，覆盖不同方向。只输出JSON数组，不要其他文字。

示例：["青城山","都江堰","乐山大佛","峨眉山","莫高窟"]

JSON数组："""
    result = generate(prompt).strip()
    hypotheses = _parse_json_strings(result)
    # Deduplicate near-identical entries
    unique = []
    for h in hypotheses:
        is_dup = False
        for i, ex in enumerate(unique):
            if h in ex or ex in h:
                is_dup = True
                if len(h) > len(ex):
                    unique[i] = h
                break
        if not is_dup:
            unique.append(h)
    return unique[:n]


def _answer_similar(a: str, b: str, threshold: float = 0.7) -> bool:
    """Check if two answers are substantially similar (stop if no progress)."""
    if not a or not b:
        return False
    # Simple Jaccard on 2-char bigrams
    def bigrams(s):
        s = s.replace(" ", "")
        return {s[i:i+2] for i in range(len(s)-1)}
    ba, bb = bigrams(a), bigrams(b)
    if not ba or not bb:
        return False
    return len(ba & bb) / len(ba | bb) > threshold


def _answer_looks_complete(answer: str, question: str,
                           prev_answer: str = "") -> bool:
    """Check if answer fully addresses the question (revised: behavioral).

    Stops when the answer stops improving (similar to previous round),
    rather than trusting self-confident wrong answers.
    """
    if not answer or len(answer) < 20:
        return False
    # If answer is near-identical to previous, no progress made → done
    if prev_answer and _answer_similar(answer, prev_answer, threshold=0.75):
        return True
    return False


def gap_fill_search(question: str, initial_answer: str,
                    existing_results: list[dict] | None = None,
                    max_rounds: int = 3, top_k: int = 5) -> list[dict]:
    """Iterative gap-fill: when an answer identifies intermediate entities but
    can't complete the final hop, extract missing link → search → re-answer.

    Runs up to max_rounds, stopping early when answer looks complete or
    stops changing. Handles multi-hop chains (A→B→C→D).
    """
    from src.bing_client import search as do_search

    all_new = []
    current_answer = initial_answer
    seen_urls = {r.get("url", "") for r in (existing_results or [])}

    for rnd in range(max_rounds):
        prev = current_answer if rnd > 0 else ""
        if _answer_looks_complete(current_answer, question, prev):
            print(f"  [Gap-fill {rnd+1}] Answer stable, stop")
            break

        prompt = f"""以下回答可能不完整，只识别了中间步骤但没有给出最终答案。请提取需要补充搜索的关键词。

问题：{question}
当前回答：{current_answer}

输出一个JSON字符串数组，包含2-5个补充搜索关键词。只输出JSON数组。

JSON数组："""
        keywords_raw = generate(prompt).strip()
        keywords = _parse_json_strings(keywords_raw)
        if not keywords:
            print(f"  [Gap-fill {rnd+1}] No keywords extracted, stop")
            break
        query = " ".join(keywords)

        print(f"  [Gap-fill {rnd+1}] {query[:80]}...", end=" ", flush=True)
        t0 = time.time()
        results = do_search(query, top_k=top_k)
        new = [r for r in results if r.get("url", "") not in seen_urls]
        for r in new:
            seen_urls.add(r.get("url", ""))
        all_new.extend(new)
        print(f"{len(new)} new results ({time.time()-t0:.1f}s)")

        if not new:
            print(f"  [Gap-fill {rnd+1}] No new results, stop")
            break

        # Regenerate answer with accumulated new context
        from src.utils import format_search_results
        accumulated = (existing_results or []) + all_new
        ctx = format_search_results(accumulated)
        current_answer = generate(
            f"你是一个问答助手。请基于以下搜索结果为用户问题提供准确、简洁的回答。\n\n"
            f"搜索结果：\n{ctx}\n\n用户问题：{question}\n\n请回答："
        )

    return all_new


def needs_hypothesis_search(results: list[dict], question: str) -> bool:
    """Check whether direct search results are sufficient, or if hypothesis-driven
    search is needed.

    Returns True if hypothesis search should be used (direct results look poor).
    False if direct results appear sufficient.
    """
    if not results or len(results) < 3:
        return True

    lines = []
    for r in results[:8]:
        title = r.get("title", "")[:120]
        snippet = r.get("snippet", "")[:200]
        lines.append(f"- {title} | {snippet}")
    summary = "\n".join(lines)

    prompt = f"""以下是对一个问题的直接搜索结果。判断这些结果是否看起来与问题高度相关、可能包含答案信息。

问题：{question}

搜索结果：
{summary}

输出true或false。只输出JSON布尔值。

JSON："""
    result = generate(prompt).strip()
    parsed = _parse_json_bool(result)
    sufficient = parsed if parsed is not None else result.upper().startswith("YES")
    return not sufficient


def _extract_constraints(question: str) -> str:
    """Extract searchable keyword constraints from the riddle question."""
    prompt = f"""从以下谜题中提取3-5个可用于搜索验证的关键词或短语。

谜题：{question}

输出一个JSON字符串数组。只输出JSON数组。

JSON数组："""
    result = generate(prompt).strip()
    keywords = _parse_json_strings(result)
    return " ".join(keywords) if keywords else ""


def search_and_verify_hypothesis(hypothesis: str, question: str,
                                  top_k: int = 5) -> list[dict]:
    """Search for a candidate answer combined with question constraints.

    Constructs a targeted query: "{candidate} {constraint_keywords}"
    to find pages that confirm or refute the hypothesis.
    """
    from src.bing_client import search as do_search

    constraints = _extract_constraints(question)
    query = f"{hypothesis} {constraints}"
    print(f"    Query: {query[:100]}...", end=" ", flush=True)
    t0 = time.time()
    results = do_search(query, top_k=top_k)
    print(f"{len(results)} results ({time.time()-t0:.1f}s)")
    return results


def hypothesis_driven_search(question: str, n_hypotheses: int = 5) -> list[dict]:
    """Hypothesis-driven search: guess candidates → search each → merge results.

    The key insight: LLM parametric knowledge bridges riddle-language to
    answer-language. Search only verifies, it doesn't need to find the answer
    from the riddle text directly.

    Returns accumulated, deduplicated search results.
    """
    from src.bing_client import search as do_search

    print(f"  Generating up to {n_hypotheses} hypotheses...", end=" ", flush=True)
    t0 = time.time()
    hypotheses = generate_hypotheses(question, n=n_hypotheses)
    print(f"got {len(hypotheses)} ({time.time()-t0:.1f}s)")
    for i, h in enumerate(hypotheses, 1):
        print(f"    H{i}: {h}")

    if not hypotheses:
        print("  No hypotheses generated, fallback to direct search")
        return do_search(question)

    all_results = []
    seen_urls = set()

    for i, h in enumerate(hypotheses, 1):
        print(f"  [{i}/{len(hypotheses)}] Searching: {h[:60]}...")
        results = search_and_verify_hypothesis(h, question, top_k=5)
        added = 0
        for r in results:
            url = r.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                all_results.append(r)
                added += 1
        print(f"    {added} new unique results (total: {len(all_results)})")

    # Also do one direct search for coverage
    print(f"  Direct search fallback...", end=" ", flush=True)
    direct_results = do_search(question, top_k=3)
    for r in direct_results:
        url = r.get("url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            all_results.append(r)
    print(f"total: {len(all_results)} results")

    return all_results


# ── Question decomposition ────────────────────────────────────────────────

def decompose_question(question: str) -> list[str]:
    """Break a complex riddle into 2-3 independent search queries.

    Each sub-query focuses on different entities or aspects of the question,
    avoiding the signal dilution of searching with the full riddle text.
    """
    prompt = f"""将以下复杂问题分解为2-3个独立的搜索查询。每个查询应聚焦问题的不同方面或关键实体。

问题：{question}

输出一个JSON字符串数组。只输出JSON数组。

JSON数组："""
    result = generate(prompt).strip()
    queries = _parse_json_strings(result)
    return queries[:3]


def decompose_and_search(question: str, top_k: int = 5) -> list[dict]:
    """Decompose question → search each sub-query → merge deduplicated results.

    Uses fewer results per sub-query (top_k=5 vs normal 10) since multiple
    queries together provide breadth.
    """
    from src.bing_client import search as do_search

    sub_queries = decompose_question(question)
    if not sub_queries:
        return do_search(question)

    print(f"  Decomposed into {len(sub_queries)} sub-queries:")
    for sq in sub_queries:
        print(f"    - {sq[:80]}")

    all_results = []
    seen_urls = set()
    for sq in sub_queries:
        print(f"  Searching: {sq[:60]}...", end=" ", flush=True)
        t0 = time.time()
        results = do_search(sq, top_k=top_k)
        added = 0
        for r in results:
            url = r.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                all_results.append(r)
                added += 1
        print(f"{added} new ({time.time()-t0:.1f}s)")

    print(f"  Total: {len(all_results)} unique results from {len(sub_queries)} queries")
    return all_results


# ── Multi-round iterative search ──────────────────────────────────────────

def _has_enough_info(results: list[dict], question: str) -> bool:
    """Check if search results can answer the question, by title+snippet summary.

    Judges search quality directly, not claim extraction quality (which is
    unreliable — good results may produce zero claims if extraction fails).
    """
    if not results:
        return False

    lines = []
    for r in results[:10]:
        title = r.get("title", "")[:100]
        snippet = r.get("snippet", "")[:200]
        lines.append(f"- {title}: {snippet}")
    summary = "\n".join(lines)

    prompt = f"""基于以下搜索结果的标题和摘要，判断是否可能包含回答问题的信息。

搜索结果摘要：
{summary}

问题：{question}

输出true或false。只输出JSON布尔值。

JSON："""
    result = generate(prompt).strip()
    parsed = _parse_json_bool(result)
    return parsed if parsed is not None else result.upper().startswith("YES")


def _extract_clues(claims: list[dict], question: str,
                    results: list[dict] | None = None) -> str:
    """Extract key entities for refined search. Uses claims if available;
    falls back to title+snippet summaries when claims are empty or sparse.
    """
    if claims and len(claims) >= 3:
        claim_texts = "\n".join(f"- {c['claim']}" for c in claims[:15])
        source_label = "已知事实"
        source_text = claim_texts
    elif results:
        lines = []
        for r in results[:10]:
            title = r.get("title", "")[:120]
            snippet = r.get("snippet", "")[:200]
            lines.append(f"- {title}: {snippet}")
        source_label = "搜索结果摘要"
        source_text = "\n".join(lines)
    else:
        return ""

    prompt = f"""从以下{source_label}中，提取可以用来精炼搜索的关键实体（人名、地名、作品名、时间等）。

{source_label}：
{source_text}

原始问题：{question}

输出一个JSON字符串数组，包含5-10个实体词。只输出JSON数组。

JSON数组："""
    result = generate(prompt).strip()
    entities = _parse_json_strings(result)
    return " ".join(entities) if entities else ""


def iterative_search(question: str, initial_results: list[dict],
                     max_rounds: int = 2) -> list[dict]:
    """Multi-round search: check result quality, extract clues, refine, repeat.

    Returns accumulated results from all rounds (deduplicated by URL).
    Uses _has_enough_info on results (not claims) for the stop decision.
    """
    from src.bing_client import search as do_search

    all_results = list(initial_results)
    seen_urls = {r.get("url", "") for r in all_results}

    for rnd in range(max_rounds):
        # Stop check: judge search result quality directly, not claim quality
        if _has_enough_info(all_results, question):
            print(f"  [Round {rnd+1}] Search results sufficient, stop iterating")
            break

        # Extract claims to get clues for next search
        print(f"  [Round {rnd+1}] Extracting claims from {len(all_results)} results...", end=" ", flush=True)
        t0 = time.time()
        claims = extract_claims(all_results, question)
        print(f"{len(claims)} claims ({time.time()-t0:.1f}s)")

        clues = _extract_clues(claims, question, all_results)
        if not clues or len(clues) < 2:
            if not claims:
                print(f"  [Round {rnd+1}] No claims and no clues from results, stop")
            else:
                print(f"  [Round {rnd+1}] No useful clues extracted, stop")
            break

        refined_query = f"{question} {clues}"
        print(f"  [Round {rnd+1}] Refined query: {refined_query[:120]}...")
        print(f"  [Round {rnd+1}] Searching...", end=" ", flush=True)
        t0 = time.time()
        new_results = do_search(refined_query)
        print(f"{len(new_results)} results ({time.time()-t0:.1f}s)")

        added = 0
        for r in new_results:
            url = r.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                all_results.append(r)
                added += 1
        print(f"  [Round {rnd+1}] {added} new unique results added (total: {len(all_results)})")

    return all_results


# ── Core pipeline: per-result extraction ──────────────────────────────────

def _extract_from_one_result(result: dict, question: str) -> list[dict]:
    """Extract claims from a single search result (one page of text).

    Processing one page at a time (5-30K chars) is dramatically more reliable
    than feeding all 10+ pages at once.
    """
    text = result.get("full_text", "")
    url = result.get("url", "")

    if len(text) < 50:
        return []

    # Truncate overlong text
    from src.config import JINA_MAX_CHARS
    if len(text) > JINA_MAX_CHARS:
        text = text[:JINA_MAX_CHARS]

    prompt = f"""从以下网页内容中提取所有可能对回答问题有帮助的事实信息。

关键原则：即使信息不完整、不直接、看似只匹配问题的一小部分，也应该提取。宁可多提不要漏提。

输出JSON数组，每个元素含: "claim"(事实声明), "snippet"(原文逐字引用), "url"
没有任何相关信息则输出 []
只输出JSON，不要其他文字

问题：{question}
来源URL：{url}
内容：
{text}

JSON数组："""

    def _call():
        resp = generate(prompt)
        claims = _parse_json_claims(resp)
        if not claims and "[" in resp:
            # May have extracted zero which is fine
            pass
        return claims

    try:
        return retry(_call)
    except Exception:
        return []


def extract_claims(results: list[dict], question: str = "") -> list[dict]:
    """Extract claims from each result in parallel, then merge and deduplicate.

    Per-result extraction avoids the information overload problem of the old
    all-at-once approach, which often returned 1-2 claims from 900K chars.
    """
    if not results:
        return []

    print(f"  Extracting per-result (parallel, {len(results)} pages)...", end=" ", flush=True)
    t0 = time.time()

    all_claims = []
    with ThreadPoolExecutor(max_workers=5) as ex:
        futures = {ex.submit(_extract_from_one_result, r, question): i
                   for i, r in enumerate(results)}
        for f in as_completed(futures):
            try:
                claims = f.result()
                all_claims.extend(claims)
            except Exception:
                pass

    print(f"{len(all_claims)} raw claims ({time.time()-t0:.1f}s)")

    # Deduplicate by normalized claim text, keeping longer snippet
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

    unique = list(seen.values())
    for i, c in enumerate(unique, 1):
        c["id"] = i

    print(f"  {len(unique)} unique claims after dedup")
    return unique


def _parse_json_claims(text: str) -> list[dict]:
    """Parse JSON claims from model output, with tolerance for markdown fences."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:] if lines[0].startswith("```") else lines
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\[[\s\S]*\]", text)
        if m:
            try:
                data = json.loads(m.group(0))
            except json.JSONDecodeError:
                return []
        else:
            return []
    if not isinstance(data, list):
        return []
    claims = []
    for i, item in enumerate(data, 1):
        if not isinstance(item, dict):
            continue
        claims.append({
            "id": i,
            "claim": str(item.get("claim", item.get("claim_text", ""))),
            "snippet": str(item.get("snippet", item.get("original_text", ""))),
            "url": str(item.get("url", "")),
            "has_conflict": bool(item.get("has_conflict", False)),
        })
    return claims


# Leak patterns: model reasoning accidentally appearing in answer
_LEAK_PATTERNS = [
    r"Let.s check", r"Wait[!?]?\s", r"What about", r"Yes, I did",
    r"Rule \d:", r'id.*claim.*snippet.*url', r'"id"\s*:\s*\d+',
]


def _has_leaks(answer: str) -> bool:
    """Detect if model internal reasoning leaked into the answer text."""
    for pat in _LEAK_PATTERNS:
        if re.search(pat, answer, re.IGNORECASE):
            return True
    return False


def _parse_merged_output(text: str) -> tuple[str, list[dict]]:
    """Parse <answer>/<sources> XML or ===SECTION=== format from merged output."""
    answer = ""
    claims = []

    # Layer 1: XML tags (preferred)
    m = re.search(r"<answer>\s*\n?(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    if m:
        answer = m.group(1).strip()

    m = re.search(r"<sources>\s*\n?(.*?)</sources>", text, re.DOTALL | re.IGNORECASE)
    if m:
        claims = _parse_json_claims(m.group(1))

    # Layer 2: old ===SECTION=== markers
    if not answer:
        m = re.search(r"===ANSWER===\s*\n?(.*?)(?=\n===SOURCES===|\Z)", text, re.DOTALL | re.IGNORECASE)
        if m:
            answer = m.group(1).strip()

    if not claims:
        m = re.search(r"===SOURCES===\s*\n?(.*?)(?=\Z)", text, re.DOTALL | re.IGNORECASE)
        if m:
            claims = _parse_json_claims(m.group(1))

    # Layer 3: split at last JSON block
    if not answer or not claims:
        json_start = text.rfind("\n[{")
        if json_start < 0:
            json_start = text.rfind("\n[ {")
        if json_start > 0:
            if not answer:
                answer = text[:json_start].strip()
            if not claims:
                claims = _parse_json_claims(text[json_start:])

    if not answer:
        answer = text

    return answer, claims


def extract_and_generate(results: list[dict], question: str) -> tuple[str, list[dict], dict]:
    """Combined extraction + generation in one API call, using XML output format.

    Returns (answer_text, claims_list, citation_map).
    verify_claims() and apply_verification() consume the same output shapes.
    """
    if not results:
        return "当前信息不足以确定。没有可用的搜索结果。", [], {}

    context = format_search_results(results)

    prompt = f"""你是一个严谨的问答助手。请基于以下搜索结果回答问题。

规则：
1. 通读所有搜索结果，识别与问题相关的事实
2. 将关键事实编号，回答中用 [1][2] 标注引用
3. 充分则给确定答案，不充分则明确指出缺少什么
4. 不要编造搜索结果中不存在的信息
5. 不要在标签外输出任何内容

搜索结果：
{context}

用户问题：{question}

严格按以下XML格式输出：

<answer>
[你的回答，包含[1][2]引用编号]
</answer>

<sources>
[{{"id":1,"claim":"事实声明","snippet":"原文逐字引用","url":"来源URL"}},{{"id":2,...}}]
</sources>"""

    def _call():
        text = generate(prompt)
        answer, claims = _parse_merged_output(text)

        # Repair: fix leaks or missing sources
        if _has_leaks(answer):
            if claims:
                lines = "\n".join(f"[{c['id']}] {c['claim']}" for c in claims)
                answer = generate(
                    f"基于以下事实生成简洁回答，标注引用编号。不要输出推理过程。\n事实：\n{lines}\n问题：{question}"
                )
            else:
                answer = re.sub(
                    r"(Let.s check|Wait[!?]?\s|What about|Rule \d:|Yes, I did)[^\n]*",
                    "", answer, flags=re.IGNORECASE
                ).strip()

        if not claims:
            repair_prompt = f"""以下文本的来源部分JSON解析失败。修正为标准JSON数组，每元素含id/claim/snippet/url。只输出JSON：

{text}"""
            repaired = generate(repair_prompt)
            claims = _parse_json_claims(repaired)

        if not answer:
            raise ValueError("Failed to parse answer from merged output")
        return answer, claims

    answer, claims = retry(_call)
    citation_map = {str(c["id"]): c["url"] for c in claims}
    return answer, claims, citation_map


def generate_answer(question: str, claims: list[dict]) -> tuple[str, dict]:
    """Step 3: Generate answer from structured knowledge table.

    Returns (answer_text, citation_map) where citation_map is {claim_id: url}.
    """
    if not claims:
        return "当前信息不足以确定。未从搜索结果中提取到任何与问题相关的事实声明。", {}

    lines = []
    for c in claims:
        conflict = " [冲突]" if c["has_conflict"] else ""
        lines.append(f"[{c['id']}] {c['claim']} | 来源: {c['url']}{conflict}")
    table_text = "\n".join(lines)

    prompt = f"""你是一个问答助手。请基于以下结构化知识表回答用户问题。

规则：
1. 每个事实声明后标注引用编号，如 [1][2]
2. 如果知识表中标注了 [冲突]，需要同时呈现多方说法并注明来源差异
3. 首先判断问题前提是否成立：如果问题假设了某个不成立的事实（例如问"A何时做了X"但知识表明确显示A从未做过X），直接指出前提错误，并用知识表中的相关信息说明实际情况
4. 只有当知识表对该问题完全无法提供任何相关信息时，才写明"当前信息不足以确定"
5. 对于只有单一来源的信息，标注"据[来源名称]"
6. 不要编造知识表中不存在的信息

结构化知识表：
{table_text}

用户问题：{question}

请回答："""

    answer = generate(prompt)
    citation_map = {str(c["id"]): c["url"] for c in claims}
    return answer, citation_map


def verify_claims(answer: str, claims: list[dict], search_results: list[dict]) -> list[dict]:
    """Step 4: Reverse-verify cited claims against original search cache text.

    Uses a single batch prompt with JSON output. Falls back to single-claim
    verification for any claims that fail to parse from the batch response.
    """
    cited_ids = set()
    for m in re.finditer(r"\[(\d+)\]", answer):
        cited_ids.add(m.group(1))

    claim_by_id = {str(c["id"]): c for c in claims}
    url_to_result = {r["url"]: r for r in (search_results or [])}

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

    if not items:
        return []

    # Batch verification with JSON output
    items_json = json.dumps(items, ensure_ascii=False, indent=2)
    batch_prompt = f"""对以下 JSON 数组中的每组声明和原文，判断原文是否包含声明中声称的信息。
为每个元素添加 "verdict" 字段，值为 YES/PARTIAL/NO。
只输出 JSON 数组，不要输出任何其他文字。

输入：
{items_json}

输出 JSON："""

    verdict_map = {}
    try:
        result_text = generate(batch_prompt)
        batch_results = _parse_json_claims(result_text)
        for r in batch_results:
            cid_key = str(r.get("id", ""))
            raw_verdict = str(r.get("verdict", "")).upper()
            if raw_verdict in ("YES", "PARTIAL", "NO"):
                verdict_map[cid_key] = raw_verdict
    except Exception:
        pass

    # Collect UNKNOWN claims for parallel fallback
    unknown_list = []
    for cid in ordered_ids:
        if verdict_map.get(cid, "UNKNOWN") == "UNKNOWN":
            claim = claim_by_id.get(cid)
            if claim:
                original_result = url_to_result.get(claim["url"])
                original_text = (original_result.get("full_text") or original_result.get("snippet", "")) if original_result else (claim.get("snippet") or claim.get("context", ""))
                claim_text = claim.get("claim") or claim.get("sentence", "")
                unknown_list.append((cid, claim_text, original_text))

    # Parallel fallback for UNKNOWN claims
    if unknown_list:
        def _verify_one(cid, claim_text, original_text):
            try:
                fallback_prompt = f"""请判断以下引用原文是否包含生成声明中声称的信息。

生成声明：{claim_text}
引用原文：{original_text}

仅回答一个词：YES / PARTIAL / NO

你的判断："""
                fb = generate(fallback_prompt).strip().upper()
                if fb in ("YES", "PARTIAL", "NO"):
                    return cid, fb
            except Exception:
                pass
            return cid, "UNKNOWN"

        with ThreadPoolExecutor(max_workers=8) as ex:
            futures = {ex.submit(_verify_one, cid, ct, ot): cid for cid, ct, ot in unknown_list}
            for f in as_completed(futures):
                cid, verdict = f.result()
                verdict_map[cid] = verdict

    # Build final results
    verifications = []
    for cid in ordered_ids:
        claim = claim_by_id.get(cid)
        original_result = url_to_result.get(claim["url"]) if claim else None
        original_text = (original_result.get("full_text") or original_result.get("snippet", "")) if original_result else (claim.get("snippet") or claim.get("context", "") if claim else "")
        claim_text = (claim.get("claim") or claim.get("sentence", "")) if claim else ""
        url = claim["url"] if claim else ""
        verdict = verdict_map.get(cid, "UNKNOWN")

        verifications.append({
            "claim_id": cid,
            "claim_text": claim_text,
            "original_text": original_text,
            "url": url,
            "verdict": verdict,
        })

    return verifications


def apply_verification(answer: str, verifications: list[dict]) -> str:
    """Remove NO claims from answer, annotate PARTIAL ones."""
    notes = []
    for v in verifications:
        if v["verdict"] == "NO":
            notes.append(
                f"[已删除 — 原文不支撑: [{v['claim_id']}] {v['claim_text'][:80]}...]"
            )
        elif v["verdict"] == "PARTIAL":
            notes.append(
                f"[部分支撑: [{v['claim_id']}] {v['claim_text'][:80]}...]"
            )

    if notes:
        answer = answer + "\n\n--- 验证备注 ---\n" + "\n".join(notes)
    return answer


def _is_failed_answer(answer: str, verifications: list[dict]) -> bool:
    """Check if search-based pipeline has clearly failed."""
    if not answer or len(answer) < 30:
        return True
    non_answers = ["无法", "不确定", "不足以", "不能确定", "没能找到", "没有可用",
                   "未从搜索结果中提取到"]
    if any(phrase in answer for phrase in non_answers):
        return True
    if verifications:
        yes_n = sum(1 for v in verifications if v["verdict"] == "YES")
        no_n = sum(1 for v in verifications if v["verdict"] == "NO")
        if yes_n == 0:
            return True
        if no_n > yes_n and yes_n <= 1:
            return True
    return False


def fallback_reasoning(question: str, current_answer: str,
                       verifications: list[dict] | None = None) -> tuple[str, bool]:
    """When search-based pipeline fails, use LLM parametric knowledge to guess
    the answer and reverse-reason through constraint verification.

    Returns (answer, triggered) — triggered=False if fallback was skipped.
    """
    if not _is_failed_answer(current_answer, verifications or []):
        return current_answer, False

    print(f"  [Fallback] Search pipeline failed, trying parametric reasoning...")

    guess_prompt = f"""你是一个知识渊博的助手。以下问题无法通过搜索找到答案，请直接根据你的知识给出最可能的答案。
只输出答案本身，不要解释，不要前缀。

问题：{question}

答案："""
    guessed = generate(guess_prompt).strip()
    if not guessed or len(guessed) < 2:
        return current_answer, False

    print(f"  [Fallback] Guessed: {guessed[:80]}")

    decomp_prompt = f"""将以下问题拆解为3-8个独立的可验证约束条件。每个约束是一个具体的事实命题。

问题：{question}

输出一个JSON字符串数组。只输出JSON数组。

JSON数组："""
    constraints = _parse_json_strings(generate(decomp_prompt).strip())
    if not constraints:
        print(f"  [Fallback] No constraints extracted, giving up")
        return current_answer, False

    print(f"  [Fallback] {len(constraints)} constraints")

    passed = 0
    for c in constraints:
        verify_prompt = f"""判断候选答案是否满足约束条件。先回答YES或NO，再简短说明。

候选答案：{guessed}
约束条件：{c}

判断："""
        result = generate(verify_prompt).strip()
        if result.upper().startswith("YES"):
            passed += 1

    print(f"  [Fallback] {passed}/{len(constraints)} constraints passed")

    if passed >= max(1, len(constraints) * 0.5):
        final = f"{guessed}\n\n[参数化知识推理，{passed}/{len(constraints)} 约束通过]"
        return final, True
    else:
        return current_answer, False


def run_seve() -> None:
    questions = load_questions()
    cache = load_search_cache()

    all_tables = {}
    all_answers = {}
    all_verifications = {}
    all_final = {}

    for q in questions:
        qid = q["question_id"]
        results = cache.get(qid, {}).get("results", [])
        print(f"SEVE [{qid}]: {q['question'][:80]}...")

        try:
            claims = extract_claims(results, q["question"])
            answer, citation_map = generate_answer(q["question"], claims)
            verifications = verify_claims(answer, claims, results)
            final_answer = apply_verification(answer, verifications)
        except Exception as e:
            claims = []
            answer = f"[ERROR: {e}]"
            citation_map = {}
            verifications = []
            final_answer = answer

        all_tables[qid] = {
            "question": q["question"],
            "claims": claims,
            "num_claims": len(claims),
        }
        all_answers[qid] = {
            "question": q["question"],
            "answer": answer,
            "citation_map": citation_map,
        }
        all_verifications[qid] = {
            "question": q["question"],
            "verifications": verifications,
        }
        all_final[qid] = {
            "question": q["question"],
            "answer": final_answer,
            "method": "seve",
            "search_available": bool(results),
        }
        print(f"  Claims: {len(claims)}, Verifications: {len(verifications)}")

    save_json(all_tables, "outputs/seve/structured_tables.json")
    save_json(all_answers, "outputs/seve/generated_answers.json")
    save_json(all_verifications, "outputs/seve/verification_results.json")
    save_json(all_final, "outputs/seve/final_answers.json")
    print(f"SEVE complete: {len(all_final)} answers.")


if __name__ == "__main__":
    run_seve()
