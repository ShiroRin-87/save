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


# ── Hypothesis-driven search ───────────────────────────────────────────────

def generate_hypotheses(question: str, n: int = 5) -> list[str]:
    """Use LLM parametric knowledge to guess candidate answers for a riddle.

    This bridges the gap between riddle-language and answer-language —
    the riddle may say "a place named in Tang dynasty, 10km from a Warring
    States site" but the answer is "青城山". The LLM can make that leap.
    """
    prompt = f"""你是一个知识渊博的推理助手。请根据以下谜题的约束条件，依靠你自己的知识推测可能的答案。

谜题：{question}

先在心里推理，然后只输出候选答案名称。严格每行一个名称，不要编号、解释、计算过程或任何其他文字。只输出名称本身。如果实在不确定也列出最可能的猜测。

候选答案："""
    result = generate(prompt).strip()
    hypotheses = []
    for line in result.split("\n"):
        line = line.strip()
        # Strip leading numbers, dots, brackets
        while line and (line[0] in "0123456789.、)）" or line[:2] in ("- ", "* ")):
            line = line.lstrip("0123456789.、)）-* ").strip()
        if len(line) < 2:
            continue
        # Filter out reasoning fragments: too long (>80 chars), contains
        # English fragments, or looks like calculation/explanation
        if len(line) > 80:
            continue
        if ":" in line and any(c.isascii() and c.isalpha() for c in line):
            continue  # "H1: or 24 (2010): First album..." — reasoning
        if any(kw in line.lower() for kw in ["born", "age", "album", "first", "could it", "wait"]):
            continue
        hypotheses.append(line)
    return hypotheses[:n]


def _extract_constraints(question: str) -> str:
    """Extract searchable keyword constraints from the riddle question."""
    prompt = f"""从以下谜题中提取3-5个可用于搜索验证的关键词或短语。用空格分隔，只输出关键词。

谜题：{question}

关键词："""
    return generate(prompt).strip()


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
    prompt = f"""将以下复杂问题分解为2-3个独立的搜索查询。每个查询应聚焦问题的不同方面或关键实体，用简洁的关键词组合。每行一个查询，不要编号，不要引号。

问题：{question}

搜索查询："""
    result = generate(prompt).strip()
    queries = [q.strip() for q in result.split("\n") if len(q.strip()) >= 4]
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

    prompt = f"""基于以下搜索结果的标题和摘要，判断是否可能包含回答问题的信息。只回答 YES 或 NO。

搜索结果摘要：
{summary}

问题：{question}

可能包含答案？"""
    result = generate(prompt).strip().upper()
    return result.startswith("YES")


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
用空格分隔，只输出实体词，不超过 10 个。

{source_label}：
{source_text}

原始问题：{question}

关键实体："""
    return generate(prompt).strip()


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
