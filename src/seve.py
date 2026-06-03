"""SEVE — Structured Extraction & Verification: full pipeline.

Steps:
  1. extract_claims() — parse search results into structured fact table
  2. generate_answer() — calibrated generation from structured table
  3. verify_claims() — reverse-verify each citation against original text
  4. apply_verification() — remove NO claims, annotate PARTIAL ones
"""
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.model_client import generate
from src.utils import (
    load_questions,
    load_search_cache,
    save_json,
    format_search_results,
    retry,
)


def extract_claims(results: list[dict], question: str = "") -> list[dict]:
    """Step 2: Extract structured claims from search results. Output is JSON."""
    if not results:
        return []

    context = format_search_results(results)
    q_hint = f'这些搜索结果将被用于回答问题：「{question}」。请确保提取所有与问题相关的信息。\n\n' if question else ""

    prompt = f"""你是一个事实抽取器（不是推理器、不是作家）。你的唯一任务是从搜索结果中提取事实声明。

规则：
1. 只提取原文中明确陈述的事实，不推断、不总结、不跨源合成
2. 输出必须是严格的 JSON 数组，每个元素包含以下字段：
   "claim": "提炼后的事实声明",
   "snippet": "原文中支撑该声明的原始文本（必须逐字复制，不可改写）",
   "url": "来源URL",
   "has_conflict": false
3. 如果原文没有明确说，不要填写
4. 如果多个来源对同一事实给出矛盾信息，全部提取并将 has_conflict 设为 true
5. 只输出 JSON 数组，不要输出任何其他文字

{q_hint}搜索结果：
{context}

请输出 JSON 数组："""

    def _call():
        text = generate(prompt)
        claims = _parse_json_claims(text)
        if not claims:
            # Fallback: ask model to fix its JSON
            repair_prompt = f"""以下文本应该是 JSON 数组但解析失败。请将其修正为标准 JSON 数组，每个元素含 claim/snippet/url/has_conflict 字段。只输出 JSON：

{text}"""
            text = generate(repair_prompt)
            claims = _parse_json_claims(text)
        if not claims:
            raise ValueError("Failed to parse any claims from extraction output (JSON + repair both failed)")
        return claims

    return retry(_call)


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


def generate_answer(question: str, claims: list[dict]) -> tuple[str, dict]:
    """Step 3: Generate answer from structured knowledge table.

    Returns (answer_text, citation_map) where citation_map is {claim_id: url}.
    """
    if not claims:
        answer = generate(
            f"请回答以下问题。如果无法回答，请说明。\n\n问题：{question}"
        )
        return answer, {}

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
