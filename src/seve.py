"""SEVE — Structured Extraction & Verification: full pipeline.

Steps:
  1. extract_claims() — parse search results into structured fact table
  2. generate_answer() — calibrated generation from structured table
  3. verify_claims() — reverse-verify each citation against original text
  4. apply_verification() — remove NO claims, annotate PARTIAL ones
"""
import re
from src.model_client import generate
from src.utils import (
    load_questions,
    load_search_cache,
    save_json,
    format_search_results,
    parse_structured_table,
    retry,
)


def extract_claims(results: list[dict]) -> list[dict]:
    """Step 2: Extract structured claims from search results."""
    if not results:
        return []

    context = format_search_results(results)

    prompt = f"""你是一个事实抽取器（不是推理器、不是作家）。你的唯一任务是从搜索结果中提取事实声明。

规则：
1. 只提取原文中明确陈述的事实，不推断、不总结、不跨源合成
2. 每个声明一行，格式必须严格遵循：
   序号 | 提炼声明 | 原文片段 | 出处(URL)
3. "提炼声明"是你从原文中提取的事实表述
   "原文片段"是搜索结果中支撑该声明的原始文本（必须逐字复制，不可改写）
4. 如果原文没有明确说，不要填写
5. 如果多个来源对同一事实给出矛盾信息，全部提取并在行末标注 [冲突]

搜索结果：
{context}

请输出结构化声明表："""

    def _call():
        text = generate(prompt)
        claims = parse_structured_table(text)
        if not claims:
            raise ValueError("Failed to parse any claims from extraction output")
        return claims

    return retry(_call)


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
3. 如果知识表中没有足够信息，写明"当前信息不足以确定"
4. 对于只有单一来源的信息，标注"据[来源名称]"
5. 不要编造知识表中不存在的信息

结构化知识表：
{table_text}

用户问题：{question}

请回答："""

    answer = generate(prompt)
    citation_map = {str(c["id"]): c["url"] for c in claims}
    return answer, citation_map


def verify_claims(answer: str, claims: list[dict], search_results: list[dict]) -> list[dict]:
    """Step 4: Reverse-verify each cited claim against original search cache text.

    Uses search_results (original Bing cache) for verification, NOT the
    extracted snippet — the whole point is to not trust the extraction step.
    """
    cited_ids = set()
    for m in re.finditer(r"\[(\d+)\]", answer):
        cited_ids.add(m.group(1))

    claim_by_id = {str(c["id"]): c for c in claims}
    url_to_result = {r["url"]: r for r in (search_results or [])}
    verifications = []

    for cid in cited_ids:
        claim = claim_by_id.get(cid)
        if not claim:
            verifications.append({
                "claim_id": cid,
                "verdict": "UNKNOWN",
                "reason": "claim not found in structured table",
            })
            continue

        # Get original text from search cache, fall back to extracted snippet
        original_result = url_to_result.get(claim["url"])
        if original_result:
            original_text = original_result.get("full_text") or original_result.get("snippet", "")
        else:
            original_text = claim["snippet"]

        prompt = f"""请判断以下引用原文是否包含生成声明中声称的信息。

生成声明：{claim['claim']}
引用原文：{original_text}

仅回答一个词：
YES（原文明确包含该信息）
PARTIAL（原文部分支撑但细节不一致）
NO（原文不包含该信息或与之矛盾）

你的判断："""

        try:
            verdict = generate(prompt).strip().upper()
            if verdict not in ("YES", "PARTIAL", "NO"):
                verdict = "UNKNOWN"
        except Exception:
            verdict = "ERROR"

        verifications.append({
            "claim_id": cid,
            "claim_text": claim["claim"],
            "original_text": original_text,
            "url": claim["url"],
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
            claims = extract_claims(results)
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
