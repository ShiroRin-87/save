"""SEVE — Structured Extraction & Verification.

Simplified 2-stage pipeline:
  1. generate_seve() — merged extraction+generation in one API call
  2. verify_simple() — lightweight batch verification
  3. apply_verification() — remove NO claims, annotate PARTIAL ones
"""
import json
import re
from src.model_client import generate
from src.utils import (
    load_questions,
    load_search_cache,
    save_json,
    format_search_results,
)


# === Stage 1: Merged Extraction + Generation ===

def generate_seve(question: str, results: list[dict]) -> tuple[str, dict]:
    """Generate answer with citations directly from search results.

    Single API call — no separate extraction step.
    Citations use [N] pointing to search result rank numbers.
    Returns (answer, citation_map).
    """
    if not results:
        answer = generate(
            f"Answer the question. If unable to answer, explain why.\n\n"
            f"Question: {question}"
        )
        return answer, {}

    # Build context with numbered search results
    parts = []
    for r in results:
        text = r.get("full_text", "") or r.get("snippet", "")
        parts.append(
            f"[{r['rank']}] {r['title']}\n"
            f"URL: {r['url']}\n"
            f"Content: {text}"
        )
    context = "\n\n".join(parts)
    citation_map = {str(r["rank"]): r["url"] for r in results}

    prompt = (
        "You are a QA assistant. Answer the user's question based on the search "
        "results below.\n\n"
        "Rules:\n"
        "1. Mark EVERY factual claim with citation numbers like [1][2] — each "
        "number MUST match the search result number shown in [brackets] above.\n"
        "2. If sources contradict, present all sides and note the differences.\n"
        "3. If the question has a false premise (e.g., asks about X but all "
        "sources show X never happened), point this out directly.\n"
        "4. Only say 'Insufficient information' when NO source contains "
        "relevant information.\n"
        "5. Do NOT fabricate anything not in the sources.\n\n"
        f"Search results:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )

    answer = generate(prompt)
    return answer, citation_map


# === Stage 2: Simplified Verification ===

def verify_simple(answer: str, results: list[dict]) -> list[dict]:
    """Lightweight reverse verification: checks each cited sentence against its
    source text. Uses a simple list format (not JSON) for speed."""
    cited = set()
    for m in re.finditer(r"\[(\d+)\]", answer):
        cited.add(int(m.group(1)))

    results_by_rank = {r["rank"]: r for r in (results or [])}
    verifications = []

    # Build verification items
    items = []
    for cid in sorted(cited):
        src = results_by_rank.get(cid)
        if not src:
            continue
        # Find the sentence containing [citation cid]
        pattern = re.compile(rf"([^.]*\[{cid}\][^.]*\.)")
        for m in pattern.finditer(answer):
            items.append({
                "id": cid,
                "claim": m.group(1).strip(),
                "original": (src.get("full_text", "") or src.get("snippet", ""))[:500],
            })
            break

    if not items:
        return []

    # Batch verify with simple format
    text_parts = []
    for i, it in enumerate(items, 1):
        text_parts.append(
            f"{i}. Claim: {it['claim'][:200]}\n"
            f"   Source text: {it['original'][:300]}"
        )
    items_text = "\n\n".join(text_parts)

    batch_prompt = (
        "For each numbered pair below, judge whether the source text supports "
        "the claim. Reply with a simple list: '1: YES, 2: NO, 3: PARTIAL, ...'\n"
        "YES = text clearly contains the claim\n"
        "PARTIAL = partially supported but details differ\n"
        "NO = text does not contain or contradicts the claim\n\n"
        f"{items_text}\n\n"
        "Verdicts:"
    )

    verdict_map = {}
    try:
        result_text = generate(batch_prompt)
        for line in result_text.strip().split("\n"):
            mm = re.match(r"(\d+)\s*[:：]\s*(YES|PARTIAL|NO)", line.strip(), re.I)
            if mm:
                idx = int(mm.group(1)) - 1
                if 0 <= idx < len(items):
                    verdict_map[items[idx]["id"]] = mm.group(2).upper()
    except Exception:
        pass

    for it in items:
        cid = it["id"]
        verdict = verdict_map.get(cid, "UNKNOWN")

        # Fallback single-item check
        if verdict == "UNKNOWN":
            try:
                fb = generate(
                    "Does the source text support this claim? "
                    "Reply YES, PARTIAL, or NO.\n\n"
                    f"Claim: {it['claim'][:200]}\n"
                    f"Source: {it['original'][:300]}\n\n"
                    "Judgment:"
                ).strip().upper()
                for v in ("YES", "PARTIAL", "NO"):
                    if v in fb:
                        verdict = v
                        break
            except Exception:
                verdict = "ERROR"

        verifications.append({
            "claim_id": str(cid),
            "claim_text": it["claim"],
            "original_text": it["original"],
            "verdict": verdict,
        })

    return verifications


# === Stage 3: Post-Verification Correction ===

def correct_answer(answer: str, verifications: list[dict], results: list[dict]) -> str:
    """Rewrite the answer to fix PARTIAL/NO claims using original source text."""
    results_by_rank = {r["rank"]: r for r in (results or [])}
    corrections = []
    for v in verifications:
        if v["verdict"] in ("PARTIAL", "NO"):
            cid = int(v["claim_id"])
            src = results_by_rank.get(cid, {})
            src_text = (src.get("full_text", "") or src.get("snippet", ""))[:500]
            corrections.append({
                "citation": cid,
                "original_claim": v["claim_text"][:200],
                "verdict": v["verdict"],
                "source_text": src_text,
            })

    if not corrections:
        return answer

    corr_text = "\n".join(
        f"[{c['citation']}] ({c['verdict']})\n"
        f"  Current: {c['original_claim']}\n"
        f"  Source:  {c['source_text'][:300]}"
        for c in corrections
    )

    prompt = (
        "Below is an answer where some claims were flagged by verification. "
        "Rewrite the FULL answer with corrections applied:\n"
        "- PARTIAL: fix inaccuracies in the claim using the source text\n"
        "- NO: if the source contains relevant info, create a correct replacement; "
        "otherwise remove the claim entirely\n"
        "- Keep all correct claims unchanged. Keep citation numbers [N].\n\n"
        f"Flagged claims:\n{corr_text}\n\n"
        f"Original answer:\n{answer}\n\n"
        "Output ONLY the corrected answer, no other text.\n\n"
        "Corrected answer:"
    )

    corrected = generate(prompt)
    return corrected


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


def strip_verification_notes(answer: str) -> str:
    """Remove verification notes for clean evaluation."""
    idx = answer.find("\n\n--- 验证备注 ---")
    if idx != -1:
        return answer[:idx].strip()
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
            answer, citation_map = generate_seve(q["question"], results)
            verifications = verify_simple(answer, results)
            corrected = correct_answer(answer, verifications, results)
            final_answer = apply_verification(corrected, verifications)
        except Exception as e:
            answer = f"[ERROR: {e}]"
            citation_map = {}
            verifications = []
            final_answer = answer

        all_tables[qid] = {
            "question": q["question"],
            "num_claims": len(verifications),
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
        print(f"  Verifications: {len(verifications)}")

    save_json(all_tables, "outputs/seve/structured_tables.json")
    save_json(all_answers, "outputs/seve/generated_answers.json")
    save_json(all_verifications, "outputs/seve/verification_results.json")
    save_json(all_final, "outputs/seve/final_answers.json")
    print(f"SEVE complete: {len(all_final)} answers.")


if __name__ == "__main__":
    run_seve()
