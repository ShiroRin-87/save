"""Ablation experiments: wo_both and wo_rev_verify."""
import sys
from src.model_client import generate
from src.seve import generate_seve
from src.utils import (
    load_questions,
    load_search_cache,
    save_json,
    format_search_results_numbered,
)


def run_ablation_wo_both() -> None:
    """Remove both structured extraction and reverse verification.

    Uses an independent RAG + Citation prompt (not SEVE's calibrated
    generation prompt, which depends on structured knowledge table).
    """
    questions = load_questions()
    cache = load_search_cache()
    answers = {}

    for q in questions:
        qid = q["question_id"]
        results = cache.get(qid, {}).get("results", [])

        context = (
            format_search_results_numbered(results)
            if results
            else "[无搜索结果]"
        )

        prompt = f"""你是一个问答助手。请基于以下搜索结果为用户问题提供准确、简洁的回答。
每个事实声明后标注引用编号，如 [1][2]，对应搜索结果的序号。

搜索结果：
{context}

用户问题：{q["question"]}

请回答："""

        try:
            answer = generate(prompt)
        except Exception as e:
            answer = f"[ERROR: {e}]"

        answers[qid] = {
            "question": q["question"],
            "answer": answer,
            "method": "ablation_wo_both",
            "search_available": bool(results),
        }
        print(f"wo_both [{qid}]: {answer[:100]}...")

    save_json(answers, "outputs/ablation_wo_both/answers.json")
    print(f"wo_both done: {len(answers)} answers.")


def run_ablation_wo_rev_verify() -> None:
    """Keep structured extraction + calibrated generation, skip verification."""
    questions = load_questions()
    cache = load_search_cache()
    answers = {}

    for q in questions:
        qid = q["question_id"]
        results = cache.get(qid, {}).get("results", [])

        try:
            answer, citation_map = generate_seve(q["question"], results)
        except Exception as e:
            answer = f"[ERROR: {e}]"
            citation_map = {}

        answers[qid] = {
            "question": q["question"],
            "answer": answer,
            "citation_map": citation_map,
            "method": "ablation_wo_rev_verify",
            "search_available": bool(results),
        }
        print(f"wo_rev_verify [{qid}]: {answer[:100]}...")

    save_json(answers, "outputs/ablation_wo_rev_verify/answers.json")
    print(f"wo_rev_verify done: {len(answers)} answers.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "wo_rev_verify":
        run_ablation_wo_rev_verify()
    else:
        run_ablation_wo_both()
