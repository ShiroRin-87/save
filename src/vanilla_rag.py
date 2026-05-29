"""Vanilla RAG baseline — generate answers directly from search results."""
from src.model_client import generate
from src.utils import (
    load_questions,
    load_search_cache,
    save_json,
    format_search_results_numbered,
)


def run_vanilla_rag() -> None:
    questions = load_questions()
    cache = load_search_cache()
    answers = {}

    for q in questions:
        qid = q["question_id"]
        cached = cache.get(qid, {})
        results = cached.get("results", [])

        if results:
            context = format_search_results_numbered(results)
        else:
            context = "[无搜索结果]"

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
            "method": "vanilla_rag",
            "search_available": bool(results),
        }
        print(f"Vanilla RAG [{qid}]: {answer[:100]}...")

    save_json(answers, "outputs/vanilla_rag/answers.json")
    print(f"Done: {len(answers)} answers saved.")


if __name__ == "__main__":
    run_vanilla_rag()
