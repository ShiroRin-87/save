"""Self-RAG baseline — 3-round generate-reflect-refine (prompt-based approximation).

Note: This is NOT the original fine-tuned Self-RAG (Asai et al., 2023).
It approximates the reflection mechanism via instruction prompting on Gemini API,
following the same 3-round structure but without model-trained reflection tokens.
"""
from src.model_client import generate
from src.utils import (
    load_questions,
    load_search_cache,
    save_json,
    format_search_results_numbered,
)


def run_self_rag() -> None:
    questions = load_questions()
    cache = load_search_cache()
    answers = {}

    for q in questions:
        qid = q["question_id"]
        cached = cache.get(qid, {})
        results = cached.get("results", [])
        context = (
            format_search_results_numbered(results)
            if results
            else "[无搜索结果]"
        )

        r1_answer = None
        r2_reflection = None
        final_answer = None

        try:
            r1_prompt = f"""基于以下搜索结果回答问题。在回答中，为每个事实片段标注 [Relevant] 或 [Irrelevant]。

搜索结果：
{context}

问题：{q["question"]}"""

            r1_answer = generate(r1_prompt)

            r2_prompt = f"""以下是一个回答及其标注。请检查每个声明是否被原文搜索结果支撑。
标注每个声明为 [Supported]、[Partially]、[Unsupported]。

回答：
{r1_answer}

搜索结果：
{context}"""

            r2_reflection = generate(r2_prompt)

            r3_prompt = f"""基于反思结果重新生成最终答案。只保留 [Supported] 和 [Partially] 的声明。

原始回答：
{r1_answer}

反思结果：
{r2_reflection}

用户问题：{q["question"]}"""

            final_answer = generate(r3_prompt)

        except Exception as e:
            final_answer = f"[ERROR: {e}]"

        answers[qid] = {
            "question": q["question"],
            "answer": final_answer,
            "method": "self_rag",
            "round1": r1_answer,
            "round2": r2_reflection,
            "search_available": bool(results),
        }
        print(f"Self-RAG [{qid}]: {final_answer[:100]}...")

    save_json(answers, "outputs/self_rag/answers.json")
    print(f"Done: {len(answers)} answers saved.")


if __name__ == "__main__":
    run_self_rag()
