"""Run Vanilla RAG and Self-RAG on Saga Planets question."""
import json, sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.model_client import generate
from src.utils import load_search_cache, format_search_results_numbered, save_json

QID = "CUSTOM_SAGA"
QUESTION = "SAGA PLANETS是在哪一部作品后离开Visual Arts的？"

cache = load_search_cache()
results = cache.get(QID, {}).get("results", [])
context = format_search_results_numbered(results) if results else "[无搜索结果]"

# === Vanilla RAG ===
print("=" * 60)
print("VANILLA RAG")
print("=" * 60)
v_prompt = f"""你是一个问答助手。请基于以下搜索结果为用户问题提供准确、简洁的回答。
每个事实声明后标注引用编号，如 [1][2]，对应搜索结果的序号。

搜索结果：
{context}

用户问题：{QUESTION}

请回答："""
v_answer = generate(v_prompt)
print(v_answer)

# === Self-RAG ===
print("\n" + "=" * 60)
print("SELF-RAG — Round 1")
print("=" * 60)
r1 = generate(f"""基于以下搜索结果回答问题。在回答中，为每个事实片段标注 [Relevant] 或 [Irrelevant]。

搜索结果：
{context}

问题：{QUESTION}""")
print(r1)

print("\n" + "=" * 60)
print("SELF-RAG — Round 3 (Final)")
print("=" * 60)
r2 = generate(f"""以下是一个回答及其标注。请检查每个声明是否被原文搜索结果支撑。
标注每个声明为 [Supported]、[Partially]、[Unsupported]。

回答：
{r1}

搜索结果：
{context}""")
r3 = generate(f"""基于反思结果重新生成最终答案。只保留 [Supported] 和 [Partially] 的声明。

原始回答：
{r1}

反思结果：
{r2}

用户问题：{QUESTION}""")
print(r3)

# Save
try:
    existing_v = json.loads((ROOT / "outputs/vanilla_rag/answers.json").read_text("utf-8"))
except: existing_v = {}
existing_v[QID] = {"question": QUESTION, "answer": v_answer, "method": "vanilla_rag", "search_available": bool(results)}
save_json(existing_v, "outputs/vanilla_rag/answers.json")

try:
    existing_s = json.loads((ROOT / "outputs/self_rag/answers.json").read_text("utf-8"))
except: existing_s = {}
existing_s[QID] = {"question": QUESTION, "answer": r3, "method": "self_rag", "round1": r1, "round2": r2, "search_available": bool(results)}
save_json(existing_s, "outputs/self_rag/answers.json")

print("\nDone.")
