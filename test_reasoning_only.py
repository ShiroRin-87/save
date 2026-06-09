"""Run V0+V1 prompts on 4 new questions, save reasoning only (no search)."""
import json, sys, io
from pathlib import Path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from src.model_client import generate, get_last_reasoning
from src.seve import _parse_json_strings
from src.utils import load_json, save_json

qs = load_json("data/browsecomp-zh-decrypted.json")

for qid in [0, 1, 7, 24]:
    q = qs[qid]
    question = q["Question"]
    ref = q["Answer"].strip()
    label = f"BC{qid:03d}"
    print(f"\n{'='*60}")
    print(f"{label}  Ref: {ref}")
    print(f"Q: {question[:150]}")

    result = {"question": question, "ref": ref, "versions": {}}

    # V0
    print("\n  V0 (hypotheses)...")
    v0_prompt = f"""你是一个知识渊博的推理助手。请根据以下谜题的约束条件，依靠你自己的知识推测可能的答案。

谜题：{question}

输出一个JSON字符串数组，包含5个方向各异的最终候选答案。每个元素直接是答案名称，覆盖不同方向。只输出JSON数组，不要其他文字。

示例：["青城山","都江堰","乐山大佛","峨眉山","莫高窟"]

JSON数组："""
    raw = generate(v0_prompt).strip()
    reasoning = get_last_reasoning()
    hyps = _parse_json_strings(raw)[:5]
    print(f"    hyps={hyps}")
    if reasoning:
        print(f"    reasoning: {len(reasoning)} chars")
        # Show condensed key deduction
        lines = reasoning.split('\n')
        key_lines = [l for l in lines if any(kw in l for kw in ['线索','推断','可能是','确定','结论','答案','候选','所以','因此','综上','年月','是哪','哪个','球队','国家','年份'])]
        for l in key_lines[:15]:
            l = l.strip()
            if l and len(l) > 10:
                print(f"      {l[:150]}")
    result["versions"]["V0"] = {"hypotheses": hyps, "reasoning": reasoning}

    # V1
    print("\n  V1 (entities)...")
    v1_prompt = f"""你是一个知识渊博的推理助手。请根据以下谜题的约束条件，推测谜题中涉及的可搜索实体名称。

重要：输出的是谜题的中间实体（物质名、人名、地名、作品名等具体名词），而不是最终答案。
- 如果问题问"哪个年份"，应输出该事件涉及的具体物质/人物名称，而非年份
- 如果问题问"哪个朝代"，应输出该事物/人物的具体名称，而非朝代名
- 如果问题问"哪家公司"，应输出创始人的名字或相关产品名，而非公司名

谜题：{question}

输出一个JSON字符串数组，包含5个可搜索的中间实体名称，覆盖不同方向。只输出JSON数组，不要其他文字。

示例：["青霉素","链霉素","四环素","磺胺","红霉素"]

JSON数组："""
    raw = generate(v1_prompt).strip()
    reasoning = get_last_reasoning()
    hyps = _parse_json_strings(raw)[:5]
    print(f"    hyps={hyps}")
    if reasoning:
        print(f"    reasoning: {len(reasoning)} chars")
        lines = reasoning.split('\n')
        key_lines = [l for l in lines if any(kw in l for kw in ['线索','推断','可能是','确定','结论','答案','候选','所以','因此','综上','年月','是哪','哪个','球队','国家','年份','实体','人物','地点','组织'])]
        for l in key_lines[:15]:
            l = l.strip()
            if l and len(l) > 10:
                print(f"      {l[:150]}")
    result["versions"]["V1"] = {"hypotheses": hyps, "reasoning": reasoning}

    save_json(result, f"data/adf_test_cache/{label}_reasoning.json")

print("\nDone! 4 reasoning files saved to data/adf_test_cache/BC*_reasoning.json")
