"""Test: bare hypothesis search vs constraint-heavy search on BC057."""
import json, sys, time, io
from pathlib import Path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from src.bing_client import search as do_search
from src.model_client import generate
from src.seve import _parse_json_strings
from src.utils import load_json

q = load_json("data/browsecomp-zh-decrypted.json")[57]
question = q["Question"]
ref = q["Answer"].strip()

print(f"Q: {question[:150]}")
print(f"Ref: {ref}\n")

# Generate V0 hypotheses
v0_prompt = f"""你是一个知识渊博的推理助手。请根据以下谜题的约束条件，依靠你自己的知识推测可能的答案。

谜题：{question}

输出一个JSON字符串数组，包含5个方向各异的最终候选答案。每个元素直接是答案名称，覆盖不同方向。只输出JSON数组，不要其他文字。

示例：["青城山","都江堰","乐山大佛","峨眉山","莫高窟"]

JSON数组："""
raw = generate(v0_prompt).strip()
hyps = _parse_json_strings(raw)[:5]
print(f"Hyps: {hyps}\n")

# ── Method A: bare keyword search ──
print("=" * 60)
print("A: BARE SEARCH (hypo only, no constraints)")
print("=" * 60)
seen_a = set()
results_a = []
for h in hyps:
    print(f"  Search: '{h}'...", end=" ", flush=True)
    t0 = time.time()
    res = do_search(h, top_k=5)
    dt = time.time() - t0
    added = 0
    hit_one = False
    for r in res:
        if r.get("url") and r["url"] not in seen_a:
            seen_a.add(r["url"])
            results_a.append(r)
            added += 1
        full = (r.get("full_text","") + " " + r.get("snippet","") + " " + r.get("title","")).lower()
        if ref.lower() in full:
            hit_one = True
    print(f"{len(res)} results, {added} new, ref={'HIT' if hit_one else 'MISS'} ({dt:.1f}s)")
    for r in res[:2]:
        print(f"    [{r['title'][:80]}]")
        print(f"     {r['snippet'][:200]}")

overall_a = any(ref.lower() in (r.get("full_text","")+" "+r.get("snippet","")+" "+r.get("title","")).lower() for r in results_a)
print(f"\n  BARE: overall recall={'HIT' if overall_a else 'MISS'} ({len(results_a)} results)")

# ── Method B: constraint-heavy search (old way) ──
print("\n" + "=" * 60)
print("B: CONSTRAINT-HEAVY (hypo + 5 constraint phrases)")
print("=" * 60)

con_prompt = f"""从以下谜题中提取3-5个可用于搜索验证的关键词或短语。

谜题：{question}

输出一个JSON字符串数组。只输出JSON数组。

JSON数组："""
con_raw = generate(con_prompt).strip()
constraints = _parse_json_strings(con_raw)
con_str = " ".join(constraints) if constraints else ""
print(f"Constraints: {constraints}")
print(f"Concat ({len(con_str)} chars): {con_str[:150]}...\n")

seen_b = set()
results_b = []
for h in hyps:
    query = f"{h} {con_str}"
    print(f"  Search: '{query[:120]}...'...", end=" ", flush=True)
    t0 = time.time()
    res = do_search(query, top_k=5)
    dt = time.time() - t0
    added = 0
    hit_one = False
    for r in res:
        if r.get("url") and r["url"] not in seen_b:
            seen_b.add(r["url"])
            results_b.append(r)
            added += 1
        full = (r.get("full_text","") + " " + r.get("snippet","") + " " + r.get("title","")).lower()
        if ref.lower() in full:
            hit_one = True
    print(f"{len(res)} results, {added} new, ref={'HIT' if hit_one else 'MISS'} ({dt:.1f}s)")
    for r in res[:2]:
        print(f"    [{r['title'][:80]}]")
        print(f"     {r['snippet'][:200]}")

overall_b = any(ref.lower() in (r.get("full_text","")+" "+r.get("snippet","")+" "+r.get("title","")).lower() for r in results_b)
print(f"\n  HEAVY: overall recall={'HIT' if overall_b else 'MISS'} ({len(results_b)} results)")

# ── Summary ──
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  A (bare search):       recall={'HIT' if overall_a else 'MISS'} ({len(results_a)} results)")
print(f"  B (constraint-heavy):  recall={'HIT' if overall_b else 'MISS'} ({len(results_b)} results)")
