import json, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

d = json.load(open("data/adf_test_cache/BC057_prompt_ab.json", "r", encoding="utf-8"))
v0 = d["versions"]["V0_original"]

print("=== Search detail ===")
for sd in v0["search_detail"]:
    print(f"\nQuery: {sd['query'][:120]}")
    print(f"Results: {sd['n_results']}, New: {sd['n_new']}")
    for r in sd["top3"]:
        print(f"  [{r['title'][:100]}]")
        print(f"   {r['snippet'][:250]}")
