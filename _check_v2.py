import json, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
d = json.load(open("outputs/prompt_ab_test.json", "r", encoding="utf-8"))

for vname in ["V0_original", "V1_fix_hypotheses", "V2_fix_both"]:
    v = d["versions"][vname]
    print(f"\n{'='*60}")
    print(f"{vname}: recall={v['recall']}")
    print(f"Hypotheses: {v.get('hypotheses', 'N/A')}")
    if 'queries_parsed' in v:
        print("Queries:")
        for item in v['queries_parsed']:
            if isinstance(item, dict):
                print(f"  {item.get('entity','?')}: {item.get('search_query','?')}")
    elif 'queries' in v:
        print(f"Queries: {v['queries'][:3]}...")
    print(f"\nChecking all results for '1953':")
    for r in v.get("all_results", []):
        full = (r.get("title","") + " " + r.get("snippet","") + " " + r.get("full_text","")).lower()
        if "1953" in full:
            print(f"  FOUND '1953': [{r['title'][:100]}]")
            print(f"    snippet: {r['snippet'][:300]}")
            print(f"    url: {r['url']}")
    print(f"Total results: {len(v.get('all_results',[]))}")
