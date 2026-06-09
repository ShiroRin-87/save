import json, sys, io, glob as g
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

files = g.glob("data/adf_test_cache/BC*_prompt_ab.json")
tested_ids = set()
for f in sorted(files):
    d = json.load(open(f, "r", encoding="utf-8"))
    versions = d.get("versions", {})
    print(f"\n=== {f} ===")
    print(f"  Q: {d.get('question','')[:120]}")
    print(f"  Ref: {d.get('ref','')}")
    for vname, v in versions.items():
        recall = "HIT" if v.get("recall") else "MISS"
        hyps = v.get("hypotheses", [])
        print(f"  {vname:<30} recall={recall:<5} results={v.get('total_results',0):<4} hyps={hyps}")
