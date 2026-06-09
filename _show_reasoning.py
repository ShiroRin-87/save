import json, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

for qid in ["033", "035", "060"]:
    fname = f"data/adf_test_cache/BC{qid}_prompt_ab.json"
    d = json.load(open(fname, "r", encoding="utf-8"))
    print(f"\n{'#'*60}")
    print(f"BC{qid}  Ref: {d['ref']}")
    print(f"Q: {d['question'][:150]}")

    for vname in ["V0_original", "V1_fix_hypotheses"]:
        v = d["versions"].get(vname, {})
        hyps = v.get("hypotheses", [])
        recall = "HIT" if v.get("recall") else "MISS"
        rkey = "hypotheses_reasoning" if vname == "V0_original" else "hypotheses_reasoning"
        reasoning = v.get(rkey, "")

        print(f"\n{vname}: hyps={hyps} recall={recall}")
        if reasoning:
            print(f"  reasoning ({len(reasoning)} chars):")
            print(f"  {reasoning[:1500]}")
            if len(reasoning) > 1500:
                print(f"  ...(truncated, {len(reasoning)-1500} more)")
        else:
            print("  (no reasoning saved)")
