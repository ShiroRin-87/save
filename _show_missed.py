import json, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

qs = json.load(open("data/browsecomp-zh-decrypted.json", "r", encoding="utf-8"))
for qid in [18, 21, 33, 35, 60]:
    q = qs[qid]
    print(f"\n{'='*60}")
    print(f"[{qid}] Ref: {q['Answer'].strip()}")
    print(f"Q: {q['Question']}")
    print()
