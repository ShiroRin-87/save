import json, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
qs = json.load(open("data/browsecomp-zh-decrypted.json", "r", encoding="utf-8"))
for i in [3, 5]:
    q = qs[i]
    print(f"[{i}] Ref: {q['Answer'].strip()}")
    print(f"Q: {q['Question']}")
    print()
