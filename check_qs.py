import json
qs = json.load(open("data/browsecomp-zh-decrypted.json", encoding="utf-8"))
for idx in [1, 4, 7, 10, 12]:
    q = qs[idx]
    print(f"BC{idx:03d} [{q['Topic']}] {q['Question'][:120]}")
    print(f"  => {q['Answer']}")
    print()
