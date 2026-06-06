import json
qs = json.load(open("data/browsecomp-zh-decrypted.json", encoding="utf-8"))
for i, q in enumerate(qs):
    print(f"BC{i:03d} [{q['Topic']}] {q['Question'][:100]}... => {q['Answer']}")
