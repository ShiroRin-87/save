import json
qs = json.load(open("data/browsecomp-zh-decrypted.json", "r", encoding="utf-8"))
tested = {2,6,11,16,18,21,25,31,33,35,42,50,57,60,74}
for i in range(len(qs)):
    if i not in tested:
        a = qs[i]["Answer"].strip()
        print(f"[{i}] {a[:35]:<40} {qs[i]['Question'][:80]}")
