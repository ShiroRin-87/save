import json
qs = json.load(open("data/browsecomp-zh-decrypted.json", "r", encoding="utf-8"))
tested = {11, 16, 21, 31, 50, 74}
for i, q in enumerate(qs):
    if i not in tested:
        ans = q["Answer"].strip()
        print(f"[{i}] {ans[:40]:<45} {q['Question'][:100]}")
