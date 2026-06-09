import json
data = json.load(open('data/browsecomp-zh-decrypted.json', encoding='utf-8'))
print(f"Total: {len(data)} questions\n")
for i, q in enumerate(data):
    qid = f"BC{str(i).zfill(3)}"
    ans = q.get('Answer', '?')[:30]
    topic = q.get('Topic', '?')
    qtext = q.get('Question', '?')[:80]
    print(f"[{i}] {qid} | {topic} | ans={ans} | {qtext}")
