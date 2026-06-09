import json, sys, io, os, glob
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
cache = 'data/adf_test_cache'
for f in sorted(glob.glob(f'{cache}/*_new.json')):
    d = json.load(open(f, 'r', encoding='utf-8'))
    qid = os.path.basename(f).split('_')[0]
    v = d['versions']['V0']
    print(f'=== {qid}  Ref: {d["ref"]}  HIT={v["hit"]}')
    print(f'hyps={v["hypotheses"]}')
    print(f'reasoning: {len(v["reasoning"])} chars')
    lines = v['reasoning'].split('\n')
    key = [l.strip() for l in lines if any(kw in l for kw in ['线索','推断','可能是','确定','结论','答案','所以','综上','推测','得出','认为','关键','作家','小说','去世','病逝','作品','心脏病','仙侠','电视剧'])]
    for l in key[:8]:
        if len(l) > 15:
            print(f'  {l[:200]}')
    print()
