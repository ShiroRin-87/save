import json, glob, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
cache = 'data/adf_test_cache'
for f in sorted(glob.glob(f'{cache}/*_prompt_ab.json')):
    d = json.load(open(f, 'r', encoding='utf-8'))
    qid = f.split('\\')[-1].split('_')[0]
    for vname in ['V0_original', 'V1', 'V2']:
        v = d.get('versions', {}).get(vname, {})
        if not v:
            continue
        hyps = v.get('hypotheses', [])[:3]
        constraints = v.get('constraints', [])
        queries = v.get('queries', [])
        print(f'=== {qid} {vname}  Ref: {d.get("ref","?")} ===')
        print(f'  hypotheses: {hyps}')
        print(f'  constraints: {constraints}')
        print(f'  query_example: {queries[0] if queries else "N/A"}')
        print()
