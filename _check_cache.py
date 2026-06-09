import json, sys, io, os, glob
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
cache = 'data/adf_test_cache'
for f in sorted(glob.glob(f'{cache}/*_cache.json')):
    d = json.load(open(f, 'r', encoding='utf-8'))
    qid = os.path.basename(f).split('_')[0]
    ref = d['ref'].strip()
    for vname in d.get('versions', {}):
        if vname.startswith('V0'):
            hyps = d['versions'][vname].get('hypotheses', [])
            if not hyps:
                continue
            hit = any(ref.lower() == h.strip().lower() for h in hyps)
            loose = any(ref.lower() in h.strip().lower() or h.strip().lower() in ref.lower() for h in hyps)
            rank = next((i+1 for i,h in enumerate(hyps) if ref.lower() in h.strip().lower() or h.strip().lower() in ref.lower()), '-')
            print(f'{qid}  ref={ref}  strict={hit}  loose={loose}  rank=#{rank}  top3={hyps[:3]}')
