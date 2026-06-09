import json, glob, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
cache = 'data/adf_test_cache'
for f in sorted(glob.glob(f'{cache}/*_prompt_ab.json')):
    d = json.load(open(f, 'r', encoding='utf-8'))
    qid = f.split('\\')[-1].split('_')[0]
    for vname in ['V0', 'V1', 'V2']:
        v = d.get('versions', {}).get(vname, {})
        if not v:
            continue
        hyps = v.get('hypotheses', [])[:3]
        cq = v.get('constraint_query', '?')
        print(f'=== {qid} {vname}  Ref: {d.get("ref","?")} ===')
        print(f'  hyps: {hyps}')
        print(f'  constraint_query: {cq[:200]}')
        if vname == 'V2':
            sq = v.get('search_queries', '?')
            print(f'  search_queries: {str(sq)[:300]}')
        recall = v.get('recall_stats', {})
        if recall:
            print(f'  recall_hits: {recall.get("hits_in_top10", recall.get("hits", "?"))}')
            print(f'  hits_detail: {recall.get("hit_details", "")[:200]}')
        print()
