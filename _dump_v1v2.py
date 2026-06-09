import json, glob, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
cache = 'data/adf_test_cache'
for f in sorted(glob.glob(f'{cache}/*_prompt_ab.json')):
    d = json.load(open(f, 'r', encoding='utf-8'))
    qid = f.split('\\')[-1].split('_')[0]
    ref = d.get('ref', '?')

    v0 = d['versions'].get('V0_original', {})
    v1 = d['versions'].get('V1_fix_hypotheses', {})
    v2 = d['versions'].get('V2_fix_both', {})

    print(f'========== {qid}  Ref: {ref} ==========')

    # V0
    print(f'\n--- V0 (hypotheses=最终答案, constraints=谜题提取) ---')
    print(f'  hyps: {v0.get("hypotheses",[])[:3]}')
    print(f'  constraints: {v0.get("constraints",[])}')

    # V1
    print(f'\n--- V1 (entities=中间实体, constraints=谜题提取) ---')
    print(f'  entities: {v1.get("hypotheses",[])[:3]}')
    print(f'  constraints: {v1.get("constraints",[])}')

    # V2
    print(f'\n--- V2 (entities=中间实体, queries=LLM生成搜索词) ---')
    print(f'  entities: {v2.get("hypotheses",[])[:3]}')
    sq = v2.get('search_queries', [])
    if isinstance(sq, list):
        for s in sq[:3]:
            print(f'  LLM_query: {s}')
    else:
        print(f'  LLM_query: {sq}')

    print()
