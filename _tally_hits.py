import json, sys, io, os, glob
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

cache = 'data/adf_test_cache'
rows = []

# prompt_ab files (old tests)
for f in sorted(glob.glob(f'{cache}/*_prompt_ab.json')):
    d = json.load(open(f, 'r', encoding='utf-8'))
    ref = d['ref'].strip().lower()
    qid = os.path.basename(f).split('_')[0]
    for vname in ['V0_original', 'V1_fix_hypotheses', 'V2_search_entities']:
        v = d['versions'].get(vname, {})
        hyps = v.get('hypotheses', [])
        if not hyps:
            continue
        hit = any(ref == h.strip().lower() for h in hyps)
        rank = next((i+1 for i,h in enumerate(hyps) if ref == h.strip().lower()), '-')
        rows.append((qid, vname[:2], 'HIT' if hit else 'MISS', rank,
                     ', '.join(str(x)[:40] for x in hyps[:3])))

# reasoning files (new tests)
for f in sorted(glob.glob(f'{cache}/*_reasoning.json')):
    d = json.load(open(f, 'r', encoding='utf-8'))
    ref = d['ref'].strip().lower()
    qid = os.path.basename(f).split('_')[0]
    for vname in ['V0', 'V1']:
        v = d['versions'].get(vname, {})
        hyps = v.get('hypotheses', [])
        if not hyps:
            continue
        hit = any(ref == h.strip().lower() for h in hyps)
        rank = next((i+1 for i,h in enumerate(hyps) if ref == h.strip().lower()), '-')
        rows.append((qid, vname[:2], 'HIT' if hit else 'MISS', rank,
                     ', '.join(str(x)[:40] for x in hyps[:3])))

# summary
v0_hit = v0_miss = 0
v1_hit = v1_miss = 0
v2_hit = v2_miss = 0

print(f'{"QID":<7} {"V":<4} {"R":<5} {"Rank":<5} Top-3 hyps')
print('-' * 80)
for r in rows:
    print(f'{r[0]:<7} {r[1]:<4} {r[2]:<5} {r[3]:<5} {r[4]}')
    if r[1] == 'V0':
        if r[2] == 'HIT': v0_hit += 1
        else: v0_miss += 1
    elif r[1] == 'V1':
        if r[2] == 'HIT': v1_hit += 1
        else: v1_miss += 1
    elif r[1] == 'V2':
        if r[2] == 'HIT': v2_hit += 1
        else: v2_miss += 1

print()
v0_total = v0_hit + v0_miss
v1_total = v1_hit + v1_miss
v2_total = v2_hit + v2_miss
if v0_total: print(f'V0: {v0_hit}/{v0_total} = {100*v0_hit/v0_total:.0f}% hit')
if v1_total: print(f'V1: {v1_hit}/{v1_total} = {100*v1_hit/v1_total:.0f}% hit')
if v2_total: print(f'V2: {v2_hit}/{v2_total} = {100*v2_hit/v2_total:.0f}% hit')
