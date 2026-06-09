import json, sys, io, os, glob
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

cache = 'data/adf_test_cache'
files = sorted(glob.glob(f'{cache}/*.json'))

missed = []
hit_strict = []
hit_loose = []

for f in files:
    d = json.load(open(f, 'r', encoding='utf-8'))
    ref = d['ref'].strip()
    qid = os.path.basename(f).split('_')[0]

    for vname in d.get('versions', {}):
        if not vname.startswith('V0'):
            continue
        v = d['versions'][vname]
        hyps = v.get('hypotheses', [])
        if not hyps:
            continue

        # strict match
        strict_hit = any(ref.lower() == h.strip().lower() for h in hyps)
        # loose: contains
        loose_hit = any(ref.lower() in h.strip().lower() or h.strip().lower() in ref.lower() for h in hyps)

        rank = next((i+1 for i,h in enumerate(hyps) if ref.lower() in h.strip().lower() or h.strip().lower() in ref.lower()), '-')

        if strict_hit:
            hit_strict.append((qid, ref, hyps[:3], rank))
        elif loose_hit:
            hit_loose.append((qid, ref, hyps[:3], rank))
        else:
            missed.append((qid, ref, hyps[:3], rank))

print("=== STRICT HIT ===")
for qid, ref, hyps, rank in hit_strict:
    print(f"  {qid}  ref={ref}  rank=#{rank}  top3={hyps}")

print(f"\n=== LOOSE HIT (format/partial) ===")
for qid, ref, hyps, rank in hit_loose:
    print(f"  {qid}  ref={ref}  rank=#{rank}  top3={hyps}")

print(f"\n=== TRUE MISS ===")
for qid, ref, hyps, rank in missed:
    print(f"  {qid}  ref={ref}  top3={hyps}")

total = len(hit_strict) + len(hit_loose) + len(missed)
print(f"\nTotal: {total}  |  Strict: {len(hit_strict)}  |  Loose: {len(hit_loose)}  |  Miss: {len(missed)}")
print(f"Strict rate: {len(hit_strict)}/{total} = {100*len(hit_strict)/total:.0f}%")
print(f"Strict+Loose: {len(hit_strict)+len(hit_loose)}/{total} = {100*(len(hit_strict)+len(hit_loose))/total:.0f}%")
