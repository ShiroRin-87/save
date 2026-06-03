"""Quick 5-question test with timing."""
import sys, json, time, random
sys.path.insert(0, '.')
import numpy as np

from src.bing_client import search
from src.model_client import generate
from src.seve import extract_claims, generate_answer as seve_gen, verify_claims, apply_verification, strip_verification_notes
from src.evaluator import nli_check, compute_factscore, compute_citation_precision, compute_answer_accuracy, compute_answer_recall, _join_results, bootstrap_test
from src.utils import save_json, format_search_results_numbered

all_qs = json.load(open('data/freshqa_questions.json', encoding='utf-8'))
random.seed(42)
sample = random.sample(all_qs, 5)
print(f'Sampled {len(sample)} questions')
timings = {}

# Step 1: Search
print('\n=== Step 1: Search + Jina ===')
t0 = time.time()
cache = {}
for i, q in enumerate(sample):
    qid = q['question_id']
    t1 = time.time()
    results = search(q['question'], top_k=5, use_jina=True)
    dt = time.time() - t1
    cache[qid] = {'query': q['question'], 'results': results}
    jina_count = sum(1 for r in results if r.get('full_text_source') == 'jina')
    lens = [r.get('full_text_len', 0) for r in results]
    print(f'  [{i+1}/5] {qid}: {len(results)} results, jina={jina_count}, lens={lens} | {dt:.1f}s')
    if i < 4:
        time.sleep(0.3)
save_json(cache, 'data/search_cache.json')
timings['search'] = time.time() - t0
print(f'  Done: {timings["search"]:.1f}s ({timings["search"]/5:.1f}s/q)')

# Step 2: Vanilla RAG
print('\n=== Step 2: Vanilla RAG ===')
t0 = time.time()
vanilla_out = {}
for i, q in enumerate(sample):
    qid = q['question_id']
    results = cache[qid]['results']
    ctx = format_search_results_numbered(results) if results else '[No results]'
    prompt = f'You are a helpful assistant. Based on the search results, answer the question accurately.\nMark each claim with citation numbers like [1][2].\n\nSearch results:\n{ctx}\n\nQuestion: {q["question"]}\n\nAnswer:'
    t1 = time.time()
    ans = generate(prompt)
    dt = time.time() - t1
    vanilla_out[qid] = {'question': q['question'], 'answer': ans, 'method': 'vanilla_rag', 'search_available': bool(results)}
    print(f'  [{i+1}/5] {qid}: {dt:.1f}s | {ans[:100]}')
save_json(vanilla_out, 'outputs/vanilla_rag/answers.json')
timings['vanilla'] = time.time() - t0
print(f'  Done: {timings["vanilla"]:.1f}s ({timings["vanilla"]/5:.1f}s/q)')

# Step 3: Self-RAG
print('\n=== Step 3: Self-RAG ===')
t0 = time.time()
self_out = {}
for i, q in enumerate(sample):
    qid = q['question_id']
    results = cache[qid]['results']
    ctx = format_search_results_numbered(results) if results else '[No results]'
    t1 = time.time()
    r1 = generate(f'Based on search results, answer the question. Label each claim [Relevant] or [Irrelevant].\n\nSearch results:\n{ctx}\n\nQuestion: {q["question"]}')
    t2 = time.time()
    r2 = generate(f'Check each claim against the search results. Label [Supported]/[Partially]/[Unsupported].\n\nAnswer:\n{r1}\n\nSearch results:\n{ctx}')
    t3 = time.time()
    r3 = generate(f'Regenerate keeping only [Supported] and [Partially] claims.\n\nOriginal answer:\n{r1}\n\nReflection:\n{r2}\n\nQuestion: {q["question"]}')
    dt = time.time() - t1
    self_out[qid] = {'question': q['question'], 'answer': r3, 'method': 'self_rag', 'round1': r1, 'round2': r2, 'search_available': bool(results)}
    print(f'  [{i+1}/5] {qid}: total={dt:.1f}s | {(r3 or "")[:80]}')
save_json(self_out, 'outputs/self_rag/answers.json')
timings['self_rag'] = time.time() - t0
print(f'  Done: {timings["self_rag"]:.1f}s ({timings["self_rag"]/5:.1f}s/q)')

# Step 4: SEVE
print('\n=== Step 4: SEVE ===')
t0 = time.time()
seve_final_out, seve_tables, seve_gen_out, seve_verif = {}, {}, {}, {}
t_extract_total, t_gen_total, t_verify_total = 0, 0, 0
for i, q in enumerate(sample):
    qid = q['question_id']
    results = cache[qid]['results']
    t1 = time.time()
    claims = extract_claims(results, q['question'])
    te = time.time() - t1
    t2 = time.time()
    answer, cmap = seve_gen(q['question'], claims)
    tg = time.time() - t2
    t3 = time.time()
    verifs = verify_claims(answer, claims, results)
    tv = time.time() - t3
    final = apply_verification(answer, verifs)
    t_extract_total += te; t_gen_total += tg; t_verify_total += tv
    seve_tables[qid] = {'question': q['question'], 'claims': claims, 'num_claims': len(claims)}
    seve_gen_out[qid] = {'question': q['question'], 'answer': answer, 'citation_map': cmap}
    seve_verif[qid] = {'question': q['question'], 'verifications': verifs}
    seve_final_out[qid] = {'question': q['question'], 'answer': final, 'method': 'seve', 'search_available': bool(results)}
    print(f'  [{i+1}/5] {qid}: {te+tg+tv:.1f}s (extract={te:.1f}s gen={tg:.1f}s verify={tv:.1f}s) | {len(claims)}c {len(verifs)}v | {strip_verification_notes(final)[:80]}')
save_json(seve_tables, 'outputs/seve/structured_tables.json')
save_json(seve_gen_out, 'outputs/seve/generated_answers.json')
save_json(seve_verif, 'outputs/seve/verification_results.json')
save_json(seve_final_out, 'outputs/seve/final_answers.json')
timings['seve'] = time.time() - t0
print(f'  extract={t_extract_total:.0f}s gen={t_gen_total:.0f}s verify={t_verify_total:.0f}s total={timings["seve"]:.1f}s')

# Step 5: Evaluation
print('\n=== Step 5: Evaluation ===')
t0 = time.time()
all_scores = {}
for method, answers in [('vanilla_rag', vanilla_out), ('self_rag', self_out), ('seve', seve_final_out)]:
    print(f'  Evaluating {method}...')
    scores = {'factscore': [], 'cit_precision': [], 'ans_accuracy': [], 'ans_recall': [], 'refusal_acc': []}
    for qid, a in answers.items():
        qinfo = next((x for x in sample if x['question_id'] == qid), {})
        answer_text = strip_verification_notes(a.get('answer', ''))
        results = cache[qid]['results']
        ref = qinfo.get('reference_answer', '')
        fs = compute_factscore(answer_text, results)
        cp = compute_citation_precision(answer_text, results)
        aa = compute_answer_accuracy(answer_text, ref) if ref else None
        ar = compute_answer_recall(answer_text, ref) if ref else None
        scores['factscore'].append(fs)
        if cp is not None: scores['cit_precision'].append(cp)
        if aa is not None: scores['ans_accuracy'].append(aa)
        if ar is not None: scores['ans_recall'].append(ar)
        is_unanswerable = not qinfo.get('is_answerable', True)
        if is_unanswerable:
            refused = any(kw in answer_text for kw in ['信息不足','无法','不确定','信息不足以确定'])
            scores['refusal_acc'].append(1.0 if refused else 0.0)
    all_scores[method] = scores
timings['eval'] = time.time() - t0
print(f'  Done: {timings["eval"]:.1f}s')

# Results
print('\n' + '=' * 70)
print('RESULTS')
print('=' * 70)
print(f'{"Method":<20} {"FactScore":>10} {"CitPrec":>10} {"AnsAcc":>10} {"AnsRec":>10}')
print('-' * 70)
for method, scores in all_scores.items():
    fs = f'{np.mean(scores["factscore"]):.3f}' if scores['factscore'] else 'N/A'
    cp = f'{np.mean(scores["cit_precision"]):.3f}' if scores['cit_precision'] else 'N/A'
    aa = f'{np.mean(scores["ans_accuracy"]):.3f}' if scores['ans_accuracy'] else 'N/A'
    ar = f'{np.mean(scores["ans_recall"]):.3f}' if scores['ans_recall'] else 'N/A'
    print(f'{method:<20} {fs:>10} {cp:>10} {aa:>10} {ar:>10}')

# Bootstrap
print(f'\n--- Bootstrap ---')
seve_sc = all_scores.get('seve', {})
for baseline in ['vanilla_rag', 'self_rag']:
    bs = all_scores.get(baseline, {})
    if not seve_sc or not bs: continue
    for metric in ['factscore']:
        a = seve_sc.get(metric, [])
        b = bs.get(metric, [])
        if len(a) == len(b) and len(a) > 0:
            bt = bootstrap_test(a, b)
            print(f'  SEVE vs {baseline} [{metric}]: diff={bt["observed_diff"]:.3f} p={bt["p_value"]:.3f}')

# Timing
print(f'\n--- Timing ---')
total = sum(timings.values())
for k, v in timings.items():
    print(f'  {k:<10}: {v:6.0f}s ({v/5:.1f}s/q)')
print(f'  {"TOTAL":<10}: {total:6.0f}s ({total/60:.1f} min)')
