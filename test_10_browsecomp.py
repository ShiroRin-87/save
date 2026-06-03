"""Test 10 BrowseComp-ZH questions with all 3 methods — with progress visualization."""
import json
import sys
import time
import requests
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.bing_client import search
from src.model_client import generate
from src.seve import extract_claims, generate_answer, verify_claims, apply_verification
from src.utils import load_json, save_json, format_search_results_numbered

SELECTED_IDS = [8, 11, 42, 19, 17, 15, 59, 12, 84, 6]
JINA_ENDPOINT = "https://r.jina.ai"
OUT_DIR = ROOT / "outputs" / "browsecomp_10"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TOTAL = len(SELECTED_IDS)


def now():
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


def jina_read(url: str) -> str | None:
    try:
        resp = requests.get(f"{JINA_ENDPOINT}/{url}",
                          headers={"Accept": "text/markdown"}, timeout=30)
        if resp.status_code == 200 and resp.text.strip():
            return resp.text.strip()
    except Exception:
        pass
    return None


def enrich_with_jina(results: list[dict], q_prefix: str) -> list[dict]:
    enriched = []
    n = len(results)
    for i, r in enumerate(results):
        entry = dict(r)
        url = r["url"]
        print(f"  {q_prefix} Jina [{i+1}/{n}] {url[:60]}...", end=" ", flush=True)
        t0 = time.time()
        full = jina_read(url)
        if full:
            entry["full_text"] = full
            entry["full_text_source"] = "jina"
            entry["full_text_len"] = len(full)
            print(f"{len(full)} chars ({time.time()-t0:.1f}s)", flush=True)
        else:
            entry["full_text"] = r.get("snippet", "")
            entry["full_text_source"] = "snippet_fallback"
            entry["full_text_len"] = len(entry["full_text"])
            print(f"fallback ({time.time()-t0:.1f}s)", flush=True)
        enriched.append(entry)
        time.sleep(0.3)
    return enriched


def run_vanilla_rag(question: str, results: list[dict]) -> str:
    context = format_search_results_numbered(results)
    prompt = f"""你是一个问答助手。请基于以下搜索结果为用户问题提供准确、简洁的回答。
每个事实声明后标注引用编号，如 [1][2]，对应搜索结果的序号。

搜索结果：
{context}

用户问题：{question}

请回答："""
    return generate(prompt)


def run_self_rag(question: str, results: list[dict]) -> dict:
    context = format_search_results_numbered(results)
    r1 = generate(f"""基于以下搜索结果回答问题。在回答中，为每个事实片段标注 [Relevant] 或 [Irrelevant]。

搜索结果：
{context}

问题：{question}""")
    r2 = generate(f"""以下是一个回答及其标注。请检查每个声明是否被原文搜索结果支撑。
标注每个声明为 [Supported]、[Partially]、[Unsupported]。

回答：
{r1}

搜索结果：
{context}""")
    r3 = generate(f"""基于反思结果重新生成最终答案。只保留 [Supported] 和 [Partially] 的声明。

原始回答：
{r1}

反思结果：
{r2}

用户问题：{question}""")
    return {"answer": r3, "round1": r1, "round2": r2}


def run_seve(question: str, results: list[dict], q_prefix: str) -> dict:
    print(f"    {q_prefix} SEVE extract...", end=" ", flush=True); t0 = time.time()
    claims = extract_claims(results, question)
    print(f"{len(claims)} claims ({time.time()-t0:.1f}s)", flush=True)

    print(f"    {q_prefix} SEVE generate...", end=" ", flush=True); t0 = time.time()
    answer, citation_map = generate_answer(question, claims)
    print(f"{len(answer)} chars ({time.time()-t0:.1f}s)", flush=True)

    print(f"    {q_prefix} SEVE verify...", end=" ", flush=True); t0 = time.time()
    verifications = verify_claims(answer, claims, results)
    verdicts = ",".join(f"{v['verdict']}" for v in verifications) if verifications else "none"
    print(f"{verdicts} ({time.time()-t0:.1f}s)", flush=True)

    print(f"    {q_prefix} SEVE apply...", end=" ", flush=True); t0 = time.time()
    final = apply_verification(answer, verifications)
    print(f"({time.time()-t0:.1f}s)", flush=True)

    return {
        "claims": claims,
        "num_claims": len(claims),
        "answer": answer,
        "citation_map": citation_map,
        "verifications": verifications,
        "final": final,
    }


def main():
    all_qs = load_json("data/browsecomp-zh-decrypted.json")
    try:
        cache = load_json("data/search_cache.json")
    except Exception:
        cache = {}

    results_all = {}
    t_start = time.time()

    for qi, idx in enumerate(SELECTED_IDS):
        q = all_qs[idx]
        qid = f"BC{idx:03d}"
        question = q["Question"]
        answer_ref = q["Answer"]
        topic = q["Topic"]
        cache_key = qid
        pre = f"[{qi+1}/{TOTAL}]"

        print(f"\n{'━' * 70}")
        print(f"{pre} [{topic}] {qid}")
        print(f"  Q: {question[:120]}...")
        print(f"  A_ref: {answer_ref}")
        print(f"{'━' * 70}")

        # ── Step 1: Search ──
        if cache_key in cache and cache[cache_key].get("results"):
            results = cache[cache_key]["results"]
            print(f"  {pre} Search: cached {len(results)} results", flush=True)
        else:
            print(f"  {pre} Search...", end=" ", flush=True); t0 = time.time()
            try:
                results = search(question)
                cache[cache_key] = {
                    "query": question,
                    "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "results": results,
                }
                save_json(cache, "data/search_cache.json")
                print(f"{len(results)} results ({time.time()-t0:.1f}s)", flush=True)
            except Exception as e:
                print(f"ERROR: {e}", flush=True)
                results = []

        if not results:
            results_all[qid] = {"question": question, "topic": topic,
                                "answer_ref": answer_ref,
                                "vanilla": "[无搜索结果]",
                                "self_rag": {"answer": "[无搜索结果]"},
                                "seve": {"answer": "[无搜索结果]"}}
            continue

        # ── Step 2: Jina ──
        print(f"  {pre} Jina ({len(results)} URLs):", flush=True)
        t0 = time.time()
        enriched = enrich_with_jina(results, f"{pre}")
        jina_chars = sum(r.get("full_text_len", 0) for r in enriched)
        print(f"  {pre} Jina done: {jina_chars:,} chars total ({time.time()-t0:.1f}s)", flush=True)

        # ── Step 3: Vanilla RAG ──
        print(f"  {pre} Vanilla RAG...", end=" ", flush=True); t0 = time.time()
        try:
            v_answer = run_vanilla_rag(question, enriched)
            print(f"{len(v_answer)} chars ({time.time()-t0:.1f}s)", flush=True)
        except Exception as e:
            v_answer = f"[ERROR: {e}]"
            print(f"ERROR: {e}", flush=True)

        # ── Step 4: Self-RAG ──
        print(f"  {pre} Self-RAG r1...", end=" ", flush=True); t0 = time.time()
        try:
            context = format_search_results_numbered(enriched)
            sr1 = generate(f"""基于以下搜索结果回答问题。在回答中，为每个事实片段标注 [Relevant] 或 [Irrelevant]。

搜索结果：
{context}

问题：{question}""")
            print(f"({time.time()-t0:.1f}s)", end=" ", flush=True); t1 = time.time()

            print(f"r2...", end=" ", flush=True)
            sr2 = generate(f"""以下是一个回答及其标注。请检查每个声明是否被原文搜索结果支撑。
标注每个声明为 [Supported]、[Partially]、[Unsupported]。

回答：
{sr1}

搜索结果：
{context}""")
            print(f"({time.time()-t1:.1f}s)", end=" ", flush=True); t2 = time.time()

            print(f"r3...", end=" ", flush=True)
            sr3 = generate(f"""基于反思结果重新生成最终答案。只保留 [Supported] 和 [Partially] 的声明。

原始回答：
{sr1}

反思结果：
{sr2}

用户问题：{question}""")
            sr = {"answer": sr3, "round1": sr1, "round2": sr2}
            print(f"({time.time()-t2:.1f}s) total={time.time()-t0:.1f}s", flush=True)
        except Exception as e:
            sr = {"answer": f"[ERROR: {e}]", "round1": "", "round2": ""}
            print(f"ERROR: {e}", flush=True)

        # ── Step 5: SEVE ──
        print(f"  {pre} SEVE:", flush=True)
        t0 = time.time()
        try:
            seve = run_seve(question, enriched, pre)
            print(f"  {pre} SEVE total: {seve['num_claims']} claims, {len(seve['verifications'])} verified ({time.time()-t0:.1f}s)", flush=True)
        except Exception as e:
            seve = {"claims": [], "num_claims": 0, "answer": f"[ERROR: {e}]",
                    "citation_map": {}, "verifications": [], "final": f"[ERROR: {e}]"}
            print(f"  {pre} SEVE ERROR: {e}", flush=True)

        results_all[qid] = {
            "question": question,
            "topic": topic,
            "answer_ref": answer_ref,
            "vanilla": v_answer,
            "self_rag": sr,
            "seve": seve,
            "num_results": len(enriched),
            "jina_chars": jina_chars,
        }

        save_json(results_all, str(OUT_DIR / "all_results.json"))
        elapsed = time.time() - t_start
        print(f"  {pre} Done this question. Overall elapsed: {elapsed/60:.1f}min", flush=True)

    # ── Final Summary ──
    total_t = time.time() - t_start
    print(f"\n{'█' * 70}")
    print(f"ALL {TOTAL} DONE in {total_t/60:.1f} min")
    print(f"{'█' * 70}")
    for qid, r in results_all.items():
        print(f"\n── {qid} [{r['topic']}] ──")
        print(f"Q: {r['question'][:100]}")
        print(f"Ref: {r['answer_ref']}")
        print(f"── Vanilla ──\n{r['vanilla'][:200]}")
        print(f"── Self-RAG ──\n{r['self_rag']['answer'][:200]}")
        print(f"── SEVE ──\n{r['seve']['final'][:200]}")


if __name__ == "__main__":
    main()
