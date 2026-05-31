"""Test 50 questions with Jina Reader for full-text extraction, run all 3 methods."""
import json
import sys
import time
import requests
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.bing_client import search
from src.model_client import generate
from src.seve import extract_claims, generate_answer, verify_claims, apply_verification
from src.utils import load_search_cache, format_search_results_numbered, save_json, load_json

JINA_ENDPOINT = "https://r.jina.ai"
JINA_TOKEN = ""
N_QUESTIONS = 50

# Output paths
OUT_SEVE_TABLES = "outputs/seve/structured_tables.json"
OUT_SEVE_GEN = "outputs/seve/generated_answers.json"
OUT_SEVE_VERIFY = "outputs/seve/verification_results.json"
OUT_SEVE_FINAL = "outputs/seve/final_answers.json"
OUT_VANILLA = "outputs/vanilla_rag/answers.json"
OUT_SELF_RAG = "outputs/self_rag/answers.json"


def jina_read(url: str) -> str | None:
    headers = {"Accept": "text/markdown"}
    if JINA_TOKEN:
        headers["Authorization"] = f"Bearer {JINA_TOKEN}"
    try:
        resp = requests.get(f"{JINA_ENDPOINT}/{url}", headers=headers, timeout=30)
        if resp.status_code == 200 and resp.text.strip():
            return resp.text.strip()
        return None
    except Exception as e:
        print(f"      Jina error: {e}")
        return None


def enrich_results_with_jina(results: list[dict]) -> list[dict]:
    """Fetch full text via Jina for each result, fall back to snippet."""
    enriched = []
    for r in results:
        entry = dict(r)
        url = r["url"]
        print(f"      Jina fetching [{r['rank']}] {url[:80]}...")
        full = jina_read(url)
        if full:
            entry["full_text"] = full
            entry["full_text_source"] = "jina"
            entry["full_text_len"] = len(full)
            print(f"        OK: {len(full)} chars")
        else:
            entry["full_text"] = r.get("snippet", "")
            entry["full_text_source"] = "snippet_fallback"
            entry["full_text_len"] = len(entry["full_text"])
            print(f"        FALLBACK: using snippet ({entry['full_text_len']} chars)")
        enriched.append(entry)
        time.sleep(0.3)
    return enriched


def run_vanilla_rag(qid: str, question: str, results: list[dict]) -> str:
    context = format_search_results_numbered(results)
    prompt = f"""你是一个问答助手。请基于以下搜索结果为用户问题提供准确、简洁的回答。
每个事实声明后标注引用编号，如 [1][2]，对应搜索结果的序号。

搜索结果：
{context}

用户问题：{question}

请回答："""
    return generate(prompt)


def run_self_rag(qid: str, question: str, results: list[dict]) -> dict:
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


def run_seve(qid: str, question: str, results: list[dict]) -> dict:
    claims = extract_claims(results)
    answer, citation_map = generate_answer(question, claims)
    verifications = verify_claims(answer, claims, results)
    final = apply_verification(answer, verifications)
    return {
        "claims": claims,
        "num_claims": len(claims),
        "answer": answer,
        "citation_map": citation_map,
        "verifications": verifications,
        "final": final,
    }


def load_existing(path: str) -> dict:
    try:
        return json.loads((ROOT / path).read_text("utf-8"))
    except Exception:
        return {}


def main():
    questions = load_json("data/freshqa_questions.json")[:N_QUESTIONS]
    cache = load_search_cache()

    # Load existing outputs for incremental saving
    out_v = load_existing(OUT_VANILLA)
    out_sr = load_existing(OUT_SELF_RAG)
    out_st = load_existing(OUT_SEVE_TABLES)
    out_sg = load_existing(OUT_SEVE_GEN)
    out_sv = load_existing(OUT_SEVE_VERIFY)
    out_sf = load_existing(OUT_SEVE_FINAL)

    for idx, q in enumerate(questions):
        qid = q["question_id"]
        question = q["question"]
        print(f"\n{'=' * 60}")
        print(f"[{idx + 1}/{N_QUESTIONS}] {qid}: {question[:80]}")
        print(f"{'=' * 60}")

        # Step 1: Get search results (from cache or API)
        if qid in cache and cache[qid].get("results"):
            results = cache[qid]["results"]
            print(f"  Using cached {len(results)} results")
        else:
            print(f"  Searching via Bing API...")
            try:
                results = search(question)
                cache[qid] = {
                    "query": question,
                    "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "results": results,
                }
                save_json(cache, "data/search_cache.json")
                print(f"  Got {len(results)} results, cache updated")
            except Exception as e:
                print(f"  Search ERROR: {e}")
                results = []
                cache[qid] = {
                    "query": question,
                    "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "results": [],
                }
                save_json(cache, "data/search_cache.json")

        search_available = bool(results and len(results) > 0)

        if not search_available:
            print(f"  No search results, using model internal knowledge only")
            out_v[qid] = {"question": question, "answer": "[无搜索结果]", "method": "vanilla_rag", "search_available": False}
            out_sr[qid] = {"question": question, "answer": "[无搜索结果]", "method": "self_rag", "search_available": False}
            out_st[qid] = {"question": question, "claims": [], "num_claims": 0}
            out_sg[qid] = {"question": question, "answer": "[无搜索结果]", "citation_map": {}}
            out_sv[qid] = {"question": question, "verifications": []}
            out_sf[qid] = {"question": question, "answer": "[无搜索结果]", "method": "seve", "search_available": False}
            continue

        # Step 2: Enrich with Jina full text
        print(f"  Enriching {len(results)} results with Jina...")
        enriched = enrich_results_with_jina(results)

        # Step 3: Run all 3 methods
        print(f"  [Vanilla RAG] generating...")
        try:
            v_answer = run_vanilla_rag(qid, question, enriched)
            print(f"    OK: {len(v_answer)} chars")
        except Exception as e:
            print(f"    ERROR: {e}")
            v_answer = f"[ERROR: {e}]"
        out_v[qid] = {"question": question, "answer": v_answer, "method": "vanilla_rag", "search_available": True}

        print(f"  [Self-RAG] 3 rounds...")
        try:
            sr = run_self_rag(qid, question, enriched)
            print(f"    OK: {len(sr['answer'])} chars")
        except Exception as e:
            print(f"    ERROR: {e}")
            sr = {"answer": f"[ERROR: {e}]", "round1": "", "round2": ""}
        out_sr[qid] = {"question": question, "answer": sr["answer"], "method": "self_rag", "round1": sr["round1"], "round2": sr["round2"], "search_available": True}

        print(f"  [SEVE] extraction + generation + verification...")
        try:
            seve = run_seve(qid, question, enriched)
            print(f"    Claims: {seve['num_claims']}, Verdicts: {len(seve['verifications'])}")
        except Exception as e:
            print(f"    ERROR: {e}")
            seve = {"claims": [], "num_claims": 0, "answer": f"[ERROR: {e}]", "citation_map": {}, "verifications": [], "final": f"[ERROR: {e}]"}
        out_st[qid] = {"question": question, "claims": seve["claims"], "num_claims": seve["num_claims"]}
        out_sg[qid] = {"question": question, "answer": seve["answer"], "citation_map": seve["citation_map"]}
        out_sv[qid] = {"question": question, "verifications": seve["verifications"]}
        out_sf[qid] = {"question": question, "answer": seve["final"], "method": "seve", "search_available": True}

        # Save incrementally every 5 questions
        if (idx + 1) % 5 == 0:
            print(f"  [Saving checkpoints...]")
            save_json(out_v, OUT_VANILLA)
            save_json(out_sr, OUT_SELF_RAG)
            save_json(out_st, OUT_SEVE_TABLES)
            save_json(out_sg, OUT_SEVE_GEN)
            save_json(out_sv, OUT_SEVE_VERIFY)
            save_json(out_sf, OUT_SEVE_FINAL)

    # Final save
    print(f"\n{'=' * 60}")
    print("Saving final outputs...")
    save_json(out_v, OUT_VANILLA)
    save_json(out_sr, OUT_SELF_RAG)
    save_json(out_st, OUT_SEVE_TABLES)
    save_json(out_sg, OUT_SEVE_GEN)
    save_json(out_sv, OUT_SEVE_VERIFY)
    save_json(out_sf, OUT_SEVE_FINAL)

    completed = len([k for k in out_v if k.startswith("FQ")])
    print(f"Done. {completed}/{N_QUESTIONS} questions processed.")


if __name__ == "__main__":
    main()
