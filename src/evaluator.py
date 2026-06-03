"""Automatic evaluation: FactScore, citation precision, answer accuracy,
answer recall, refusal accuracy + bootstrap testing.

NLI judgments via XiaoMi API — fully decoupled from Gemini-based SEVE generation.
Hallucination Rate = 1 - FactScore (complementary, not separately computed).
"""
import csv
import re
import numpy as np
from openai import OpenAI
from src.config import NLI_API_KEY, NLI_API_BASE_URL, NLI_MODEL_NAME
from src.utils import load_json, extract_claims_from_answer, ROOT_DIR

_nli_client = OpenAI(api_key=NLI_API_KEY, base_url=NLI_API_BASE_URL, timeout=60.0)


def nli_check(premise: str, hypothesis: str) -> str:
    """Check if hypothesis is entailed by premise via LLM-as-Judge."""
    premise_short = premise[:500]
    hypothesis_short = hypothesis[:300]

    prompt = f"""You are an NLI judge. Given a premise and a hypothesis, determine whether the premise supports the hypothesis.

Reply with exactly one word: ENTAILMENT, NEUTRAL, or CONTRADICTION.

Premise: {premise_short}
Hypothesis: {hypothesis_short}
Judgment:"""

    try:
        response = _nli_client.chat.completions.create(
            model=NLI_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10,
        )
        label = (response.choices[0].message.content or "").strip().upper()
    except Exception:
        return "UNKNOWN"

    if "ENTAIL" in label:
        return "ENTAILMENT"
    if "NEUTRAL" in label:
        return "NEUTRAL"
    if "CONTRADICT" in label:
        return "CONTRADICTION"
    return "UNKNOWN"


def _join_results(results: list[dict]) -> str:
    """Concatenate search result texts for NLI context."""
    parts = []
    for r in (results or []):
        text = r.get("full_text", "") or r.get("snippet", "")
        if text:
            parts.append(text)
    return " ".join(parts)


def strip_verification_notes(answer: str) -> str:
    """Remove verification notes appended by SEVE for clean evaluation."""
    idx = answer.find("\n\n--- 验证备注 ---")
    if idx != -1:
        return answer[:idx].strip()
    return answer


# --- Metrics ---

def compute_factscore(answer: str, search_results: list[dict]) -> float:
    """Fraction of atomic claims supported by search results. UNKNOWN skipped."""
    claims = extract_claims_from_answer(answer)
    if not claims:
        return 1.0
    context = _join_results(search_results)
    if not context.strip():
        return 0.0

    judged, supported = 0, 0
    for c in claims:
        v = nli_check(context, c)
        if v == "UNKNOWN":
            continue
        judged += 1
        if v == "ENTAILMENT":
            supported += 1
    if judged == 0:
        return 0.0
    return supported / judged


def compute_citation_precision(
    answer: str, search_results: list[dict]
) -> float | None:
    """Fraction of citations that actually support their associated claim."""
    citations = re.findall(r"\[(\d+)\]", answer)
    if not citations:
        return None

    results_by_rank = {str(r["rank"]): r for r in (search_results or [])}

    correct, total = 0, 0
    for cid in citations:
        result = results_by_rank.get(cid)
        if not result:
            continue
        total += 1
        snippet = result.get("full_text", "") or result.get("snippet", "")
        pattern = re.compile(rf"([^.]*\[{re.escape(cid)}\][^.]*\.)")
        for m in pattern.finditer(answer):
            v = nli_check(snippet, m.group(1))
            if v == "ENTAILMENT":
                correct += 1
            elif v == "UNKNOWN":
                total -= 1
            break

    if total <= 0:
        return None
    return correct / total


def compute_answer_accuracy(answer: str, reference_answer: str) -> float | None:
    """Fraction of ANSWER claims supported by the reference. UNKNOWN skipped."""
    if not reference_answer:
        return None
    ans_claims = extract_claims_from_answer(answer)
    if not ans_claims:
        return None

    judged, correct = 0, 0
    for ac in ans_claims:
        v = nli_check(reference_answer, ac)
        if v == "UNKNOWN":
            continue
        judged += 1
        if v == "ENTAILMENT":
            correct += 1
    if judged == 0:
        return None
    return correct / judged


def compute_answer_recall(answer: str, reference_answer: str) -> float | None:
    """Fraction of REFERENCE facts covered by the answer. UNKNOWN skipped."""
    if not reference_answer:
        return None
    ref_claims = extract_claims_from_answer(reference_answer)
    if not ref_claims:
        return None

    judged, covered = 0, 0
    for rc in ref_claims:
        v = nli_check(answer, rc)
        if v == "UNKNOWN":
            continue
        judged += 1
        if v == "ENTAILMENT":
            covered += 1
    if judged == 0:
        return None
    return covered / judged


# --- Bootstrap ---

def bootstrap_test(
    scores_a: list[float],
    scores_b: list[float],
    n_samples: int = 10000,
    alpha: float = 0.05,
) -> dict:
    """Paired bootstrap test. Returns {observed_diff, ci_lower, ci_upper, p_value}."""
    a = np.array(scores_a)
    b = np.array(scores_b)
    n = len(a)
    observed_diff = float(np.mean(a) - np.mean(b))

    rng = np.random.default_rng(42)
    diffs = []
    for _ in range(n_samples):
        idx = rng.integers(0, n, size=n)
        diffs.append(float(np.mean(a[idx]) - np.mean(b[idx])))

    diffs_arr = np.array(diffs)
    ci_lower = float(np.percentile(diffs_arr, 100 * alpha / 2))
    ci_upper = float(np.percentile(diffs_arr, 100 * (1 - alpha / 2)))
    centered = diffs_arr - np.mean(diffs_arr)
    p_value = float(np.mean(np.abs(centered) >= np.abs(observed_diff)))

    return {"observed_diff": observed_diff, "ci_lower": ci_lower,
            "ci_upper": ci_upper, "p_value": p_value}


# --- Output Loading ---

def load_all_outputs() -> dict[str, dict]:
    """Load outputs from all methods."""
    methods = {
        "vanilla_rag": "outputs/vanilla_rag/answers.json",
        "self_rag": "outputs/self_rag/answers.json",
        "seve": "outputs/seve/final_answers.json",
        "ablation_wo_rev_verify": "outputs/ablation_wo_rev_verify/answers.json",
    }
    outputs = {}
    for name, path in methods.items():
        try:
            outputs[name] = load_json(path)
        except FileNotFoundError:
            print(f"WARNING: {path} not found, skipping {name}")
            outputs[name] = {}
    return outputs


# --- Main Runner ---

def run_full_evaluation() -> None:
    """Run all metrics on all methods, output summary CSV + per-question details."""
    outputs = load_all_outputs()
    cache = load_json("data/search_cache.json")
    questions = load_json("data/freshqa_questions.json")
    qid_to_q = {q["question_id"]: q for q in questions}

    all_scores: dict[str, dict[str, list[float]]] = {}
    per_question_rows = []

    for method, answers in outputs.items():
        print(f"\nEvaluating {method} ({len(answers)} answers)...")
        scores: dict[str, list[float]] = {
            "factscore": [], "cit_precision": [],
            "ans_accuracy": [], "ans_recall": [], "refusal_acc": [],
        }

        for qid, a in answers.items():
            q = qid_to_q.get(qid, {})
            answer_text = strip_verification_notes(a.get("answer", ""))
            results = cache.get(qid, {}).get("results", [])
            ref = q.get("reference_answer", "")

            fs = compute_factscore(answer_text, results)
            cp = compute_citation_precision(answer_text, results)
            aa = compute_answer_accuracy(answer_text, ref) if ref else None
            ar = compute_answer_recall(answer_text, ref) if ref else None

            scores["factscore"].append(fs)
            if cp is not None:
                scores["cit_precision"].append(cp)
            if aa is not None:
                scores["ans_accuracy"].append(aa)
            if ar is not None:
                scores["ans_recall"].append(ar)

            is_unanswerable = not q.get("is_answerable", True)
            if is_unanswerable:
                refused = any(
                    kw in answer_text
                    for kw in ["信息不足", "无法", "不确定", "信息不足以确定"]
                )
                scores["refusal_acc"].append(1.0 if refused else 0.0)

            per_question_rows.append({
                "question_id": qid, "method": method,
                "factscore": fs, "cit_precision": cp,
                "ans_accuracy": aa, "ans_recall": ar,
                "search_available": a.get("search_available", True),
            })

        all_scores[method] = scores

    # Summary table
    rows = []
    for method, scores in all_scores.items():
        row = {"method": method}
        for metric in ["factscore", "cit_precision", "ans_accuracy",
                        "ans_recall", "refusal_acc"]:
            vals = scores[metric]
            row[metric] = f"{np.mean(vals):.4f}" if vals else "N/A"
        rows.append(row)

    # Bootstrap: SEVE vs each baseline
    seve_scores = all_scores.get("seve", {})
    for baseline in ["vanilla_rag", "self_rag"]:
        baseline_scores = all_scores.get(baseline, {})
        if not seve_scores or not baseline_scores:
            continue
        for metric in ["factscore", "ans_accuracy"]:
            a = seve_scores.get(metric, [])
            b = baseline_scores.get(metric, [])
            if len(a) == len(b) and len(a) > 0:
                bt = bootstrap_test(a, b)
                print(f"SEVE vs {baseline} [{metric}]: "
                      f"diff={bt['observed_diff']:.4f} p={bt['p_value']:.4f}")

    # Save summary
    save_path = ROOT_DIR / "outputs/evaluation/results_summary.csv"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["method", "factscore", "cit_precision",
                   "ans_accuracy", "ans_recall", "refusal_acc"]
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults saved to {save_path}")

    # Save per-question details
    detail_path = ROOT_DIR / "outputs/evaluation/per_question_details.csv"
    detail_fields = ["question_id", "method", "factscore", "cit_precision",
                      "ans_accuracy", "ans_recall", "search_available"]
    with open(detail_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=detail_fields)
        writer.writeheader()
        writer.writerows(per_question_rows)
    print(f"Per-question details saved to {detail_path}")

    # Print table
    header = (f"{'Method':<30} {'FactScore':>10} {'CitPrec':>10} "
              f"{'AnsAcc':>10} {'AnsRec':>10} {'RefAcc':>10}")
    print("\n" + header)
    print("-" * len(header))
    for row in rows:
        print(f"{row['method']:<30} {row['factscore']:>10} "
              f"{row['cit_precision']:>10} {row['ans_accuracy']:>10} "
              f"{row['ans_recall']:>10} {row['refusal_acc']:>10}")


if __name__ == "__main__":
    run_full_evaluation()
