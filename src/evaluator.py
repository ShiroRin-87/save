"""Automatic evaluation: FactScore, hallucination rate, citation precision,
answer accuracy, answer recall, refusal accuracy + bootstrap testing.

Uses google/t5_xxl_true_nli_mixture (local HuggingFace model) for NLI
judgments, fully decoupled from Gemini-based verification in SEVE.
"""
import csv
import re
import numpy as np
from src.utils import load_json, extract_claims_from_answer, ROOT_DIR


_nli_model = None


def _get_nli():
    """Lazy-load the NLI model once per session."""
    global _nli_model
    if _nli_model is None:
        try:
            from transformers import pipeline
            _nli_model = pipeline(
                "text-classification",
                model="google/t5_xxl_true_nli_mixture",
            )
        except ImportError:
            print("WARNING: transformers not installed. NLI metrics will be NaN.")
            _nli_model = False
    return _nli_model


def nli_check(premise: str, hypothesis: str) -> str:
    """Check if hypothesis is entailed by premise.

    Returns 'ENTAILMENT', 'NEUTRAL', or 'CONTRADICTION'.
    """
    nli = _get_nli()
    if nli is False or nli is None:
        return "UNKNOWN"
    result = nli(f"premise: {premise} hypothesis: {hypothesis}")
    label = result[0]["label"].upper()
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


# --- Metrics ---

def compute_factscore(answer: str, search_results: list[dict]) -> float:
    """Fraction of atomic claims supported by search results."""
    claims = extract_claims_from_answer(answer)
    if not claims:
        return 1.0
    context = _join_results(search_results)
    if not context.strip():
        return 0.0

    supported = sum(1 for c in claims if nli_check(context, c) == "ENTAILMENT")
    return supported / len(claims)


def compute_hallucination_rate(answer: str, search_results: list[dict]) -> float:
    """Fraction of sentences NOT entailed by search results."""
    sentences = extract_claims_from_answer(answer)
    if not sentences:
        return 0.0
    context = _join_results(search_results)
    if not context.strip():
        return 1.0

    unsupported = sum(
        1 for s in sentences if nli_check(context, s) != "ENTAILMENT"
    )
    return unsupported / len(sentences)


def compute_citation_precision(
    answer: str, search_results: list[dict]
) -> float | None:
    """Fraction of citations that actually support their associated claim.

    Returns None if no citations present (not applicable).
    """
    citations = re.findall(r"\[(\d+)\]", answer)
    if not citations:
        return None

    results_by_rank = {str(r["rank"]): r for r in (search_results or [])}

    correct = 0
    total = 0
    for cid in citations:
        result = results_by_rank.get(cid)
        if not result:
            continue
        total += 1
        snippet = result.get("full_text", "") or result.get("snippet", "")
        # Find the sentence containing this citation
        pattern = re.compile(rf"([^.]*\[{re.escape(cid)}\][^.]*\.)")
        for m in pattern.finditer(answer):
            if nli_check(snippet, m.group(1)) == "ENTAILMENT":
                correct += 1
            break

    if total == 0:
        return None
    return correct / total


def compute_answer_accuracy(answer: str, reference_answer: str) -> float | None:
    """Fraction of ANSWER claims supported by the reference answer.

    Measures precision: are the generated answer's claims factually correct?
    """
    if not reference_answer:
        return None
    ans_claims = extract_claims_from_answer(answer)
    if not ans_claims:
        return None

    correct = sum(1 for ac in ans_claims if nli_check(reference_answer, ac) == "ENTAILMENT")
    return correct / len(ans_claims)


def compute_answer_recall(answer: str, reference_answer: str) -> float | None:
    """Fraction of REFERENCE facts covered by the generated answer.

    Measures coverage: does the answer cover all the reference info points?
    """
    if not reference_answer:
        return None
    ref_claims = extract_claims_from_answer(reference_answer)
    if not ref_claims:
        return None

    covered = sum(1 for rc in ref_claims if nli_check(answer, rc) == "ENTAILMENT")
    return covered / len(ref_claims)


# --- Bootstrap ---

def bootstrap_test(
    scores_a: list[float],
    scores_b: list[float],
    n_samples: int = 10000,
    alpha: float = 0.05,
) -> dict:
    """Paired bootstrap test between two sets of per-question scores.

    Returns {observed_diff, ci_lower, ci_upper, p_value}.
    """
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
    # Center at 0 for null-hypothesis testing
    centered = diffs_arr - np.mean(diffs_arr)
    p_value = float(np.mean(np.abs(centered) >= np.abs(observed_diff)))

    return {
        "observed_diff": observed_diff,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": p_value,
    }


# --- Output Loading ---

def load_all_outputs() -> dict[str, dict]:
    """Load outputs from all methods. Returns {method_name: {qid: answer_data}}."""
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

    # Collect per-question scores for every method
    all_scores: dict[str, dict[str, list[float]]] = {}
    per_question_rows = []

    for method, answers in outputs.items():
        print(f"\nEvaluating {method} ({len(answers)} answers)...")
        scores: dict[str, list[float]] = {
            "factscore": [],
            "hall_rate": [],
            "cit_precision": [],
            "ans_accuracy": [],
            "ans_recall": [],
            "refusal_acc": [],
        }

        for qid, a in answers.items():
            q = qid_to_q.get(qid, {})
            answer_text = a.get("answer", "")
            results = cache.get(qid, {}).get("results", [])
            ref = q.get("reference_answer", "")

            fs = compute_factscore(answer_text, results)
            hr = compute_hallucination_rate(answer_text, results)
            cp = compute_citation_precision(answer_text, results)
            aa = compute_answer_accuracy(answer_text, ref) if ref else None
            ar = compute_answer_recall(answer_text, ref) if ref else None

            scores["factscore"].append(fs)
            scores["hall_rate"].append(hr)
            if cp is not None:
                scores["cit_precision"].append(cp)

            if aa is not None:
                scores["ans_accuracy"].append(aa)
            if ar is not None:
                scores["ans_recall"].append(ar)

            # Refusal accuracy
            is_unanswerable = not q.get("is_answerable", True)
            if is_unanswerable:
                refused = any(
                    kw in answer_text
                    for kw in ["信息不足", "无法", "不确定", "信息不足以确定"]
                )
                scores["refusal_acc"].append(1.0 if refused else 0.0)

            per_question_rows.append({
                "question_id": qid,
                "method": method,
                "factscore": fs,
                "hall_rate": hr,
                "cit_precision": cp,
                "ans_accuracy": aa,
                "ans_recall": ar,
                "search_available": a.get("search_available", True),
            })

        all_scores[method] = scores

    # Summary table
    rows = []
    for method, scores in all_scores.items():
        row = {"method": method}
        for metric in ["factscore", "hall_rate", "cit_precision",
                        "ans_accuracy", "ans_recall", "refusal_acc"]:
            vals = scores[metric]
            row[metric] = f"{np.mean(vals):.4f}" if vals else "N/A"
        rows.append(row)

    # Bootstrap: SEVE vs each baseline
    seve_scores = all_scores.get("seve", {})
    for baseline in ["vanilla_rag", "self_rag"]:
        baseline_scores = all_scores.get(baseline, {})
        if not seve_scores or not baseline_scores:
            continue
        for metric in ["factscore", "hall_rate", "ans_accuracy"]:
            a = seve_scores.get(metric, [])
            b = baseline_scores.get(metric, [])
            if len(a) == len(b) and len(a) > 0:
                bt = bootstrap_test(a, b)
                print(
                    f"SEVE vs {baseline} [{metric}]: "
                    f"diff={bt['observed_diff']:.4f}, "
                    f"95% CI=[{bt['ci_lower']:.4f}, {bt['ci_upper']:.4f}], "
                    f"p={bt['p_value']:.4f}"
                )

    # Save summary
    save_path = ROOT_DIR / "outputs/evaluation/results_summary.csv"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method", "factscore", "hall_rate", "cit_precision",
        "ans_accuracy", "ans_recall", "refusal_acc",
    ]
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults saved to {save_path}")

    # Save per-question details
    detail_path = ROOT_DIR / "outputs/evaluation/per_question_details.csv"
    detail_fields = [
        "question_id", "method", "factscore", "hall_rate",
        "cit_precision", "ans_accuracy", "ans_recall", "search_available",
    ]
    with open(detail_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=detail_fields)
        writer.writeheader()
        writer.writerows(per_question_rows)
    print(f"Per-question details saved to {detail_path}")

    # Print table
    header = (
        f"{'Method':<30} {'FactScore':>10} {'HallRate':>10} {'CitPrec':>10} "
        f"{'AnsAcc':>10} {'AnsRec':>10} {'RefAcc':>10}"
    )
    print("\n" + header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['method']:<30} {row['factscore']:>10} {row['hall_rate']:>10} "
            f"{row['cit_precision']:>10} {row['ans_accuracy']:>10} "
            f"{row['ans_recall']:>10} {row['refusal_acc']:>10}"
        )


if __name__ == "__main__":
    run_full_evaluation()
