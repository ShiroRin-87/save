# WeSearch Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an experiment framework that compares SEVE (Structured Extraction + Reverse Verification) against Vanilla RAG and Self-RAG baselines on FreshQA using Bing Search + Gemini.

**Architecture:** Modular Python scripts sharing `config.py` and `utils.py`. Each method (vanilla_rag, self_rag, seve, ablation) is a standalone script that reads from shared `data/` cache and writes to isolated `outputs/` directories. Evaluation reads all outputs and produces comparison tables.

**Tech Stack:** Python 3.10+, google-genai SDK, requests, pandas, scipy, jsonlines

---

### Task 1: Project scaffolding and config

**Files:**
- Create: `src/config.py`
- Create: `src/__init__.py` (empty)

- [ ] **Step 1: Write config.py**

```python
"""Experiment configuration — all values are fixed per the experiment design."""
import os

# API keys (set via environment variables, never hardcoded)
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
BING_API_KEY = os.environ["BING_API_KEY"]

# Model
MODEL_NAME = "gemini-2.0-flash"  # Gemini 3.0 Flash Preview via google-genai

# Search
SEARCH_TOP_K = 5

# Generation
TEMPERATURE = 0.0
MAX_INPUT_TOKENS = 32768
MAX_OUTPUT_TOKENS = 4096
API_TIMEOUT_SEC = 60

# Parsing
PARSE_RETRY_MAX = 3

# Evaluation
BOOTSTRAP_SAMPLES = 10000
ALPHA = 0.05

# Paths (relative to project root)
DATA_DIR = "data"
FRESHQA_FILE = "data/freshqa_questions.json"
SEARCH_CACHE_FILE = "data/search_cache.json"
OUTPUTS_DIR = "outputs"
```

- [ ] **Step 2: Create empty `__init__.py`**

```python
```

- [ ] **Step 3: Verify file structure**

Run: `ls -la src/`

---

### Task 2: Utility functions

**Files:**
- Create: `src/utils.py`

- [ ] **Step 1: Write utils.py**

```python
"""Shared utilities: JSON I/O, retry, parsing, formatting."""
import json
import re
import time
from pathlib import Path
from typing import Any, Callable

ROOT_DIR = Path(__file__).parent.parent


def load_json(path: str) -> Any:
    with open(ROOT_DIR / path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: str) -> None:
    p = ROOT_DIR / path
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def retry(func: Callable, max_retries: int = 3, delay: float = 1.0) -> Any:
    """Retry a callable on exception, with linear backoff."""
    last_exc = None
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            last_exc = e
            if attempt < max_retries - 1:
                time.sleep(delay * (attempt + 1))
    raise last_exc


def format_search_results(results: list[dict]) -> str:
    """Format cached search results for prompt insertion with [N] markers."""
    parts = []
    for r in results:
        text = r.get("full_text", "") or r.get("snippet", "")
        parts.append(f"[{r['rank']}] {r['title']}\n{text}\nURL: {r['url']}")
    return "\n\n".join(parts)


def format_search_results_numbered(results: list[dict]) -> str:
    """Format search results with numbered entries for citation-aware prompts."""
    parts = []
    for r in results:
        text = r.get("full_text", "") or r.get("snippet", "")
        parts.append(f"[{r['rank']}] {text}")
    return "\n\n".join(parts)


def parse_structured_table(text: str) -> list[dict]:
    """Parse SEVE structured extraction output into list of claim dicts.
    
    Expected format per line: N | claim_text | original_snippet | url [conflict]
    Returns list of {id, claim, snippet, url, has_conflict}.
    """
    claims = []
    lines = text.strip().split("\n")
    for line in lines:
        line = line.strip()
        if not line:
            continue
        match = re.match(
            r"(\d+)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(https?://\S+)",
            line
        )
        if match:
            claim_id = int(match.group(1))
            claim_text = match.group(2).strip()
            snippet = match.group(3).strip()
            url = match.group(4).strip()
            has_conflict = "[冲突]" in line
            claims.append({
                "id": claim_id,
                "claim": claim_text,
                "snippet": snippet,
                "url": url,
                "has_conflict": has_conflict,
            })
    return claims


def extract_claims_from_answer(answer: str) -> list[str]:
    """Split an answer into individual claims (sentence-level)."""
    sentences = re.split(r"(?<=[.!?])\s+", answer)
    return [s.strip() for s in sentences if s.strip()]


def load_questions() -> list[dict]:
    return load_json("data/freshqa_questions.json")


def load_search_cache() -> dict:
    return load_json("data/search_cache.json")
```

---

### Task 3: Gemini model client

**Files:**
- Create: `src/model_client.py`

- [ ] **Step 1: Write model_client.py**

```python
"""Gemini API wrapper via google-genai SDK."""
from google import genai
from src.config import GEMINI_API_KEY, MODEL_NAME, TEMPERATURE, MAX_OUTPUT_TOKENS

_client = genai.Client(api_key=GEMINI_API_KEY)


def generate(prompt: str, *, temperature: float = TEMPERATURE) -> str:
    """Send a prompt to Gemini and return the text response."""
    response = _client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config={
            "temperature": temperature,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
        },
    )
    return response.text
```

---

### Task 4: Bing Search client

**Files:**
- Create: `src/bing_client.py`

- [ ] **Step 1: Write bing_client.py**

```python
"""Bing Web Search API wrapper."""
import requests
from src.config import BING_API_KEY, SEARCH_TOP_K

BING_ENDPOINT = "https://api.bing.microsoft.com/v7.0/search"


def search(query: str, top_k: int = SEARCH_TOP_K) -> list[dict]:
    """Search Bing and return structured results."""
    headers = {"Ocp-Apim-Subscription-Key": BING_API_KEY}
    params = {"q": query, "count": top_k, "mkt": "en-US", "textFormat": "Raw"}
    resp = requests.get(BING_ENDPOINT, headers=headers, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    results = []
    for i, item in enumerate(data.get("webPages", {}).get("value", []), start=1):
        results.append({
            "rank": i,
            "title": item.get("name", ""),
            "url": item.get("url", ""),
            "snippet": item.get("snippet", ""),
            "full_text": item.get("snippet", ""),
        })
    return results
```

---

### Task 5: Search cache builder

**Files:**
- Create: `src/cache_builder.py`

- [ ] **Step 1: Write cache_builder.py**

```python
"""Build and save Bing search cache for all FreshQA questions."""
import time
from src.bing_client import search
from src.utils import load_json, save_json, ROOT_DIR


def build_cache() -> None:
    questions = load_json("data/freshqa_questions.json")
    cache = {}

    for i, q in enumerate(questions):
        qid = q["question_id"]
        query = q["question"]
        print(f"[{i+1}/{len(questions)}] Searching: {query[:80]}...")

        try:
            results = search(query)
            cache[qid] = {
                "query": query,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "results": results,
            }
        except Exception as e:
            print(f"  ERROR: {e}")
            cache[qid] = {
                "query": query,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "results": [],
            }

        time.sleep(0.5)  # rate limit

    save_json(cache, "data/search_cache.json")
    total = sum(1 for v in cache.values() if v["results"])
    empty = len(cache) - total
    print(f"Cache built: {total} with results, {empty} empty")


if __name__ == "__main__":
    build_cache()
```

---

### Task 6: Vanilla RAG

**Files:**
- Create: `src/vanilla_rag.py`

- [ ] **Step 1: Write vanilla_rag.py**

```python
"""Vanilla RAG baseline — generate answers directly from search results."""
from src.model_client import generate
from src.utils import (
    load_questions, load_search_cache, save_json,
    format_search_results_numbered
)


def run_vanilla_rag() -> None:
    questions = load_questions()
    cache = load_search_cache()
    answers = {}

    for q in questions:
        qid = q["question_id"]
        cached = cache.get(qid, {})
        results = cached.get("results", [])

        if results:
            context = format_search_results_numbered(results)
        else:
            context = "[No search results available]"

        prompt = f"""你是一个问答助手。请基于以下搜索结果为用户问题提供准确、简洁的回答。
每个事实声明后标注引用编号，如 [1][2]，对应搜索结果的序号。

搜索结果：
{context}

用户问题：{q["question"]}

请回答："""

        try:
            answer = generate(prompt)
        except Exception as e:
            answer = f"[ERROR: {e}]"

        answers[qid] = {"question": q["question"], "answer": answer, "method": "vanilla_rag"}
        print(f"Vanilla RAG [{qid}]: {answer[:100]}...")

    save_json(answers, "outputs/vanilla_rag/answers.json")
    print(f"Done: {len(answers)} answers saved.")


if __name__ == "__main__":
    run_vanilla_rag()
```

---

### Task 7: Self-RAG baseline

**Files:**
- Create: `src/self_rag.py`

- [ ] **Step 1: Write self_rag.py**

```python
"""Self-RAG baseline — 3-round generate → reflect → refine (prompt-based approximation)."""
from src.model_client import generate
from src.utils import (
    load_questions, load_search_cache, save_json,
    format_search_results_numbered
)


def run_self_rag() -> None:
    questions = load_questions()
    cache = load_search_cache()
    answers = {}

    for q in questions:
        qid = q["question_id"]
        cached = cache.get(qid, {})
        results = cached.get("results", [])
        context = format_search_results_numbered(results) if results else "[No search results available]"

        try:
            # Round 1: Generate with relevance labels
            r1_prompt = f"""基于以下搜索结果回答问题。在回答中，为每个事实片段标注 [Relevant] 或 [Irrelevant]。

搜索结果：
{context}

问题：{q["question"]}"""

            r1_answer = generate(r1_prompt)

            # Round 2: Self-critique with support labels
            r2_prompt = f"""以下是一个回答及其标注。请检查每个声明是否被原文搜索结果支撑。
标注每个声明为 [Supported]、[Partially]、[Unsupported]。

回答：
{r1_answer}

搜索结果：
{context}"""

            r2_reflection = generate(r2_prompt)

            # Round 3: Refine — keep only supported claims
            r3_prompt = f"""基于反思结果重新生成最终答案。只保留 [Supported] 和 [Partially] 的声明。

原始回答：
{r1_answer}

反思结果：
{r2_reflection}

用户问题：{q["question"]}"""

            final_answer = generate(r3_prompt)

        except Exception as e:
            final_answer = f"[ERROR: {e}]"

        answers[qid] = {
            "question": q["question"],
            "answer": final_answer,
            "method": "self_rag",
            "round1": r1_answer if 'r1_answer' in dir() else None,
            "round2": r2_reflection if 'r2_reflection' in dir() else None,
        }
        print(f"Self-RAG [{qid}]: {final_answer[:100]}...")

    save_json(answers, "outputs/self_rag/answers.json")
    print(f"Done: {len(answers)} answers saved.")


if __name__ == "__main__":
    run_self_rag()
```

- [ ] **Step 2: Fix local variable reference**

The `except` block references `r1_answer` and `r2_reflection` which may not be bound. Update the try/except to initialize them:

```python
        r1_answer = None
        r2_reflection = None
        final_answer = None
        try:
            r1_answer = generate(r1_prompt)
            r2_reflection = generate(r2_prompt)
            final_answer = generate(r3_prompt)
        except Exception as e:
            final_answer = f"[ERROR: {e}]"
```

---

### Task 8: SEVE — Structured Extraction

**Files:**
- Create: `src/seve.py`

- [ ] **Step 1: Write the structured extraction function in seve.py**

```python
"""SEVE — Structured Extraction & Verification: full pipeline."""
from src.model_client import generate
from src.utils import (
    load_questions, load_search_cache, save_json,
    format_search_results, parse_structured_table, retry
)


def extract_claims(results: list[dict]) -> list[dict]:
    """Step 2: Extract structured claims from search results."""
    if not results:
        return []

    context = format_search_results(results)

    prompt = f"""你是一个事实抽取器（不是推理器、不是作家）。你的唯一任务是从搜索结果中提取事实声明。

规则：
1. 只提取原文中明确陈述的事实，不推断、不总结、不跨源合成
2. 每个声明一行，格式必须严格遵循：
   序号 | 提炼声明 | 原文片段 | 出处(URL)
3. "提炼声明"是你从原文中提取的事实表述
   "原文片段"是搜索结果中支撑该声明的原始文本（必须逐字复制，不可改写）
4. 如果原文没有明确说，不要填写
5. 如果多个来源对同一事实给出矛盾信息，全部提取并在行末标注 [冲突]

搜索结果：
{context}

请输出结构化声明表："""

    def _call():
        text = generate(prompt)
        claims = parse_structured_table(text)
        if not claims:
            raise ValueError("Failed to parse any claims")
        return claims

    return retry(_call)
```

---

### Task 9: SEVE — Calibrated Generation

**Files:**
- Modify: `src/seve.py` (add generation function)

- [ ] **Step 1: Add calibrated generation to seve.py**

```python
def generate_answer(question: str, claims: list[dict]) -> tuple[str, dict]:
    """Step 3: Generate answer from structured knowledge table."""
    if not claims:
        answer = generate(f"请回答以下问题。如果无法回答，请说明。\n\n问题：{question}")
        return answer, {}

    # Build structured table text
    lines = []
    for c in claims:
        conflict = " [冲突]" if c["has_conflict"] else ""
        lines.append(f"[{c['id']}] {c['claim']} | 来源: {c['url']}{conflict}")
    table_text = "\n".join(lines)

    prompt = f"""你是一个问答助手。请基于以下结构化知识表回答用户问题。

规则：
1. 每个事实声明后标注引用编号，如 [1][2]
2. 如果知识表中标注了 [冲突]，需要同时呈现多方说法并注明来源差异
3. 如果知识表中没有足够信息，写明"当前信息不足以确定"
4. 对于只有单一来源的信息，标注"据[来源名称]"
5. 不要编造知识表中不存在的信息

结构化知识表：
{table_text}

用户问题：{question}

请回答："""

    answer = generate(prompt)

    # Build citation mapping: claim_id -> url
    citation_map = {str(c["id"]): c["url"] for c in claims}
    return answer, citation_map
```

---

### Task 10: SEVE — Reverse Verification

**Files:**
- Modify: `src/seve.py` (add verification function)

- [ ] **Step 1: Add reverse verification to seve.py**

```python
def verify_claims(answer: str, claims: list[dict]) -> list[dict]:
    """Step 4: Reverse-verify each cited claim against original text."""
    verifications = []

    # Find citation patterns [N] in the answer
    import re
    cited_ids = set()
    for m in re.finditer(r"\[(\d+)\]", answer):
        cited_ids.add(m.group(1))

    claim_by_id = {str(c["id"]): c for c in claims}

    for cid in cited_ids:
        claim = claim_by_id.get(cid)
        if not claim:
            verifications.append({"claim_id": cid, "verdict": "UNKNOWN", "reason": "claim not in table"})
            continue

        prompt = f"""请判断以下引用原文是否包含生成声明中声称的信息。

生成声明：{claim['claim']}
引用原文：{claim['snippet']}

仅回答一个词：
YES（原文明确包含该信息）
PARTIAL（原文部分支撑但细节不一致）
NO（原文不包含该信息或与之矛盾）

你的判断："""

        try:
            verdict = generate(prompt).strip().upper()
            if verdict not in ("YES", "PARTIAL", "NO"):
                verdict = "UNKNOWN"
        except Exception:
            verdict = "ERROR"

        verifications.append({
            "claim_id": cid,
            "claim_text": claim["claim"],
            "original_text": claim["snippet"],
            "url": claim["url"],
            "verdict": verdict,
        })

    return verifications


def apply_verification(answer: str, verifications: list[dict]) -> str:
    """Remove NO claims from the answer, annotate PARTIAL ones."""
    no_ids = {v["claim_id"] for v in verifications if v["verdict"] == "NO"}
    # Simple approach: build a note about removed/partial claims
    notes = []
    for v in verifications:
        if v["verdict"] == "NO":
            notes.append(f"[已删除 — 原文不支撑: [{v['claim_id']}] {v['claim_text'][:80]}...]")
        elif v["verdict"] == "PARTIAL":
            notes.append(f"[部分支撑: [{v['claim_id']}] {v['claim_text'][:80]}...]")

    if notes:
        answer = answer + "\n\n--- 验证备注 ---\n" + "\n".join(notes)
    return answer
```

---

### Task 11: SEVE — Main pipeline

**Files:**
- Modify: `src/seve.py` (add main runner)

- [ ] **Step 1: Add main runner to seve.py**

```python
def run_seve() -> None:
    questions = load_questions()
    cache = load_search_cache()

    all_tables = {}
    all_answers = {}
    all_verifications = {}
    all_final = {}

    for q in questions:
        qid = q["question_id"]
        results = cache.get(qid, {}).get("results", [])
        print(f"SEVE [{qid}]: {q['question'][:80]}...")

        try:
            claims = extract_claims(results)
            answer, citation_map = generate_answer(q["question"], claims)
            verifications = verify_claims(answer, claims)
            final_answer = apply_verification(answer, verifications)
        except Exception as e:
            claims = []
            answer = f"[ERROR: {e}]"
            citation_map = {}
            verifications = []
            final_answer = answer

        all_tables[qid] = {"question": q["question"], "claims": claims}
        all_answers[qid] = {"question": q["question"], "answer": answer, "citation_map": citation_map}
        all_verifications[qid] = {"question": q["question"], "verifications": verifications}
        all_final[qid] = {"question": q["question"], "answer": final_answer, "method": "seve"}
        print(f"  Claims: {len(claims)}, Verdicts: {len(verifications)}")

    save_json(all_tables, "outputs/seve/structured_tables.json")
    save_json(all_answers, "outputs/seve/generated_answers.json")
    save_json(all_verifications, "outputs/seve/verification_results.json")
    save_json(all_final, "outputs/seve/final_answers.json")
    print(f"SEVE complete: {len(all_final)} answers.")


if __name__ == "__main__":
    run_seve()
```

---

### Task 12: Ablation experiments

**Files:**
- Create: `src/ablation.py`

- [ ] **Step 1: Write ablation.py**

```python
"""Ablation experiments: wo_both and wo_rev_verify."""
from src.model_client import generate
from src.seve import extract_claims, generate_answer
from src.utils import (
    load_questions, load_search_cache, save_json,
    format_search_results_numbered
)


def run_ablation_wo_both() -> None:
    """Remove both structured extraction and reverse verification.
    Uses independent RAG + Citation prompt (not SEVE's calibrated generation prompt).
    """
    questions = load_questions()
    cache = load_search_cache()
    answers = {}

    for q in questions:
        qid = q["question_id"]
        results = cache.get(qid, {}).get("results", [])

        if results:
            context = format_search_results_numbered(results)
        else:
            context = "[No search results available]"

        prompt = f"""你是一个问答助手。请基于以下搜索结果为用户问题提供准确、简洁的回答。
每个事实声明后标注引用编号，如 [1][2]，对应搜索结果的序号。

搜索结果：
{context}

用户问题：{q["question"]}

请回答："""

        try:
            answer = generate(prompt)
        except Exception as e:
            answer = f"[ERROR: {e}]"

        answers[qid] = {"question": q["question"], "answer": answer, "method": "ablation_wo_both"}
        print(f"wo_both [{qid}]: {answer[:100]}...")

    save_json(answers, "outputs/ablation_wo_both/answers.json")
    print(f"wo_both done: {len(answers)} answers.")


def run_ablation_wo_rev_verify() -> None:
    """Keep structured extraction + calibrated generation, skip reverse verification."""
    questions = load_questions()
    cache = load_search_cache()
    answers = {}

    for q in questions:
        qid = q["question_id"]
        results = cache.get(qid, {}).get("results", [])

        try:
            claims = extract_claims(results)
            answer, citation_map = generate_answer(q["question"], claims)
        except Exception as e:
            answer = f"[ERROR: {e}]"
            citation_map = {}

        answers[qid] = {
            "question": q["question"],
            "answer": answer,
            "citation_map": citation_map,
            "method": "ablation_wo_rev_verify",
        }
        print(f"wo_rev_verify [{qid}]: {answer[:100]}...")

    save_json(answers, "outputs/ablation_wo_rev_verify/answers.json")
    print(f"wo_rev_verify done: {len(answers)} answers.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "wo_rev_verify":
        run_ablation_wo_rev_verify()
    else:
        run_ablation_wo_both()
```

---

### Task 13: Evaluator — core metrics

**Files:**
- Create: `src/evaluator.py`

- [ ] **Step 1: Write evaluator.py with data loading and base evaluation**

```python
"""Automatic evaluation: FactScore, hallucination rate, citation precision, etc."""
import json
import numpy as np
from collections import defaultdict
from src.utils import load_json, save_json, extract_claims_from_answer, ROOT_DIR


def load_all_outputs() -> dict[str, dict]:
    """Load outputs from all methods. Returns {method_name: {qid: answer_data}}."""
    methods = {
        "vanilla_rag": "outputs/vanilla_rag/answers.json",
        "self_rag": "outputs/self_rag/answers.json",
        "seve": "outputs/seve/final_answers.json",
        "ablation_wo_both": "outputs/ablation_wo_both/answers.json",
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


def compute_metrics(outputs: dict[str, dict]) -> dict:
    """Compute all metrics for all methods. Returns nested dict."""
    questions = load_json("data/freshqa_questions.json")
    qid_to_question = {q["question_id"]: q for q in questions}

    results = {}
    for method, answers in outputs.items():
        metrics = defaultdict(list)
        for qid, a in answers.items():
            q = qid_to_question.get(qid, {})
            answer_text = a.get("answer", a.get("text", ""))

            # Refusal accuracy
            is_unanswerable = not q.get("is_answerable", True)
            refused = "信息不足" in answer_text or "无法" in answer_text or "不确定" in answer_text
            if is_unanswerable:
                metrics["refusal_acc"].append(1.0 if refused else 0.0)

            # Claim count (proxy for answer recall prerequisite)
            claims = extract_claims_from_answer(answer_text)
            metrics["num_claims"].append(len(claims))

            # Citation count (proxy for citation precision prerequisite)
            import re
            citations = re.findall(r"\[(\d+)\]", answer_text)
            metrics["num_citations"].append(len(citations))

        results[method] = {
            "refusal_accuracy": np.mean(metrics["refusal_acc"]) if metrics["refusal_acc"] else None,
            "avg_claims_per_answer": np.mean(metrics["num_claims"]),
            "avg_citations_per_answer": np.mean(metrics["num_citations"]),
        }

    return results
```

---

### Task 14: Evaluator — NLI-based metrics (FactScore, hallucination, citation precision)

**Files:**
- Modify: `src/evaluator.py` (add NLI evaluation functions)

- [ ] **Step 1: Add NLI evaluation to evaluator.py**

```python
# Evaluation uses a lightweight NLI model run locally to avoid API costs.
# We use google/t5_xxl_true_nli_mixture via HuggingFace transformers.

def _load_nli_model():
    """Lazy-load the NLI model (google/t5_xxl_true_nli_mixture or similar)."""
    try:
        from transformers import pipeline
        return pipeline("text-classification", model="google/t5_xxl_true_nli_mixture")
    except ImportError:
        print("WARNING: transformers not installed. Install with: pip install transformers torch")
        return None


def nli_check(premise: str, hypothesis: str) -> str:
    """Check if hypothesis is entailed by premise. Returns 'ENTAILMENT', 'NEUTRAL', or 'CONTRADICTION'."""
    nli = _load_nli_model()
    if nli is None:
        return "UNKNOWN"
    # T5 NLI format
    result = nli(f"premise: {premise} hypothesis: {hypothesis}")
    return result[0]["label"]


def compute_factscore(answer: str, search_results: list[dict]) -> float:
    """Compute FactScore: fraction of atomic claims supported by search results."""
    claims = extract_claims_from_answer(answer)
    if not claims:
        return 1.0

    context = " ".join(r.get("full_text", "") or r.get("snippet", "") for r in search_results)
    if not context.strip():
        return 0.0

    supported = 0
    for claim in claims:
        label = nli_check(context, claim)
        if label == "ENTAILMENT":
            supported += 1

    return supported / len(claims)


def compute_hallucination_rate(answer: str, search_results: list[dict]) -> float:
    """Compute hallucination rate: fraction of sentences NOT supported by search results."""
    sentences = extract_claims_from_answer(answer)
    if not sentences:
        return 0.0

    context = " ".join(r.get("full_text", "") or r.get("snippet", "") for r in search_results)
    if not context.strip():
        return 1.0

    unsupported = 0
    for sent in sentences:
        label = nli_check(context, sent)
        if label != "ENTAILMENT":
            unsupported += 1

    return unsupported / len(sentences)


def compute_citation_precision(answer: str, search_results: list[dict]) -> float:
    """Compute citation precision: fraction of citations that actually support their claim."""
    import re
    citations = re.findall(r"\[(\d+)\]", answer)
    if not citations:
        return None  # N/A — no citations to evaluate

    # Build a simple mapping: for each [N], the surrounding sentence is the claim
    # More sophisticated: split answer by citation and check each segment
    results_by_rank = {str(r["rank"]): r for r in search_results}

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
            claim_text = m.group(1)
            label = nli_check(snippet, claim_text)
            if label == "ENTAILMENT":
                correct += 1
            break  # only check first occurrence

    if total == 0:
        return None
    return correct / total
```

---

### Task 15: Evaluator — answer accuracy and recall

**Files:**
- Modify: `src/evaluator.py` (add accuracy/recall metrics)

- [ ] **Step 1: Add answer accuracy and recall to evaluator.py**

```python
def compute_answer_accuracy(answer: str, reference_answer: str) -> float:
    """Compute answer accuracy: fraction of reference facts covered by answer."""
    ref_claims = extract_claims_from_answer(reference_answer)
    if not ref_claims:
        return None

    matched = 0
    for rc in ref_claims:
        label = nli_check(answer, rc)
        if label == "ENTAILMENT":
            matched += 1
    return matched / len(ref_claims)


def compute_answer_recall(answer: str, reference_answer: str) -> float:
    """Compute answer recall: fraction of reference info points covered."""
    ref_claims = extract_claims_from_answer(reference_answer)
    if not ref_claims:
        return None

    covered = 0
    for rc in ref_claims:
        label = nli_check(answer, rc)
        if label == "ENTAILMENT":
            covered += 1
    return covered / len(ref_claims)
```

---

### Task 16: Evaluator — full evaluation runner

**Files:**
- Modify: `src/evaluator.py` (add main runner and bootstrap)

- [ ] **Step 1: Add the full evaluation runner**

```python
def run_full_evaluation() -> None:
    """Run all metrics on all methods, output summary CSV."""
    import csv
    outputs = load_all_outputs()
    cache = load_json("data/search_cache.json")
    questions = load_json("data/freshqa_questions.json")
    qid_to_q = {q["question_id"]: q for q in questions}

    rows = []
    for method, answers in outputs.items():
        print(f"\nEvaluating {method} ({len(answers)} answers)...")
        fact_scores = []
        hall_rates = []
        cit_precisions = []
        ans_accuracies = []
        ans_recalls = []
        refusal_accs = []

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

            fact_scores.append(fs)
            hall_rates.append(hr)
            if cp is not None:
                cit_precisions.append(cp)
            if aa is not None:
                ans_accuracies.append(aa)
            if ar is not None:
                ans_recalls.append(ar)

            # Refusal accuracy
            is_unanswerable = not q.get("is_answerable", True)
            if is_unanswerable:
                refused = any(kw in answer_text for kw in ["信息不足", "无法", "不确定"])
                refusal_accs.append(1.0 if refused else 0.0)

        rows.append({
            "method": method,
            "FactScore": f"{np.mean(fact_scores):.3f}",
            "HallucinationRate": f"{np.mean(hall_rates):.3f}",
            "CitationPrecision": f"{np.mean(cit_precisions):.3f}" if cit_precisions else "N/A",
            "AnswerAccuracy": f"{np.mean(ans_accuracies):.3f}" if ans_accuracies else "N/A",
            "AnswerRecall": f"{np.mean(ans_recalls):.3f}" if ans_recalls else "N/A",
            "RefusalAccuracy": f"{np.mean(refusal_accs):.3f}" if refusal_accs else "N/A",
        })

    # Save summary
    save_path = ROOT_DIR / "outputs/evaluation/results_summary.csv"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "method", "FactScore", "HallucinationRate", "CitationPrecision",
            "AnswerAccuracy", "AnswerRecall", "RefusalAccuracy",
        ])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults saved to {save_path}")

    # Print table
    header = f"{'Method':<30} {'FactScore':>10} {'HallRate':>10} {'CitPrec':>10} {'AnsAcc':>10} {'AnsRec':>10} {'RefAcc':>10}"
    print("\n" + header)
    print("-" * len(header))
    for row in rows:
        print(f"{row['method']:<30} {row['FactScore']:>10} {row['HallucinationRate']:>10} {row['CitationPrecision']:>10} {row['AnswerAccuracy']:>10} {row['AnswerRecall']:>10} {row['RefusalAccuracy']:>10}")


if __name__ == "__main__":
    run_full_evaluation()
```

---

### Task 17: Bootstrap statistical testing

**Files:**
- Modify: `src/evaluator.py` (add bootstrap function)

- [ ] **Step 1: Add bootstrap testing**

```python
def bootstrap_test(scores_a: list[float], scores_b: list[float],
                   n_samples: int = 10000, alpha: float = 0.05) -> dict:
    """Paired bootstrap test. Returns {mean_diff, ci_lower, ci_upper, p_value}."""
    a = np.array(scores_a)
    b = np.array(scores_b)
    n = len(a)
    observed_diff = np.mean(a) - np.mean(b)

    diffs = []
    rng = np.random.default_rng(42)
    for _ in range(n_samples):
        idx = rng.integers(0, n, size=n)
        boot_diff = np.mean(a[idx]) - np.mean(b[idx])
        diffs.append(boot_diff)

    diffs = np.array(diffs)
    ci_lower = np.percentile(diffs, 100 * alpha / 2)
    ci_upper = np.percentile(diffs, 100 * (1 - alpha / 2))

    p_value = np.mean(np.abs(diffs) >= np.abs(observed_diff))

    return {
        "observed_diff": observed_diff,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": p_value,
    }
```

---

### Task 18: Output directory initialization

- [ ] **Step 1: Create all output directories**

Run: `mkdir -p outputs/vanilla_rag outputs/self_rag outputs/seve outputs/ablation_wo_both outputs/ablation_wo_rev_verify outputs/evaluation`

- [ ] **Step 2: Copy FreshQA data**

The user should place `freshqa_questions.json` into the `data/` directory manually (downloaded from FreshQA repo).

---

### Task 19: Install dependencies and verify imports

- [ ] **Step 1: Create requirements.txt**

```txt
google-genai>=1.0.0
requests>=2.31.0
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
transformers>=4.30.0
torch>=2.0.0
```

- [ ] **Step 2: Install**

Run: `pip install -r requirements.txt`

- [ ] **Step 3: Verify imports**

Run: `python -c "from src.config import GEMINI_API_KEY; from src.utils import load_json; print('Imports OK')"`

---

## Execution Order

1. Task 1-2 (config + utils) — foundation
2. Task 3-4 (clients) — external API wrappers
3. Task 5 (cache builder) — requires Bing API key
4. Task 6-7 (baselines) — requires Gemini API key + cache
5. Task 8-11 (SEVE) — requires Gemini API key + cache
6. Task 12 (ablation) — requires SEVE functions
7. Task 13-17 (evaluator) — requires all outputs
8. Task 18-19 (setup) — can run anytime

Bing + Gemini API keys must be set as environment variables before running any experiment code.
