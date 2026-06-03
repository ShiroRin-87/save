# SEVE 改进方案

## 1. 验证步骤换用 DeepSeek API（省成本，不省调用次数）

验证任务只是简单的 YES/NO/PARTIAL 判断，不需要 `gemini-3-flash-preview` 级别的大模型。改用 DeepSeek。

### 改动点

**config.py** — 新增 DeepSeek 配置：
```python
# DeepSeek (for verification only)
DEEPSEEK_API_KEY = "sk-xxx"  # 替换为实际 key
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
VERIFY_MODEL_NAME = "deepseek-chat"
```

**model_client.py** — 新增 DeepSeek 客户端 + `generate()` 支持模型/客户端覆盖：
```python
from openai import OpenAI
from src.config import (
    GEMINI_API_KEY, MODEL_NAME, TEMPERATURE, MAX_OUTPUT_TOKENS, API_BASE_URL,
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, VERIFY_MODEL_NAME,
)

_client = OpenAI(api_key=GEMINI_API_KEY, base_url=API_BASE_URL + "/v1", timeout=120.0)
_verify_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL + "/v1", timeout=60.0)


def generate(prompt: str, *, temperature: float = TEMPERATURE, model: str = None, client: str = "default") -> str:
    """Send a prompt and return the text response.
    
    client: "default" uses Gemini, "verify" uses DeepSeek.
    """
    c = _verify_client if client == "verify" else _client
    m = model or (VERIFY_MODEL_NAME if client == "verify" else MODEL_NAME)
    response = c.chat.completions.create(
        model=m,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=MAX_OUTPUT_TOKENS,
    )
    return response.choices[0].message.content or ""
```

**seve.py** — `verify_claims()` 中传入 DeepSeek（共 2 处）：
```python
# 批处理验证 (line 179)
result_text = generate(batch_prompt, client="verify")

# fallback 单条验证 (line 211)
fb = generate(fallback_prompt, client="verify").strip().upper()
```

DeepSeek 的 YES/NO 判断完全够用，`deepseek-chat` 极便宜（￥1/百万token）。

---

## 2. 合并抽取+生成（省 1 次 API 调用）

去掉独立的 `extract_claims`，把提取规则嵌入生成 prompt，3 次调用变 2 次：
```
当前: 搜索结果 → [API 1: 抽取] → 结构化表 → [API 2: 生成] → [API 3: 验证]
改进: 搜索结果 → [API 1: 抽取+生成] → 回答 → [API 2: 验证]
```
`verify_claims` 仍然可用 — 直接从回答中的 `[1][2]` 引用编号匹配 search_cache 原文验证。

---

## 3. 两者结合

最终效果：**2 次 API 调用**（1 次大模型生成 + 1 次小模型验证）

| | 当前 | 改进后 |
|---|---|---|
| 抽取 | gemini-3-flash (1次) | — |
| 生成 | gemini-3-flash (1次) | gemini-3-flash (1次) |
| 验证 | gemini-3-flash (1次) | **deepseek-chat** (1次) |
| **总计** | **3 次大模型** | **1 次 Gemini + 1 次 DeepSeek** |

---

## 4. 并行验证 + 并行处理

### 4a. 验证并行化

当前 `verify_claims` 是批处理 + 逐个 fallback 串行：

```
当前: 批处理验证 (1次) → 逐个 fallback (N次串行)
改进: 批处理验证 (1次) → fallback 全部并发 (N次并行)
```

**seve.py** — fallback 改为并行：
```python
from concurrent.futures import ThreadPoolExecutor, as_completed

# 收集 UNKNOWN 的 claim
unknown_claims = [(cid, claim_by_id[cid]) for cid, v in verdict_map.items() if v == "UNKNOWN" and cid in claim_by_id]

# 并行 fallback
def _verify_one(cid, claim):
    original_result = url_to_result.get(claim["url"])
    original_text = (original_result.get("full_text") or "") if original_result else ""
    fb = generate(fallback_prompt_for(claim["sentence"], original_text), client="verify")
    return cid, fb.strip().upper()

with ThreadPoolExecutor(max_workers=8) as ex:
    futures = {ex.submit(_verify_one, cid, c): cid for cid, c in unknown_claims}
    for f in as_completed(futures):
        cid, verdict = f.result()
        if verdict in ("YES", "PARTIAL", "NO"):
            verdict_map[cid] = verdict
```

### 4b. 多问题并行处理

当前 `run_seve()` 是逐题串行 `for q in questions`，改成多题并行：
```python
with ThreadPoolExecutor(max_workers=4) as ex:
    futures = {ex.submit(process_one_question, q): q["question_id"] for q in questions}
    for f in as_completed(futures):
        qid = futures[f]
        all_final[qid] = f.result()
```

4 并发时 50 题约 78s × 13 = ~17 分钟（串行 ~65 分钟）。

---

## 5. 多轮搜索（突破 BrowseComp-ZH 类多跳推理题瓶颈）

### 问题

当前流程是单轮搜索 → 抽取 → 生成。BrowseComp-ZH 实测 10 题仅 1 题搜索命中答案，因为：
- 每道题需要 3-6 步链式推理
- 初始搜索词就是问题全文，太模糊
- Google 不善于链接多个不相关实体

### 方案：迭代精炼搜索

把抽取变成迭代的——每轮从已有结果中提取线索，用线索精炼下一轮搜索：

```
当前: 搜索 → Jina → [抽取 → 生成 → 验证]
改进: 搜索 → Jina → [抽取 → 信息充足? ──否──→ 提取线索 → 精炼搜索 → Jina → 抽取 → ...]
                        │是
                        └→ 生成 → 验证
```

### 新增函数

**`iterative_search()`** — 多轮搜索主循环：

```python
def iterative_search(question: str, initial_results: list[dict],
                     max_rounds: int = 3) -> list[dict]:
    """Multi-round search: extract clues from results, refine query, repeat.
    
    Returns accumulated results from all rounds.
    """
    from src.bing_client import search as do_search
    
    all_results = list(initial_results)
    
    for rnd in range(max_rounds):
        # Extract claims from current accumulated results
        claims = extract_claims(all_results, question)
        
        # Check if we have enough information
        if _has_enough_info(claims, question):
            break
        
        # Extract key entities/clues from claims for refined search
        clues = _extract_clues(claims, question)
        if not clues:
            break
        
        # Refined search with clues
        refined_query = f"{question} {clues}"
        new_results = do_search(refined_query)
        
        # Enrich with Jina
        new_results = _enrich_results(new_results)
        all_results.extend(new_results)
        
        print(f"  [Round {rnd+1}] refined query: {refined_query[:80]}... "
              f"→ {len(new_results)} new results")
    
    return all_results
```

**`_has_enough_info()`** — 轻量判断，比完整 generate 快：

```python
def _has_enough_info(claims: list[dict], question: str) -> bool:
    """Lightweight check: can the claims answer the question?"""
    if not claims:
        return False
    
    # Build a compact summary of claims
    claim_summary = "\n".join(
        f"- {c['claim']}" for c in claims[:10]
    )
    
    prompt = f"""基于以下已知事实，判断能否回答用户问题。只回答 YES 或 NO。

已知事实：
{claim_summary}

用户问题：{question}

能否回答？"""
    
    result = generate(prompt, client="verify").strip().upper()
    return result.startswith("YES")
```

**`_extract_clues()`** — 从已有事实中抓实体做新搜索词：

```python
def _extract_clues(claims: list[dict], question: str) -> str:
    """Extract key entities from claims to use as refined search terms."""
    if not claims:
        return ""
    
    claim_texts = "\n".join(
        f"- {c['claim']}" for c in claims[:15]
    )
    
    prompt = f"""从以下已知事实中，提取可以用来精炼搜索的关键实体（人名、地名、作品名、时间等）。
用空格分隔，只输出实体词，不超过 10 个。

已知事实：
{claim_texts}

原始问题：{question}

关键实体："""
    
    return generate(prompt, client="verify").strip()
```

**`_enrich_results()`** — 复用 Jina 抓取：

```python
def _enrich_results(results: list[dict]) -> list[dict]:
    """Fetch Jina full text for new search results."""
    enriched = []
    for r in results:
        entry = dict(r)
        url = r["url"]
        from src.bing_client import _jina_fetch
        full = _jina_fetch(url)
        if full:
            # Truncate to 30K chars to avoid context explosion
            entry["full_text"] = full[:30000]
            entry["full_text_source"] = "jina"
        else:
            entry["full_text"] = r.get("snippet", "")
            entry["full_text_source"] = "snippet"
        entry["full_text_len"] = len(entry["full_text"])
        enriched.append(entry)
    return enriched
```

### 调用方式

对 seve.py `run_seve()` 的改动极小，只在获取 results 后多一步：

```python
# 现在的 seve.py run_seve()
for q in questions:
    results = cache.get(qid, {}).get("results", [])
    claims = extract_claims(results, q["question"])       # 单轮
    ...

# 改为
for q in questions:
    results = cache.get(qid, {}).get("results", [])
    results = iterative_search(q["question"], results)     # 多轮
    claims = extract_claims(results, q["question"])        # 不变
    ...
```

`extract_claims` / `generate_answer` / `verify_claims` / `apply_verification` 的接口完全不变。

### 成本估算

| | 当前(单轮) | 多轮(平均2轮) | 多轮(最坏3轮) |
|---|---|---|---|
| 搜索(Bing) | 1次 | 2次 | 3次 |
| Jina抓取 | 5-10次 | 5-10次(只抓新增) | 5-10次(只抓新增) |
| extract_claims | 1次 | 2次 | 3次 |
| _has_enough_info | 0 | 1-2次(轻量) | 2次(轻量) |
| _extract_clues | 0 | 1-2次(轻量) | 2次(轻量) |
| generate/verify | 不变 | 不变 | 不变 |

关键：`_has_enough_info` 和 `_extract_clues` 都用 DeepSeek `client="verify"`（便宜），每条仅需 2-5 秒。

### 预期效果

对 BrowseComp-ZH 类多跳题：
- 第一轮搜到"苏轼""欧阳修""梅尧臣" → 第二轮精准搜"苏轼出生地 南部城市 顶流明星 2022剧"
- 命中率预计从 ~10% 提升到 ~40-50%（参考 DeepResearch 的 42.9%）
- 平均每题材耗增加约 30-60 秒（取决于轮数）

### Jina 截断（附带优化）

`_enrich_results` 中默认截断单页到 30K chars：
- 当前实测单页可达 200K+ chars（Wikipedia 全页），对模型无意义
- 30K chars 足够覆盖所有关键信息
- 解决 BC011/BC017 那样的 700K-1.3M 上下文爆炸问题
