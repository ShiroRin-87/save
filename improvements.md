# SEVE 改进方案

## 1. 验证步骤换用小模型（省成本，不省调用次数）

验证任务只是简单的 YES/NO/PARTIAL 判断，不需要 `gemini-3-flash-preview` 级别的大模型。

### 改动点

**config.py** — 新增验证专用模型配置：
```python
VERIFY_MODEL_NAME = "gpt-4o-mini"  # 或 gemini-2.0-flash-lite 等便宜模型
```

**model_client.py** — `generate()` 支持模型覆盖：
```python
def generate(prompt: str, *, temperature: float = TEMPERATURE, model: str = None) -> str:
    response = _client.chat.completions.create(
        model=model or MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=MAX_OUTPUT_TOKENS,
    )
    return response.choices[0].message.content or ""
```

**seve.py** — `verify_claims()` 中传入验证模型（共 2 处）：
```python
# 批处理验证 (line 179)
result_text = generate(batch_prompt, model=VERIFY_MODEL_NAME)

# fallback 单条验证 (line 211)
fb = generate(fallback_prompt, model=VERIFY_MODEL_NAME).strip().upper()
```

代价极低 — 验证用的是 YES/NO 判断，小模型完全够用。

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
| 验证 | gemini-3-flash (1次) | **gpt-4o-mini** (1次) |
| **总计** | **3 次大模型** | **1 次大模型 + 1 次小模型** |
