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
