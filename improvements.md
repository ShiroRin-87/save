# SEVE 改进记录

## 已实现

### 1. 多轮迭代搜索 (`iterative_search`)

**文件**：[src/seve.py](src/seve.py)

**效果**：BC008（欢颜）从全军覆没 → 三种方法全对。第1轮搜到汪苏泷 → 提取"周深"线索 → 第2轮搜到周深/欢颜相关内容。

**流程**：
```
初始搜索 → extract_claims → 信息充足? ─否→ _extract_clues → 精炼搜索 → 重复
                              │是
                              └→ 生成 → 验证
```

**新增函数**：
- `iterative_search(question, initial_results, max_rounds=3)` — 主循环
- `_has_enough_info(claims, question)` — 轻量判断，返回 YES/NO
- `_extract_clues(claims, question)` — 从claims提取实体词做搜索关键词

**已验证效果**：
- BC008：汪苏泷 → 周深 → 欢颜 ✓
- BC011：宽泛词 → 蔺相如+廉颇 → 回车巷 ✓
- BC006：苏轼 → 梅尧臣 → 眉山 → 乐山 → 王鹤棣 ✓

---

### 2. 合并抽取+生成 (`extract_and_generate`)

**文件**：[src/seve.py](src/seve.py) `extract_and_generate()`

**效果**：将 extract_claims + generate_answer 合并为1次API调用。省1次调用，验证流程不变。

**格式**：XML标签 `<answer>` / `<sources>`（三层解析兼容XML、旧格式、JSON回退）

**新增**：leak检测——如果模型推理过程泄露到答案中（如 "Let's check" "Wait!"），自动修复或清理。

**已验证效果**：
- BC006：正确答案 2004 ✓（比原版SEVE快 33.2s → 31.9s）
- 弱于两段式：单prompt信息密度不如分步，claims数量通常更少

---

### 3. 逐篇并行抽取 (`extract_claims`)

**文件**：[src/seve.py](src/seve.py)

**效果**：从"全部结果一次性抽取"改为"每篇结果独立抽取 → 合并去重"。抽取质量大幅提升。

| 题目 | 旧版(全量一次) | 新版(逐篇并行) |
|------|--------------|--------------|
| BC011 | 1条垃圾claim | **19条claim** |
| BC022 | 1条claim | **8条claim** |

**实现**：
- `_extract_from_one_result(result, question)` — 单篇抽取（5-15K字符）
- `extract_claims()` — 用 ThreadPoolExecutor(max_workers=5) 并行调用
- 合并时按claim文本去重，保留更长snippet

**关键提示词改进**："宁可多提不要漏提"替代"只提取明确陈述的事实"。

---

### 4. 上下文截断简化 + Jina质量检测

**文件**：[src/bing_client.py](src/bing_client.py)

**改动**：
- 移除 `_split_query_keywords()` 和 `_extract_context()` —— 关键词匹配对谜题式问题无效（问题中不含答案词）
- 改为简单截断：Jina全文 → 前15K字符
- 新增 Jina 质量检测：返回 <500字符 或含 CAPTCHA/安全验证 → 标记 `jina_quality: "low_quality"`
- `JINA_MAX_CHARS` 降至 15K（适配逐篇抽取）

---

### 5. 并行验证

**文件**：[src/seve.py](src/seve.py) `verify_claims()`

fallback 验证从串行改为 `ThreadPoolExecutor(max_workers=8)` 并行。

---

### 6. 0-claims 安全回退

当 `extract_claims` 返回空时，不再调用模型从自身知识回答，直接返回"当前信息不足以确定"。防止模型绕开搜索结果编造答案。

---

## 未实现（设计存留）

### DeepSeek 验证（省成本）

验证任务改为 DeepSeek API，`deepseek-chat` 极便宜。需配置 API key。

### 多问题并行处理

`run_seve()` 逐题串行改为 ThreadPoolExecutor 多题并行。

---

## 测试结果汇总（6题 × 4方法）

| 题目 | 答案 | Vanilla | Self-RAG | SEVE | SEVE-Merged |
|------|------|---------|----------|------|-------------|
| BC006 艺术 | 2004 | 2004 ✓ | 2004 ✓ | 2004 ✓ | 2004 ✓ |
| BC008 音乐 | 欢颜 | 欢颜 ✓ | 欢颜 ✓ | 欢颜 ✓ | 信息不足 |
| BC011 地理 | 回车巷 | 回车巷 ✓ | 回车巷 ✓ | 回车巷 ✓ | 信息不足 |
| BC022 历史 | 王安石 | 苏轼 ✗ | 信息不足 | 候选集 | 信息不足 |
| BC070 地理 | 青城山 | 山海关 ✗ | 居庸关 ✗ | 信息不足 | 乱码 ✗ |
| BC080 音乐 | 梁博 | 信息不足 | 信息不足 | 推理泄露 ✗ | 信息不足 |

**关键发现**：
- Vanilla 会自信幻觉（BC022: "苏轼《六一居士集叙》"，BC070: "山海关"），编造细节看起来很可信
- SEVE 逐篇抽取后用更多claim正确回答了 BC006/BC008/BC011
- BC022 搜索中缺少"器质深厚 → 王安石"映射，SEVE正确列出候选集但不猜测
- SEVE-Merged 综合表现弱于两段式SEVE

---

## 待解决

1. **SEVE-Merged 信息密度不足**：单prompt的claims数量远少于分步。可能需要更大的 MAX_OUTPUT_TOKENS 或分步引导
2. **搜索仍是天花板**：BC070(青城山)/BC080(梁博) 搜索根本没返回相关内容
3. **Jina 知乎拦截**：CAPTCHA导致关键页面只拿到344字符
4. **`_has_enough_info` 依赖claim质量**：claim少则判断"不够"→额外搜索浪费。应改为直接判断搜索结果质量
