# SEVE D方法全流程详解

D方法是一个**假设驱动 + 搜索验证**的多轮管道，核心思想是：LLM 的参数化知识可以桥接"谜题语言"到"答案语言"的鸿沟，搜索只用来验证。

实现文件：[src/seve.py](src/seve.py)（1294行）

---

## 管道总览

```
D1 直接搜索 → D2 判断充分性 → D3 假设搜索 → D4 声明提取 → D5 生成答案 → D6 缺口填补 → D8 反向验证 → D9 修正 → D10 兜底推理
```

**关键突破点：D3a = V0**。V0 实验就是把 D3a 这一步单独拎出来，绕过下游的噪声管道，直接评估 LLM 参数化知识的命中率。

---

## D1 — 直接搜索（Direct Search）

**函数**：直接在 `hypothesis_driven_search()` 中用 `do_search(question)` 调用

用完整的谜题原文作为搜索 query 丢给 Google。BrowseComp 谜题与答案之间零词汇重叠（如谜题说"距战国遗址10公里的唐代命名地点"，答案是"青城山"），直接搜索几乎不可能命中。

**结果**：这一步的结果被送到 D2 做质量判断。

---

## D2 — 判断是否需要假设搜索

**函数**：[`needs_hypothesis_search()`](src/seve.py#L203-L233)

把 D1 搜索结果的前8条（标题+摘要）喂给 LLM，让它判断这些结果是否与问题高度相关、可能包含答案。

```python
# 核心prompt逻辑
prompt = """以下是对一个问题的直接搜索结果。判断这些结果是否看起来与问题高度相关...
输出true或false。只输出JSON布尔值。"""
```

- 返回 `True` → 结果不足，进入 D3 假设搜索分支
- 返回 `False` → 结果看起来够了，跳过 D3

由于 BrowseComp 的 riddle-answer 词汇鸿沟，这一步几乎总是返回 `True`（需要假设搜索）。

---

## D3a — 生成候选假设（= V0）

**函数**：[`generate_hypotheses()`](src/seve.py#L80-L110)

**这就是 V0！** V0 实验等于把这一步从管道中独立出来单独评估。

让 LLM 凭借参数化知识直接猜测 5 个可能的最终答案：

```python
prompt = f"""你是一个知识渊博的推理助手。请根据以下谜题的约束条件，依靠你自己的知识推测可能的答案。

谜题：{question}

输出一个JSON字符串数组，包含{n}个方向各异的最终候选答案。每个元素直接是答案名称，覆盖不同方向。只输出JSON数组，不要其他文字。

示例：["青城山","都江堰","乐山大佛","峨眉山","莫高窟"]

JSON数组："""
```

### V0 为什么有效

DeepSeek V4 Pro 在推理阶段（`reasoning_content`）会做多跳推理链：
- BC001：`刊物→五卅运动→上海大学→演员→结婚日期→2019年4月23日`
- BC007：`石黑一雄→徐向前→黄杰→1946年5月`
- BC024：`伊藤智彦→戸松遥→SAO→魔法使之夜`

这些推理链被隐式地编码在模型的参数化知识中，不需要搜索就能追踪。

### V0 命中率

经用户校正后，**16 题中 13 题命中，命中率 81%**。3 道真正 miss 的题都需要极端小众的知识链。

---

## D3b — 约束词组提取

**函数**：[`_extract_constraints()`](src/seve.py#L236-L247)

从谜题中提取 3-5 个可搜索的关键词/短语，用于构造搜索 query。

```python
prompt = f"""从以下谜题中提取3-5个可用于搜索验证的关键词或短语。
谜题：{question}
输出一个JSON字符串数组。只输出JSON数组。"""
```

例如从"距战国遗址10公里的唐代命名地点"提取 `["战国遗址", "唐代", "命名", "10公里"]`。

**这是 D3c 搜索的前置步骤。**

---

## D3c — 假设+约束混合搜索（瓶颈）

**函数**：[`search_and_verify_hypothesis()`](src/seve.py#L250-L265)

对每个候选假设，拼接约束词组后进行 Google 搜索：

```python
query = f"{hypothesis} {constraints}"
# 例如: "青城山 唐代 命名 10公里 战国遗址"
results = do_search(query, top_k=5)
```

在 `hypothesis_driven_search()` 中，所有假设的搜索是**并行**的（ThreadPoolExecutor，5 workers）。

### 这是整个 D 方法的最大瓶颈

Google 的 BERT 语义匹配对**短答案字符串不友好**。比如 query `"蒂特 足球"` — "蒂特"这个词太短、太像常见词，Google 倾向返回语义相关但无关的结果。

V0 实验证明了模型猜对比例很高（81%），但 D3c 的搜索召回率显著拖累了管道。搜索失败的原因：
- 短答案词被 Google 语义模糊化
- 约束词 + 假设的组合可能过长，稀释了关键信号
- 没有使用精确引号 `"蒂特"` 强制字面匹配

---

## D3d — 直接搜索兜底

**代码位置**：[`hypothesis_driven_search()`](src/seve.py#L325-L332)

并行假设搜索结束后，再补一次完整谜题的直接搜索，确保覆盖率：

```python
direct_results = do_search(question, top_k=3)
```

---

## D4 — 声明提取（Claims Extraction）

**函数**：[`extract_claims()`](src/seve.py#L633-L689)

从搜索结果页面中提取结构化的事实声明。这是管道中最贵的步骤之一。

### 三步流程

1. **相关性过滤**（`_filter_relevant_results()`，line 513）：用一次便宜的 LLM 调用，从所有结果中筛选 top-10 最相关的页面，避免把无关页面送进昂贵的提取步骤。

2. **批量提取**（`_extract_from_batch()`，line 550）：每 3 个页面打包成一个 batch，一次 API 调用提取所有声明。输出 JSON 数组 `[{"claim": "...", "snippet": "...", "url": "..."}]`。

3. **并行处理**（line 660）：多个 batch 用 ThreadPoolExecutor（max 4 workers）并行提取。

4. **去重**（line 674）：按规范化后的 claim 文本去重，保留 snippet 更长的版本。

```python
# 提取prompt的核心指令
"""关键原则：即使信息不完整、不直接、看似只匹配问题的一小部分，也应该提取。宁可多提不要漏提。"""
```

---

## D5 — 生成答案

两个变体：

### D5a — 结构化生成（`generate_answer()`，line 853）

把 D4 提取的 claims 排成结构化知识表，让 LLM 基于此生成带引用编号 `[1][2]` 的答案。

```python
prompt = """规则：
1. 每个事实声明后标注引用编号
2. 如果标记了[冲突]，需同时呈现多方说法
3. 首先判断问题前提是否成立
4. 只有完全无相关信息时才写"当前信息不足以确定"
5. 单一来源标注"据[来源名称]"
6. 不要编造不存在的信息"""
```

### D5b — 合并提取+生成（`extract_and_generate()`，line 785）

把 D4+D5 合并成一次 API 调用，用 XML 标签格式输出：

```xml
<answer>
[回答文本，包含[1][2]引用]
</answer>

<sources>
[{"id":1,"claim":"...","snippet":"...","url":"..."}]
</sources>
```

---

## D6 — 缺口填补搜索（Gap-Fill）

**函数**：[`gap_fill_search()`](src/seve.py#L142-L199)

当答案识别了中间实体但没完成最后一步时（如 A→B→C 但卡在 C→D），迭代搜索缺失的链接。

### 流程
1. 从当前不完整答案中提取 2-5 个补充搜索关键词
2. 搜索这些关键词
3. 将新结果追加到上下文，重新生成答案
4. 检查答案是否完整/不再变化（bigram Jaccard 相似度 ≥ 0.75 视为 stable）
5. 最多 2 轮

```python
# 判断答案完整的逻辑 (_answer_looks_complete, line 127)
# 不再信任 LLM 的自信心——改为检测答案是否不再改善
if prev_answer and _answer_similar(answer, prev_answer, threshold=0.75):
    return True  # stable, stop
```

---

## D7 — 多轮迭代搜索

**函数**：[`iterative_search()`](src/seve.py#L460-L508)

与 D6 互补的更一般化的多轮搜索。每轮：
1. 判断当前搜索结果是否充分（`_has_enough_info()`，基于标题+摘要判断）
2. 提取关键实体（人名、地名等）作为精炼搜索词
3. 用 `f"{question} {entities}"` 精炼搜索
4. 追加新结果，最多 2 轮

---

## D8 — 反向验证（Reverse Verification）

**函数**：[`verify_claims()`](src/seve.py#L889-L993)

把 D5 生成的答案中引用的每个 `[N]` 声明，回原文验证。

### 流程
1. 从答案中提取所有 `[数字]` 引用编号
2. 找到对应的 claim 和原始搜索文本
3. **批量验证**：一次 API 调用验证所有 claims，输出 YES/PARTIAL/NO
4. **并行兜底**：batch 中未解析成功的 claims，逐个并行验证（ThreadPoolExecutor, 8 workers）

```python
batch_prompt = """对以下 JSON 数组中的每组声明和原文，判断原文是否包含声明中声称的信息。
为每个元素添加 "verdict" 字段，值为 YES/PARTIAL/NO。
只输出 JSON 数组。"""
```

---

## D9 — 应用验证结果

**函数**：[`apply_verification()`](src/seve.py#L996-L1011)

根据 D8 的验证结果修正答案：
- **NO** → 从答案中删除该声明，添加备注 `[已删除 — 原文不支撑]`
- **PARTIAL** → 保留但标注 `[部分支撑]`
- **YES** → 保留不改

---

## D10 — 兜底推理（Fallback Reasoning）

**函数**：[`fallback_reasoning()`](src/seve.py#L1032-L1088)

当整个搜索管道都失败时（答案短、全是"不确定"、验证 YES=0），回退到 LLM 纯参数化知识。

### 流程
1. 让 LLM 直接猜答案（类似 V0，但只用 1 个猜测）
2. 把问题拆成 3-8 个可验证的约束条件
3. 逐个验证猜测是否满足每个约束
4. 通过 ≥50% 约束 → 输出 `[参数化知识推理，N/M 约束通过]`

```python
# 判断管道失败的逻辑 (_is_failed_answer, line 1014)
if answer < 30 chars → failed
if contains "无法/不确定/不足以" → failed
if YES count == 0 → failed
if NO > YES and YES ≤ 1 → failed
```

---

## V0 与 D 方法的关系

| | V0 | D方法 |
|---|---|---|
| 实现函数 | `generate_hypotheses()` | D3a 调用同一函数 |
| Prompt | 完全相同 | 完全相同 |
| 用途 | 独立评估 LLM 猜答案能力 | 作为搜索管道的假设输入 |
| 下游 | 无（直接对比 ref） | D3b→D3c→D4→D5→... |

**V0 的价值**：证明了模型猜对的概率（81%）远高于 D 方法最终命中率。差距来自 D3c 的搜索瓶颈和 D4 的提取损耗。改进方向应该是让搜索更精确地验证假设，而不是改进假设生成。

---

## 辅助方法

### 方法 F — 链式推理 + SEVE

**函数**：[`chain_reasoning()`](src/seve.py#L1091-L1234)

1. **F1 分解**（`decompose_question()`）：把复杂谜题拆成 2-5 个有序子问题
2. **F2 逐跳搜索**：顺序搜索每个子问题，不确定则重试替代搜索词，已确认事实累积传递
3. **F3-F7**：在所有累积结果上跑完整 SEVE 管道（D4→D5→D8→D9→D10）

### 问题分解（备用入口）

**函数**：[`decompose_question()`](src/seve.py#L339-L354)、[`decompose_and_search()`](src/seve.py#L357-L389)

独立的问题分解 + 并行搜索，不依赖假设生成。适用于可以用多个独立维度搜索的题目。
