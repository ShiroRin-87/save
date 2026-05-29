# 端到端测试结果

日期: 2026-05-29

## 环境配置

| 项目 | 值 |
|------|-----|
| 模型 | gemini-3-flash-preview (via ai-wave.org) |
| 搜索 | SerpAPI / Google Search |
| 数据集 | FreshQA (600 题，完整下载) |

---

## 测试 1：单题快速验证

**问题:** "Who is the current CEO of OpenAI?"

| 方法 | API 调用 | 结果 |
|------|---------|------|
| 搜索 | 1 | 3 条结果 (Wikipedia, clay.com, openai.com) |
| Vanilla RAG | 1 | "The current CEO of OpenAI is Sam Altman [1][2][3]" |
| Self-RAG | 3 | 3 轮反思 → "OpenAI 的 CEO 是 Sam Altman [Supported]" |
| SEVE | 6 | 12 条声明 → 3 条引用全判 YES，无幻觉 |
| Ablation | 2 | 抽取+生成，跳过了验证 |

**SEVE 12 条声明的完整抽取：**

| ID | 声明 | 原文片段 | 来源 |
|----|------|---------|------|
| 1 | Sam Altman是Loopt的联合创始人 | "Loopt (co-founder)" | wikipedia.org |
| 2 | Sam Altman是OpenAI的CEO | "CEO of OpenAI" | wikipedia.org |
| 3 | Sam Altman是Helion Energy的董事长 | "Chairman of Helion Energy" | wikipedia.org |
| 4 | Sam Altman曾任Y Combinator的总裁（直至2019年） | "President of Y Combinator (until 2019)" | wikipedia.org |
| 5 | Sam Altman与Oliver Mulherin于2024年结婚 | "Oliver Mulherin. (m. 2024)" | wikipedia.org |
| 6 | Sam Altman是一位有影响力的企业家和投资者 | "Sam Altman is an influential entrepreneur and investor" | clay.com |
| 7 | Sam Altman目前担任OpenAI的CEO | "currently serving as the CEO of OpenAI" | clay.com |
| 8 | OpenAI是一个领先的人工智能研究机构 | "OpenAI, a leading artificial intelligence research organization" | clay.com |
| 9 | Sam Altman回归担任CEO | "Sam Altman returns as CEO" | openai.com |
| 10 | OpenAI拥有一个新的初始董事会 | "OpenAI has a new initial board" | openai.com |
| 11 | Mira Murati担任CTO | "Mira Murati as CTO" | openai.com |
| 12 | Greg Brockman回归担任总裁 | "Greg Brockman returns as President" | openai.com |

**搜索原始文本：**

[1] Wikipedia: `Sam Altman ; Loopt (co-founder) · CEO of OpenAI; Chairman of Helion Energy; President of Y Combinator (until 2019) · Oliver Mulherin. (m. 2024) · 1.`

[2] clay.com: `Sam Altman is an influential entrepreneur and investor, currently serving as the CEO of OpenAI, a leading artificial intelligence research organization.`

[3] openai.com: `Sam Altman returns as CEO, OpenAI has a new initial board. Mira Murati as CTO, Greg Brockman returns as President. Read messages from CEO Sam ...`

---

## 测试 2：三题多类型验证

### FQ070 — 错误前提题 (never-changing, answerable=False)

**问题:** "When did Rick Riordan first release the Harry Potter series?"
**参考答案:** "Harry Potter was created by J.K. Rowling, not Rick Riordan."

| 方法 | 结果 | 评估 |
|------|------|------|
| Vanilla RAG | "Rick Riordan did not release Harry Potter. He launched Percy Jackson in 2005. Harry Potter was first published in 1997." | 正确识破错误前提 |
| Self-RAG | "Rick Riordan 没有发布哈利波特系列 [Supported]。Rick Riordan 在 2005 年发布了波西杰克逊系列 [Supported]。" | 正确识破 |
| SEVE | 11 条声明 → "当前信息不足以确定。Rick Riordan 在 2005 年推出了波西杰克逊系列 [1][3]，哈利波特系列第一部在 1997 年问世 [10]。知识表中未提及 Rick Riordan 发布哈利波特。" | 正确，但表述过于谨慎 |
| SEVE 验证 | 3 条引用：[1] YES, [3] YES, [10] YES | 全部通过 |

### FQ130 — 慢变事实题 (slow-changing, answerable=True)

**问题:** "What is the top-ranked university in the US according to the US News Ranking?"
**参考答案:** "Princeton University"

| 方法 | 结果 | 评估 |
|------|------|------|
| Vanilla RAG | "Princeton University is the top-ranked [1]." | 正确 |
| Self-RAG | "Princeton University is the top-ranked [Supported]." | 正确 |
| SEVE | 2 条声明 → "Princeton University is ranked number 1 in the U.S. News 2026 Best Colleges ranking [1]." | 正确 |
| SEVE 验证 | [1] PARTIAL — 原文部分支撑 | 引用精准度可改进 |

### FQ398 — 快变时效题 (fast-changing, answerable=True)

**问题:** "When did this year's Passover start?"
**参考答案:** "April 1, 2026"

| 方法 | 结果 | 评估 |
|------|------|------|
| Vanilla RAG | "Passover 2026 begins before sundown on Wednesday, April 1, 2026 [1][2][5]." | 正确 |
| Self-RAG | "Passover 2026 begins on the evening of Wednesday, April 1, 2026 [Supported]." | 正确 |
| SEVE | 11 条声明 → **检测到冲突**：chabad.org、reformjudaism.org、almanac.com 都说 4月1日，但 timeanddate.com 说 4月2日。同时呈现双方并标注来源差异。 | 正确 + 冲突检测生效 |
| SEVE 验证 | 10/10 条引用 YES | 全部通过 |

---

## 流程验证总表

| 步骤 | 状态 |
|------|------|
| FreshQA 数据集下载 (Google Sheets → 600 题 JSON) | 通过 |
| SerpAPI 搜索 | 通过 |
| ai-wave.org 模型 API | 通过 |
| search_cache.json 缓存 | 通过 |
| Vanilla RAG | 通过 |
| Self-RAG (3 轮) | 通过 |
| SEVE 结构化抽取 | 通过 |
| SEVE 校准生成 | 通过 |
| SEVE 反向验证 | 通过 |
| SEVE 冲突检测 | 通过 (FQ398 Passover 日期多源冲突) |
| SEVE 错误前提处理 | 通过 (FQ070 Rick Riordan) |
| 消融实验 | 通过 |
| 全部 outputs/ 文件生成 | 通过 |

## 关键发现

1. **SEVE 的冲突检测有价值**：FQ398 中 timeanddate.com 与其他来源的日期矛盾被成功捕获并呈现
2. **SEVE 对错误前提问题的回答过于保守**：FQ070 应该直接说明"Harry Potter 不是 Rick Riordan 写的"，而非"信息不足"
3. **简单事实题上三种方法表现一致**：FQ130 (Princeton) 所有人都答对了
4. **抽取质量与 snippet 信息密度正相关**：Wikipedia info-box 型 snippet 能抽出 5 条，普通文本型 2-3 条

## 已知问题

- 终端中文乱码（Windows GBK），JSON 文件内数据完整
- NLI 模型需单独安装 transformers + torch 才能跑评估
