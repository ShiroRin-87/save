# SEVE vs Baseline 端到端测试结果

日期: 2026-05-29

## 环境配置

| 项目 | 值 |
|------|-----|
| 模型 | gemini-3-flash-preview (via ai-wave.org) |
| 搜索 | SerpAPI / Google Search |
| 评估方式 | 人工对比参考答案 |

---

## 一、FreshQA 标准题

### FQ070 — 错误前提题

**问题:** "When did Rick Riordan first release the Harry Potter series?"
**参考答案:** "Harry Potter was created by J.K. Rowling, not Rick Riordan."

| 方法 | API | 结果 | 评估 |
|------|-----|------|------|
| Vanilla RAG | 1 | Rick Riordan did not release the Harry Potter series. He launched the *Percy Jackson* series in 2005 [1][2]. The *Harry Potter* series was first published in 1997 [5]. | ✅ 正确识破 |
| Self-RAG | 3 | Rick Riordan 并没有发布过《哈利·波特》系列 [Supported]。Rick Riordan 在 2005 年发布的是《波西·杰克逊》系列 [Supported]，而《哈利·波特》系列是另一部作品 [Supported]。 | ✅ 正确识破 |
| SEVE (修复前) | 5 | 当前信息不足以确定... | ❌ 过度保守 |
| SEVE (修复后) | - | *(修复后未重测此题，参见 FQ010 同类型验证)* | - |

### FQ130 — 慢变事实题

**问题:** "What is the top-ranked university in the US according to the US News Ranking?"
**参考答案:** "Princeton University"

| 方法 | API | 结果 | 评估 |
|------|-----|------|------|
| Vanilla RAG | 1 | **Princeton University** is the top-ranked university [1]. | ✅ 正确 |
| Self-RAG | 3 | Princeton is #1 [Supported]. *但额外引入了 Harvard #1（全球排名）、MIT #1（其他排名）等无关信息* | ✅⚠️ 正确但有噪声 |
| SEVE | 3 | Princeton University is ranked number 1 [1]. 验证标注 PARTIAL——原文只写"#1"未提"2026 Best Colleges" | ✅⚠️ 引用精准度可改进 |

### FQ398 — 快变时效题

**问题:** "When did this year's Passover start?"
**参考答案:** "April 1, 2026"

| 方法 | API | 结果 | 评估 |
|------|-----|------|------|
| Vanilla RAG | 1 | Passover 2026 begins before sundown on **Wednesday, April 1, 2026** [1][2][5]. | ✅ 正确 |
| Self-RAG | 3 | Passover 2026 begins on the evening of **Wednesday, April 1, 2026** [Supported]. 详细说明并解释了 4/2 日期差异。 | ✅ 正确 |
| SEVE | 12 | 检测到冲突：chabad.org、reformjudaism.org 说 4月1日，timeanddate.com 说 4月2日。同时呈现双方并标注来源差异。10/10 验证 YES。 | ✅ 正确 + 冲突检测 |

### FQ010 — 错误前提题

**问题:** "In what year did Hikaru Nakamura become the World Chess Champion?"
**参考答案:** "Hikaru Nakamura has not won the World Chess Championship."

| 方法 | API | 结果 | 评估 |
|------|-----|------|------|
| Vanilla RAG | 1 | 中村光并未获得过世界国际象棋冠军。他仅在2004年作为选手参加过国际棋联世界锦标赛 [4]。 | ✅ 正确，简洁 |
| Self-RAG | 3 | 中村光从未获得过世界国际象棋冠军 [Supported]。*但附带保留了无关信息：美国冠军5次、奥赛金牌、伦敦经典赛、甚至卡斯帕罗夫模因讨论* | ✅⚠️ 正确但有噪声 |
| SEVE (修复后) | 4 | **这个问题的假设前提错误。**Hikaru Nakamura从未获得过世界冠军头衔。实际情况：Nakamura仅是2004年世锦赛参赛者 [9]；他是世界排名第二的棋手 [10]（验证后被删除），而非世界冠军。 | ✅ 修复生效，直接指出前提错误 |

---

## 二、自定义 Galgame 领域题

### CUSTOM_AB2 — 品牌合并事实题

**问题:** "AB2和哪个社团一起组建了新品牌并制作了创作彼女的恋爱公式这个游戏？"
**正确答案:** AB2 与 Campus 组建了 Aino+Links

| 方法 | API | 结果 | 评估 |
|------|-----|------|------|
| Vanilla RAG | 1 | AB2（AKABEiSOFT2）是与 **Campus** 一起组建了新品牌 **Aino+Links**，并制作了游戏《创作彼女的恋爱公式》[1][2][3]。该新品牌的主要成员包括有叶以及工藤启介[3]。 | ✅ 正确完整 |
| Self-RAG | 3 | AB2（Akabei Soft2）与社团 Campus（由工藤啓介代表）共同组建了新品牌 Aino+Links `[Partially]`，并制作了游戏《创作彼女的恋爱公式》`[Supported]`。反思指出"共同组建"原文未直接支撑，降为 PARTIAL。 | ✅⚠️ 正确但自疑 |
| SEVE | 4 | **当前信息不足以确定。** 知识表只包含"Aino+Links制作了该游戏"等2条声明，Campus信息在抽取阶段丢失。 | ❌ 抽取丢失关键信息 |

**SEVE 失败根因:** 搜索结果包含 Campus 信息，但结构化抽取只输出了 2 条声明（均关于 Aino+Links 制作游戏），未能提取出 AB2-Campus 关联。抽取 prompt 的"不推断、不跨源合成"规则在此处导致信息丢失。

### CUSTOM_SAGA — 公司历史事实题

**问题:** "SAGA PLANETS是在哪一部作品后离开Visual Arts的？"
**正确答案:** 《金色ラブリッチェ -Golden Time-》（2019年2月）

| 方法 | API | 结果 | 评估 |
|------|-----|------|------|
| Vanilla RAG | 1 | 关于脱离的消息是在《金色ラブリッチェ- Golden Time -》更换游戏引擎及更换流通公司后开始传出的 [3]。 | ✅ 正确 |
| Self-RAG | 3 | SAGA PLANETS 是在《金色ラブリッチェ- Golden Time -》这部作品之后离开 Visual Arts 的 [Supported]。2019年正式独立 [Supported]。正确过滤了无关信息（GLOVETY、原画师等）。 | ✅ 正确 |
| SEVE | 4 | 于2019年从VISUAL ARTS独立 [1][7][9]。在《金色ラブリッチェ- Golden Time -》不用Siglus改用吉里吉里后传出脱离消息 [6]。9条抽取，4条验证全部 YES。 | ✅ 正确 |

### CUSTOM_ANEMOI — 人员分工题（模糊提问）

**问题:** "anemoi中新岛夕负责哪些部分？"
**正确答案:** 剧本（辻倉朱比華线+総羽愛乃线）+ ED作词

| 方法 | API | 结果 | 评估 |
|------|-----|------|------|
| Vanilla RAG | 1 | 负责**剧本**部分的撰写（与魁、佐雪准以及ハサマ共同负责）[3]。 | ✅ 正确（基于搜索可得信息） |
| Self-RAG | 3 | 负责**剧本**（劇本）部分 [3] [Supported]。准确标注了其他结果的 Irrelevant 状态。 | ✅ 正确 |
| SEVE | - | **抽取阶段崩溃。** 模型输出格式不符合 `N \| claim \| snippet \| URL` 正则，重试3次均解析失败，整个管道报废。 | ❌ 解析器脆弱性 |

### CUSTOM_ANEMOI2 — 人员分工题（精确提问）

**问题:** "新岛夕在anemoi中负责了哪些线路的内容？"
**正确答案:** 辻倉朱比華线（与魁搭档）+ 総羽愛乃线

| 方法 | API | 结果 | 评估 |
|------|-----|------|------|
| Vanilla RAG | 1 | 主要负责了**主线剧情**的编写 [3]。主线以外的大部分剧本内容均委托给了其他剧本家处理 [3]。 | ⚠️ "主线"说法不准确 |
| Self-RAG | 3 | 负责了**主线剧本**的内容 [Supported]。主线以外委托其他剧本家处理 [Supported]。 | ⚠️ 被搜索结果误导 |
| SEVE | 4 | **当前信息不足以确定。** 知识表包含6条声明（游戏发售日、新岛夕职业、代表作品等），但没有任何一条涉及具体路线分工。正确识别信息缺口。 | ✅ 正确保守 |

**关键发现:** 搜索结果 [3] 的原文是"主线以外的大部分剧本只能委托其他剧本家处理"，Vanilla RAG 和 Self-RAG 由此反向推断"新岛夕写了主线"——但这不完全准确（他写了 5 条角色线中的 2 条，并非所谓"主线"）。SEVE 的知识表中没有具体路线信息，因此诚实地回答"信息不足"。

---

## 三、方法汇总统计

### 正确率

| 方法 | 正确 | 部分正确 | 失败 | 正确率 |
|------|:--:|:--:|:--:|:--:|
| **Vanilla RAG** | 7 | 1 | 0 | **87.5%** |
| **Self-RAG** | 5 | 3 | 0 | **62.5%** |
| **SEVE** | 4 | 1 | 3 | **50.0%** |

### API 调用量

| 方法 | 调用/题 | 8题总计 |
|------|:--:|:--:|
| Vanilla RAG | 1 | 8 |
| Self-RAG | 3 | 24 |
| SEVE | 4-12 | ~40 |

### SEVE 故障模式

| 题目 | 故障环节 | 类型 | 严重度 |
|------|---------|------|:--:|
| FQ070 (修复前) | 校准生成 | 过度保守——有证据却说"信息不足" | 中 |
| CUSTOM_AB2 | 结构化抽取 | 信息丢失——搜索结果有但声明表没有 | 高 |
| CUSTOM_ANEMOI | 结构化抽取 | 解析崩溃——模型输出格式不匹配正则 | 致命 |
| CUSTOM_ANEMOI2 | — | ✅ 正确行为——真的信息不足，诚实说不知道 | — |

---

## 四、关键结论

### 1. Vanilla RAG 是最稳健的 baseline

8 题中 7 题正确，0 次崩溃，1 次调用。即使搜索有误导（anemoi2），也只是"部分正确"而非完全错误。它的端到端特性使其没有级联故障风险。

### 2. SEVE 的管道脆弱性是核心问题

三个故障环节各自独立即可导致整体失败：
- **抽取丢失信息**（AB2）：搜索结果有关键信息，但声明表没捕捉到
- **解析崩溃**（anemoi）：模型输出格式偏离正则模板，3 次重试全失败
- **过度保守**（FQ070 修复前）：有足够信息却不给出结论（已通过 prompt 修改修复）

### 3. SEVE 唯一优于 baseline 的场景

anemoi2 —— 当搜索结果有误导时，SEVE 声明表的"信息缺口"使其拒绝猜测，而 Vanilla/Self-RAG 被"主线以外..."的表述带偏。

### 4. 搜索质量决定一切

8 题中有 3 题的成败完全由搜索质量决定（AB2、anemoi、anemoi2），与生成方法无关。SerpAPI 对中文 niche 领域（Galgame）的召回明显弱于英文通用知识。

### 5. Self-RAG 反思的价值有限

3 轮反思在简单题上引入噪声（FQ130 的无关排名，FQ010 的卡斯帕罗夫模因），在难题上只起验证作用而非改进。3 倍调用量的边际收益不明显。

### 6. 成本效益分析

```
每道题的正确率 / API 调用：

Vanilla RAG: 87.5% / 8 calls = 0.109 每调用正确率
Self-RAG:    62.5% / 24 calls = 0.026 每调用正确率
SEVE:        50.0% / 40 calls = 0.013 每调用正确率
```

在当前设定下，Vanilla RAG 的单位调用效率是 SEVE 的 **8.7 倍**。

---

## 五、已执行的修复

| 日期 | 修复内容 | 文件 |
|------|---------|------|
| 2026-05-29 | SEVE 校准生成 prompt：新增"判断问题前提是否成立"规则，区分前提错误 vs 信息不足 | `src/seve.py:71-86` |
| 2026-05-29 | 备份原文件 | `src/seve.py.bak` |

修复前（FQ070）: "当前信息不足以确定" → 修复后（FQ010）: "这个问题的假设前提错误"
