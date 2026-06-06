"""Generate SEVE pipeline flow diagram using matplotlib."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.font_manager import FontProperties, fontManager
import numpy as np

# Chinese font — use Microsoft YaHei with explicit fname
_CN_FONT_PATH = 'C:/Windows/Fonts/msyh.ttc'
_CN_FONT = FontProperties(fname=_CN_FONT_PATH, size=10)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei'] + plt.rcParams['font.sans-serif']
# Force re-scan
fontManager.addfont(_CN_FONT_PATH)

fig, ax = plt.subplots(1, 1, figsize=(28, 40))
ax.set_xlim(0, 28)
ax.set_ylim(0, 40)
ax.axis('off')

# Color scheme
C_INPUT = '#1a1a2e'
C_SEARCH = '#16213e'
C_EXTRACT = '#0f3460'
C_GENERATE = '#533483'
C_VERIFY = '#e94560'
C_CORRECT = '#c23152'
C_FALLBACK = '#f5a623'
C_OUTPUT = '#2d6a4f'
C_ARROW = '#555555'
C_LABEL = '#ffffff'
C_BG = '#fafafa'

fig.patch.set_facecolor(C_BG)
ax.set_facecolor(C_BG)

def draw_box(x, y, w, h, text, color, fontsize=13, text_color='white', bold=False):
    """Draw a rounded box with text."""
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle="round,pad=0.15", facecolor=color,
                          edgecolor='#333333', linewidth=1.2, alpha=0.95)
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            color=text_color, weight=weight)

def draw_phase_label(x, y, text, color):
    """Draw a phase label on the left side."""
    ax.text(x, y, text, ha='right', va='center', fontsize=15,
            color=color, weight='bold', family='sans-serif')

def draw_arrow(x1, y1, x2, y2, color=C_ARROW, style='simple', lw=1.5):
    """Draw an arrow between two points."""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                               connectionstyle='arc3,rad=0'))

def draw_connector(x1, y1, x2, y2, color=C_ARROW, lw=1.5, dashed=False):
    """Draw a line connector."""
    ls = '--' if dashed else '-'
    ax.plot([x1, x2], [y1, y2], color=color, lw=lw, linestyle=ls)

def draw_bracket(x, y, w, h, color, label='', alpha=0.08):
    """Draw a rounded rectangle bracket around a group."""
    rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle="round,pad=0.3", facecolor=color,
                          edgecolor=color, linewidth=2, alpha=alpha, linestyle='--')
    ax.add_patch(rect)
    if label:
        ax.text(x - w/2 + 0.3, y + h/2 - 0.4, label, fontsize=11,
                color=color, weight='bold', family='sans-serif', alpha=0.8)

CX = 14  # center x
# ── Row positions ──
Y_INPUT = 38.0
Y_SEARCH_DIRECT = 35.0
Y_GATE = 32.5
Y_HYPO_GEN = 29.5
Y_HYPO_SEARCH = 27.0
Y_MERGE = 25.0
Y_GAP = 23.0
Y_EXTRACT = 20.0
Y_TABLE = 18.0
Y_GENERATE = 16.0
Y_VERIFY_BATCH = 13.5
Y_VERIFY_FALLBACK = 11.5
Y_VERDICT = 9.5
Y_CORRECT = 7.5
Y_FB_GATE = 5.5
Y_FB_GUESS = 3.8
Y_FB_VERIFY = 2.2
Y_OUTPUT = 0.8

BOX_W = 11
BOX_H = 1.2
SMALL_W = 8
SMALL_H = 1.0

# ═══════════════════════════════════════════
# Phase labels
# ═══════════════════════════════════════════
draw_phase_label(1.5, Y_SEARCH_DIRECT, 'Phase 1\n搜索层', C_SEARCH)
draw_phase_label(1.5, Y_EXTRACT, 'Phase 2\n提取层', C_EXTRACT)
draw_phase_label(1.5, Y_GENERATE, 'Phase 3\n生成层', C_GENERATE)
draw_phase_label(1.5, Y_VERIFY_BATCH, 'Phase 4\n验证层', C_VERIFY)
draw_phase_label(1.5, Y_CORRECT, 'Phase 5\n修正层', C_CORRECT)
draw_phase_label(1.5, Y_FB_GATE, 'Phase 6\nFallback', C_FALLBACK)

# ═══════════════════════════════════════════
# Phase 1: Search
# ═══════════════════════════════════════════
draw_box(CX, Y_INPUT, BOX_W, BOX_H, '用户问题（谜语/多跳）', C_INPUT, bold=True, fontsize=14)
draw_arrow(CX, Y_INPUT - BOX_H/2, CX, Y_SEARCH_DIRECT + BOX_H/2)

draw_box(CX, Y_SEARCH_DIRECT, BOX_W, BOX_H, 'A) 谜面直接搜索 (SerpAPI + Jina)', C_SEARCH, fontsize=13)
draw_arrow(CX, Y_SEARCH_DIRECT - BOX_H/2, CX, Y_GATE + BOX_H/2)

draw_box(CX, Y_GATE, BOX_W, BOX_H, 'needs_hypothesis_search()?', C_SEARCH, fontsize=13)

# YES branch → hypothesis
draw_arrow(CX + BOX_W/2, Y_GATE, CX + BOX_W/2 + 1.5, Y_GATE, lw=1.2)
draw_arrow(CX + BOX_W/2 + 1.5, Y_GATE, CX + BOX_W/2 + 1.5, Y_HYPO_GEN + SMALL_H/2, lw=1.2)
ax.text(CX + BOX_W/2 + 2.0, (Y_GATE + Y_HYPO_GEN)/2, 'YES', fontsize=11, color=C_VERIFY, weight='bold')

draw_box(CX + 2.5, Y_HYPO_GEN, SMALL_W, SMALL_H, '假设生成: LLM猜5方向候选', '#2c5282', fontsize=11)
draw_arrow(CX + 2.5, Y_HYPO_GEN - SMALL_H/2, CX + 2.5, Y_HYPO_SEARCH + SMALL_H/2)

draw_box(CX + 2.5, Y_HYPO_SEARCH, SMALL_W, SMALL_H, '多路搜索验证 (每候选+约束词)', '#2c5282', fontsize=11)

# NO branch → straight down
ax.text(CX + 0.7, Y_GATE - 0.3, 'NO', fontsize=11, color='#38a169', weight='bold')
draw_connector(CX, Y_GATE - BOX_H/2, CX, Y_MERGE + BOX_H/2, lw=1.2)
draw_connector(CX + 2.5, Y_HYPO_SEARCH - SMALL_H/2, CX + 2.5, Y_MERGE + BOX_H/2, lw=1.2)
draw_connector(CX + 2.5, Y_MERGE + BOX_H/2, CX, Y_MERGE + BOX_H/2, lw=1.2)

draw_box(CX, Y_MERGE, BOX_W, BOX_H, '合并去重搜索结果', '#2a4365', fontsize=13)
draw_arrow(CX, Y_MERGE - BOX_H/2, CX, Y_GAP + BOX_H/2)

draw_box(CX, Y_GAP, BOX_W, BOX_H, 'B) Gap-Fill 迭代补充 (最多3轮)', '#2a4365', fontsize=13)
# Loop arrow
ax.annotate('', xy=(CX - BOX_W/2 - 0.3, Y_GAP - 0.2), xytext=(CX - BOX_W/2 - 0.3, Y_GAP + 0.2),
            arrowprops=dict(arrowstyle='->', color=C_VERIFY, lw=1.2,
                           connectionstyle='arc3,rad=-0.8'))
ax.text(CX - BOX_W/2 - 1.5, Y_GAP, '稳定?', fontsize=10, color=C_VERIFY, ha='center')

# Phase 1 bracket
draw_bracket(CX + 0.5, (Y_SEARCH_DIRECT + Y_GAP)/2, BOX_W + 5, Y_SEARCH_DIRECT - Y_GAP + 2.5, C_SEARCH, 'Phase 1: 搜索层', alpha=0.06)

# ═══════════════════════════════════════════
# Phase 2: SEVE Extraction
# ═══════════════════════════════════════════
draw_arrow(CX, Y_GAP - BOX_H/2, CX, Y_EXTRACT + BOX_H/2)
draw_box(CX, Y_EXTRACT, BOX_W, BOX_H, 'extract_claims() — ThreadPoolExecutor 并行', C_EXTRACT, fontsize=13)
draw_box(CX, Y_EXTRACT - 0.7, BOX_W - 1, 0.35, '每页独立提取 → 合并去重 (claim文本hash)', C_EXTRACT, fontsize=10, text_color='#a0c4ff')

draw_arrow(CX, Y_EXTRACT - BOX_H/2 - 0.5, CX, Y_TABLE + BOX_H/2)
draw_box(CX, Y_TABLE, BOX_W, BOX_H, '结构化知识表 [id] claim | snippet | url', C_EXTRACT, fontsize=13)

# ═══════════════════════════════════════════
# Phase 3: Generation
# ═══════════════════════════════════════════
draw_arrow(CX, Y_TABLE - BOX_H/2, CX, Y_GENERATE + BOX_H/2)
draw_box(CX, Y_GENERATE, BOX_W, BOX_H, 'generate_answer() — 知识表 → 带引用[1][2]', C_GENERATE, fontsize=13)

# ═══════════════════════════════════════════
# Phase 4: Verification
# ═══════════════════════════════════════════
draw_arrow(CX, Y_GENERATE - BOX_H/2, CX, Y_VERIFY_BATCH + BOX_H/2)
draw_box(CX, Y_VERIFY_BATCH, BOX_W, BOX_H, 'verify_claims() — 引用[1][2]反查原文 → 批量验证', C_VERIFY, fontsize=13)

draw_arrow(CX, Y_VERIFY_BATCH - BOX_H/2, CX, Y_VERIFY_FALLBACK + BOX_H/2)
draw_box(CX, Y_VERIFY_FALLBACK, BOX_W, BOX_H, 'UNKNOWN? → ThreadPoolExecutor(8) 并行逐条fallback', C_VERIFY, fontsize=13)

draw_arrow(CX, Y_VERIFY_FALLBACK - BOX_H/2, CX, Y_VERDICT + BOX_H/2)
draw_box(CX, Y_VERDICT, BOX_W, BOX_H, '判决:  YES  /  PARTIAL  /  NO', C_VERIFY, fontsize=13, bold=True)

# ═══════════════════════════════════════════
# Phase 5: Correction
# ═══════════════════════════════════════════
draw_arrow(CX, Y_VERDICT - BOX_H/2, CX, Y_CORRECT + BOX_H/2)
draw_box(CX, Y_CORRECT, BOX_W, BOX_H, 'apply_verification() — 删除NO / 标注PARTIAL', C_CORRECT, fontsize=13)

# ═══════════════════════════════════════════
# Phase 6: Fallback (Method E)
# ═══════════════════════════════════════════
draw_arrow(CX, Y_CORRECT - BOX_H/2, CX, Y_FB_GATE + BOX_H/2)

# Gate diamond
gate_w, gate_h = 7, 1.0
diamond = plt.Polygon([
    (CX, Y_FB_GATE + gate_h/2),
    (CX + gate_w/2, Y_FB_GATE),
    (CX, Y_FB_GATE - gate_h/2),
    (CX - gate_w/2, Y_FB_GATE)
], facecolor=C_FALLBACK, edgecolor='#333333', linewidth=1.2, alpha=0.9)
ax.add_patch(diamond)
ax.text(CX, Y_FB_GATE, '_is_failed_answer()?', ha='center', va='center',
        fontsize=13, color='white', weight='bold')

# NO → output
draw_arrow(CX + gate_w/2, Y_FB_GATE, CX + gate_w/2 + 1.2, Y_FB_GATE, lw=1.2)
draw_arrow(CX + gate_w/2 + 1.2, Y_FB_GATE, CX + gate_w/2 + 1.2, Y_OUTPUT + BOX_H/2, lw=1.2)
ax.text(CX + gate_w/2 + 1.5, (Y_FB_GATE + Y_OUTPUT)/2, 'NO', fontsize=11, color='#38a169', weight='bold')

# YES → fallback flow
draw_arrow(CX, Y_FB_GATE - gate_h/2, CX, Y_FB_GUESS + SMALL_H/2)

draw_box(CX, Y_FB_GUESS, SMALL_W, SMALL_H, '1. LLM直接猜答案（无搜索）', C_FALLBACK, fontsize=11)
draw_arrow(CX, Y_FB_GUESS - SMALL_H/2, CX, Y_FB_GUESS - 0.8)
draw_box(CX, Y_FB_GUESS - 1.4, SMALL_W, SMALL_H, '2. 拆解问题 → N个约束条件', C_FALLBACK, fontsize=11)

draw_arrow(CX, Y_FB_GUESS - 1.4 - SMALL_H/2, CX, Y_FB_VERIFY + SMALL_H/2)
draw_box(CX, Y_FB_VERIFY, SMALL_W, SMALL_H, '3. 逐约束验证: 答案满足约束?', C_FALLBACK, fontsize=11)

draw_arrow(CX, Y_FB_VERIFY - SMALL_H/2, CX, Y_OUTPUT + BOX_H/2)

# Fallback decision annotation
ax.text(CX - SMALL_W/2 - 1.2, Y_FB_VERIFY - 0.5, 'passed≥50%?\nyes→输出\nno→保留原答案',
        fontsize=10, color=C_FALLBACK, va='center')

# ═══════════════════════════════════════════
# Output
# ═══════════════════════════════════════════
draw_box(CX, Y_OUTPUT, BOX_W, BOX_H, '最终答案（带验证备注 / 推理链）', C_OUTPUT, bold=True, fontsize=13)

# ═══════════════════════════════════════════
# Method legend (bottom-right)
# ═══════════════════════════════════════════
legend_y = 0.6
legend_items = [
    ('A: Direct', '#a0a0a0'),
    ('B: +gap', '#718096'),
    ('C: +Hypo', '#2a4365'),
    ('D: SEVE (Phase 1-5)', C_VERIFY),
    ('E: +Fallback (Phase 1-6)', C_FALLBACK),
]
for i, (label, color) in enumerate(legend_items):
    lx = 14 + i * 1.8
    ax.add_patch(FancyBboxPatch((lx - 0.7, legend_y - 0.15), 1.4, 0.3,
                                boxstyle='round,pad=0.05',
                                facecolor=color, edgecolor='#333', linewidth=0.8))
    ax.text(lx, legend_y, label, fontsize=9, ha='center', color='#333333', weight='bold')

ax.text(14, legend_y + 0.5, '方法覆盖范围:', fontsize=10, color='#666666', weight='bold')

plt.tight_layout(pad=1)
plt.savefig('outputs/seve_pipeline.png', dpi=180, bbox_inches='tight',
            facecolor=C_BG, edgecolor='none')
plt.close()
print('Saved to outputs/seve_pipeline.png')
