# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 21:53:55 2025

@author: Administrator
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# =============================================================================
# 各CD8+T细胞占比
# =============================================================================


data = pd.read_csv("E:/工作/论文/黑色素瘤/code/MelanomaRcode/result5/data/result/CD8_T细胞亚群中_AM_CM占比统计.csv")

pivot_df = data.pivot(index='group', columns='subtype', values='ratio')
subtype_order = ["Naive CD8+ T", "Cytotoxic CD8+ T", "Exhausted CD8+ T"]
colors = ["#1e6bae", "#ff840e", "#8c574b"]
pivot_df = pivot_df[subtype_order]
groups = data["group"].unique().tolist()

ax = pivot_df.plot(kind="bar", stacked=True, color=colors, figsize=(20, 60), width=0.6)

plt.xticks(rotation=0, fontsize=200)
plt.yticks([], fontsize=0)
plt.tick_params(length=60, width=16, pad=20)
ax.spines['top'].set_linewidth(20)
ax.spines['right'].set_linewidth(20)
ax.spines['left'].set_linewidth(20)
ax.spines['bottom'].set_linewidth(20)
# plt.legend(loc='upper left', fontsize=160, frameon=False, borderaxespad=0, handlelength=1.5, handleheight=2, labelspacing=0.1, bbox_to_anchor=(1.01, 0.6))
plt.legend(fontsize=0)
plt.savefig("../../figure/result5/AM和CM中CD8+T细胞数量占比.pdf", format='pdf', dpi=600, bbox_inches='tight')
plt.show()


# =============================================================================
# T细胞亚群占比
# =============================================================================

# all_data = pd.read_csv("E:/工作/论文/黑色素瘤/code/MelanomaRcode/result5/data/result/T细胞细分亚群中AM_CM占比统计.csv")

# all_pivot_df = all_data.pivot(index='group', columns='subtype', values='ratio')
# subtype_order = ["CD8+ T", "CD4Tmem", "Treg", "Th17", "NK-like"]
# colors = ["#619cff", "#2ca031", "#cd161d", "#e377bc", "#9767bd"]


# all_pivot_df = all_pivot_df[subtype_order]
# groups = all_data["group"].unique().tolist()

# ax = all_pivot_df.plot(kind="bar", stacked=True, color=colors, figsize=(20, 60), width=0.6)
# plt.xticks(rotation=0, fontsize=200)
# plt.yticks(fontsize=200)
# plt.tick_params(length=60, width=16, pad=20)
# ax.spines['top'].set_linewidth(20)
# ax.spines['right'].set_linewidth(20)
# ax.spines['left'].set_linewidth(20)
# ax.spines['bottom'].set_linewidth(20)
# plt.legend(loc='upper left', fontsize=160, frameon=False, borderaxespad=0, handlelength=1.5, handleheight=2, labelspacing=0.1, bbox_to_anchor=(1.01, 0.7))
# plt.savefig("../../figure/result5/AM和CM中T细胞细分亚群数量占比.pdf", format='pdf', dpi=600, bbox_inches='tight')
# plt.show()





all_data = pd.read_csv("E:/工作/论文/黑色素瘤/code/MelanomaRcode/result5/data/result/T细胞细分亚群中AM_CM占比统计.csv")

all_pivot_df = all_data.pivot(index='group', columns='subtype', values='ratio')
subtype_order = ["CD8+ T", "CD4Tmem", "Treg", "Th17", "NK-like"]
colors = ["#00bfc4", "#2ca031", "#cd161d", "#e377bc", "#9767bd"]
all_pivot_df = all_pivot_df[subtype_order]


fig, ax = plt.subplots(figsize=(20, 60))
all_pivot_df.plot(kind="bar", stacked=True, color=colors, ax=ax, width=0.6)

plt.xticks(rotation=0, fontsize=160)
plt.yticks(fontsize=200)
plt.tick_params(length=60, width=16, pad=20)
for spine in ['top', 'right', 'left', 'bottom']:
    ax.spines[spine].set_linewidth(20)


tcell_legend_labels = subtype_order
tcell_patches = [Patch(facecolor=c, label=l) for c, l in zip(colors, tcell_legend_labels)]
legend1 = ax.legend(handles=tcell_patches,
                    loc='upper left',
                    fontsize=160,
                    frameon=False,
                    borderaxespad=0,
                    handlelength=1.5,
                    handleheight=2,
                    labelspacing=0.1,
                    bbox_to_anchor=(1.01, 0.5))
ax.add_artist(legend1)


cd8_labels = ["Naive CD8+ T", "Cytotoxic CD8+ T", "Exhausted CD8+ T"]
cd8_colors = ["#1e6bae", "#ff840e", "#8c574b"]
cd8_patches = [Patch(facecolor=c, label=l) for c, l in zip(cd8_colors, cd8_labels)]
legend2 = ax.legend(handles=cd8_patches,
                    loc='upper left',
                    fontsize=160,
                    frameon=False,
                    borderaxespad=0,
                    handlelength=1.5,
                    handleheight=2,
                    labelspacing=0.1,
                    bbox_to_anchor=(1.01, 0.8))


plt.savefig("../../figure/result5/AM和CM中T细胞细分亚群数量占比.pdf", format='pdf', dpi=600, bbox_inches='tight')
plt.show()

