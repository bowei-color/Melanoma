# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 16:09:58 2025

@author: Administrator
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns




colors = ["#f8766d", "#00bfc4"]


# =============================================================================
# 通顺数量柱状图
# =============================================================================
interaction_count = pd.read_csv("E:/工作/论文/黑色素瘤/code/MelanomaRcode/result5/data/result/AM和CM中通讯事件数量统计.csv")


plt.figure(figsize=(10, 20))
sns.barplot(data=interaction_count, x="group", y="interaction_count", palette=colors)
for i,row in interaction_count.iterrows():
    plt.text(i, row["interaction_count"]+2, f"{row['interaction_count']}", ha="center", fontsize=70)
plt.xlabel("", fontsize=0)
plt.ylabel("Interaction Count", fontsize=70, labelpad=20)
plt.xticks(fontsize=70)
plt.yticks(fontsize=70)
plt.tick_params(length=15, width=6, pad=20)
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.gca().spines["left"].set_linewidth(6)
plt.gca().spines["bottom"].set_linewidth(6)
plt.savefig("../../figure/result5/T细胞细分亚型细胞之间通讯事件数量柱状图.pdf", format="pdf", dpi=600, bbox_inches="tight")
plt.show()



# =============================================================================
# 通顺强度柱状图
# =============================================================================


interaction_strength = pd.read_csv("E:/工作/论文/黑色素瘤/code/MelanomaRcode/result5/data/result/AM和CM中通讯强度统计.csv")

plt.figure(figsize=(10, 20))
sns.barplot(data=interaction_strength, x="group", y="interaction_strength", palette=colors)
for i, row in interaction_strength.iterrows():
    plt.text(i, row["interaction_strength"] + 0.2, f"{row['interaction_strength']:.3f}", ha='center', fontsize=70)
plt.xlabel("", fontsize=0)
plt.ylabel("Interaction Strength", fontsize=70, labelpad=20)
plt.xticks(fontsize=70)
plt.yticks(fontsize=70)
plt.tick_params(length=15, width=6, pad=20)
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.gca().spines["left"].set_linewidth(6)
plt.gca().spines["bottom"].set_linewidth(6)
plt.savefig("../../figure/result5/T细胞细分亚型细胞之间通讯强度柱状图.pdf", format="pdf", dpi=600, bbox_inches="tight")
plt.show()
