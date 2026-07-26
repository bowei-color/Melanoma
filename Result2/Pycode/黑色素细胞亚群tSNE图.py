# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 15:25:41 2025

@author: Administrator
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("E:/工作/论文/黑色素瘤/code/MelanomaRcode/result2/data/result/黑色素细胞亚群细分tsne数据.csv", index_col=0)
df["seurat_clusters"] = df["seurat_clusters"].astype(str)
df["sample"] = df["sample"].str.replace(r"^.*?_", "", regex=True)





# =============================================================================
# 按簇着色
# =============================================================================
df["seurat_clusters"] = pd.Categorical(df["seurat_clusters"], 
                                       categories=sorted(df["seurat_clusters"].unique(), key=lambda x: int(x)),
                                       ordered=True)

plt.figure(figsize=(20, 24))
sns.scatterplot(data=df, x="tSNE_1", y="tSNE_2", hue="seurat_clusters", s=200, linewidth=1)
plt.xlabel("tSNE_1",fontsize=80, labelpad=20)
plt.ylabel("tSNE_2", fontsize=80)
plt.xticks([-40, -20, 0, 20, 40], fontsize=80)
plt.yticks(fontsize=80)
plt.tick_params(length=20, width=8, pad=20)
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.gca().spines["left"].set_linewidth(8)
plt.gca().spines["bottom"].set_linewidth(8)
plt.legend(loc="upper right", title="cluster", title_fontsize=80, fontsize=60,labelspacing=0.1, markerscale=3, handlelength=1, handleheight=1, bbox_to_anchor=(1.35, 1.1))
plt.savefig("../../figure/result2/黑色素瘤亚群按照簇划分tSNE图.pdf", format='pdf', dpi=600, bbox_inches='tight')
plt.show()




# =============================================================================
# 按样本着色
# =============================================================================


sample_colors = {
  "CM1": "#1e6bae",
  "CM2": "#2ca031",
  "CM3": "#9767bd",
  "ZTY1": "#ff840e",
  "WBZ1": "#8c574b",
  "FGQI": "#ffe000",
  "HYM1": "#e377bc",
  "SL1": "#7f7f7f",
  "LY1": "#cd161d"
}
df["sample"].unique().tolist()

plt.figure(figsize=(20, 24))
sns.scatterplot(data=df, x="tSNE_1", y="tSNE_2",  palette=sample_colors, hue="sample", s=200, linewidth=1)
plt.xlabel("tSNE_1",fontsize=80, labelpad=20)
plt.ylabel("tSNE_2", fontsize=80, labelpad=20)
plt.xticks([-40, -20, 0, 20, 40], fontsize=80)
plt.yticks(fontsize=80)
plt.tick_params(length=20, width=8, pad=20)
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.gca().spines["left"].set_linewidth(8)
plt.gca().spines["bottom"].set_linewidth(8)
plt.legend(loc="upper right", title="sample", title_fontsize=80, fontsize=80,labelspacing=0.1, markerscale=3,  handlelength=1, handleheight=1, bbox_to_anchor=(1.35, 0.8))
plt.savefig("../../figure/result2/黑色素瘤亚群按照样本划分tSNE图.pdf", format='pdf', dpi=600, bbox_inches='tight')
plt.show()



# =============================================================================
# 按分组着色
# =============================================================================

group_colors = { "AM": "#cd161d", "CM":"#1e6bae"}

plt.figure(figsize=(20, 24))
sns.scatterplot(data=df, x="tSNE_1", y="tSNE_2", palette=group_colors, hue="group", s=200, linewidth=1)
plt.xlabel("tSNE_1",fontsize=80, labelpad=20)
plt.ylabel("tSNE_2", fontsize=80, labelpad=20)
plt.xticks([-40, -20, 0, 20, 40], fontsize=80)
plt.yticks(fontsize=80)
plt.tick_params(length=20, width=8, pad=20)
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.gca().spines["left"].set_linewidth(8)
plt.gca().spines["bottom"].set_linewidth(8)
plt.legend(loc="upper right", title="group", title_fontsize=80, fontsize=80,labelspacing=0.1, markerscale=3,  handlelength=1, handleheight=1, bbox_to_anchor=(1.4, 0.8))
plt.savefig("../../figure/result2/黑色素瘤亚群按照分组划分tSNE图.pdf", format='pdf', dpi=600, bbox_inches='tight')
plt.show()


# =============================================================================
# 按功能亚组着色
# =============================================================================

group_colors = { 
    'Subgroup1': '#cd161d', 
    'Subgroup2': '#ff840e', 
    'Subgroup3': '#1e6bae', 
    'Subgroup4': '#2ca031', 
    'Others': 'grey'
    }

subgroup_order = list(group_colors.keys())

plt.figure(figsize=(20, 24))
sns.scatterplot(data=df, x="tSNE_1", y="tSNE_2", palette=group_colors, hue="subgroup", hue_order=subgroup_order, s=200, linewidth=1)
plt.xlabel("tSNE_1",fontsize=80, labelpad=20) 
plt.ylabel("tSNE_2", fontsize=80, labelpad=20)
plt.xticks([-40, -20, 0, 20, 40], fontsize=80)
plt.yticks(fontsize=80)
plt.tick_params(length=20, width=8, pad=20)
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.gca().spines["left"].set_linewidth(8)
plt.gca().spines["bottom"].set_linewidth(8)
plt.legend(loc="upper right", title="subgroup", title_fontsize=80, fontsize=80,labelspacing=0.1, markerscale=3,  handlelength=1, handleheight=1, bbox_to_anchor=(1.6, 0.8))
plt.savefig("../../figure/result2/黑色素瘤亚群按照功能亚组划分tSNE图.pdf", format='pdf', dpi=600, bbox_inches='tight')
plt.show()