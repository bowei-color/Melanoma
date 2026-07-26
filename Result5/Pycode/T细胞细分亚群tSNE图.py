# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 15:25:41 2025

@author: Administrator
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("E:/工作/论文/黑色素瘤/code/MelanomaRcode/result5/data/result/T细胞亚群细分tsne数据.csv", index_col=0)
df["seurat_clusters"] = df["seurat_clusters"].astype(str)
df["sample"] = df["sample"].str.replace(r"^.*?_", "", regex=True)




celltype_colors = {
  "Naive CD8+ T": "#1e6bae",
  "Cytotoxic CD8+ T": "#ff840e",
  "Exhausted CD8+ T": "#8c574b",
  "NK-like": "#9767bd",
  "CD4Tmem": "#2ca031",
  "Th17": "#e377bc",
  "Treg": "#cd161d"
}

subtype_order = list(celltype_colors.keys())

df["subtype"].unique().tolist()

plt.figure(figsize=(12, 20))
sns.scatterplot(data=df, x="tSNE_1", y="tSNE_2", palette=celltype_colors, hue="subtype", hue_order=subtype_order, s=50, linewidth=0.5)
plt.xlabel("tSNE_1",fontsize=60, labelpad=20)
plt.ylabel("tSNE_2", fontsize=60, labelpad=20)
plt.xticks([-40, -20, 0, 20, 40], fontsize=60)
plt.yticks(fontsize=60)
plt.tick_params(length=20, width=8, pad=20)
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.gca().spines["left"].set_linewidth(8)
plt.gca().spines["bottom"].set_linewidth(8)
plt.legend(loc="upper right", title="celltype", title_fontsize=0, fontsize=50, frameon=False, labelspacing=0.1, markerscale=3,  handlelength=1, handleheight=1, bbox_to_anchor=(1.9, 0.75))
# plt.savefig("../../figure/result5/T细胞细分亚群细胞类型注释tSNE图.pdf", format='pdf', dpi=600, bbox_inches='tight')
plt.show()

