# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 14:13:42 2025

@author: Administrator
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("../../data/result1/source/画肢端和皮肤黑色素瘤UMAP的数据.csv")


celltype_palette = {
    "TCells": "#e44a34",
    "BCells": "#48b9d3",
    "MPs": "#1f9f85",
    "Keratinocytes": "#375287",
    "Melanocytes": "#ef9a7e",
    "Fibroblasts": "#8490b3",
    "LECs": "#92cebf",
    "VascularECs": "#db0a17",
    "MastCells": "#7c5f45",
    "SGCs": "#cccccc",
    "SMCs": "#ba8a83",
    "SchwannCells": "#24abac",
    "Adipocytes": "#2e7987"
}



desired_order = [
    "TCells", "BCells", "MPs", "Keratinocytes", "Melanocytes",
    "Fibroblasts", "LECs", "VascularECs", "MastCells", "SGCs",
    "SMCs", "SchwannCells", "Adipocytes"
]

df["celltype"] = pd.Categorical(df["celltype"], categories=desired_order, ordered=True)


plt.figure(figsize=(20, 24))
sns.scatterplot(data=df, x="UMAP_1", y="UMAP_2", hue="celltype", palette=celltype_palette, s=15, alpha=1, linewidth=0)
plt.xlabel("UMAP_1", fontsize=80, labelpad=20)
plt.ylabel("UMAP_2", fontsize=80)
plt.xticks(fontsize=80)
plt.yticks(fontsize=80)
plt.tick_params(length=20, width=8, pad=20)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(8)
ax.spines['bottom'].set_linewidth(8)

plt.legend(bbox_to_anchor=(1.0, 0.95), loc="upper left", fontsize=60, frameon=False, markerscale=10)
plt.savefig('../../figure/result1/肢端和皮肤黑色素瘤的Umap图.pdf', format='pdf', dpi=600, bbox_inches='tight')
plt.show()


df[df['celltype']=='TCells']
