# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 15:46:43 2025

@author: Administrator
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

ssgsea_data = pd.read_csv("E:/工作/论文/黑色素瘤/code/MelanomaRcode/result3/data/result/Adjusted_ssGSEA_Matrix.csv", index_col=0)
clinic_data = pd.read_csv("../../data/result6/source/GSE190113_clinical_data.csv")
clinic_data.insert(0, "Sample", ["Sample" + str(i + 1) for i in range(len(clinic_data))])
clinic_data_filter = clinic_data[["Sample", "sample", "group"]]

ssgsea_data_new = ssgsea_data.T
ssgsea_data_new = ssgsea_data_new.reset_index()
ssgsea_data_new = ssgsea_data_new.rename(columns={"index": "Sample"})

merged_data = clinic_data_filter.merge(ssgsea_data_new, on="Sample", how="inner")

sample_ids = merged_data["sample"]
group_labels = merged_data["group"]
data_only = merged_data.drop(columns=["group", "Sample", "sample"]).iloc[:, 4:]

data_scaled = data_only.sub(data_only.mean(axis=1), axis=0).div(data_only.std(axis=1), axis=0)
data_scaled.index = sample_ids

group_colors = group_labels.map({"AM": "#e7c161", "CM": "#3c87b2"})
row_colors = pd.Series(group_colors.values, index=sample_ids)

col_order = data_scaled.columns
col_annotation = []
for i in range(len(col_order)):
    if i == 0:
        col_annotation.append("#cc151f")
    elif 1 <= i <= 5:
        col_annotation.append("#f08223")
    elif 6 <= i <= 10:
        col_annotation.append("#176aac")
    elif 11 <= i <= 15:
        col_annotation.append("#309f3a")
col_colors = pd.Series(col_annotation, index=col_order)

sns.set(font_scale=5)
g = sns.clustermap(data_scaled,
                   linewidths=2,
                   method='ward',
                   metric='euclidean',
                   row_cluster=True,
                   col_cluster=False,
                   row_colors=row_colors,
                   col_colors=col_colors,
                   cmap="vlag",
                   figsize=(20, 16),
                   xticklabels=False,
                   yticklabels=False,
                   dendrogram_ratio=(0.2, 0),
                   cbar_pos=(0.92, 0.06, 0.04, 0.85))
g.ax_heatmap.set_ylabel("")
for spine in g.ax_row_dendrogram.spines.values():
    spine.set_linewidth(4)
for line in g.ax_row_dendrogram.collections:
    line.set_linewidth(4)
plt.show()

g.savefig("../../figure/result6/基于黑色素细胞功能的结果图/GSE190113数据按四个功能特征评分聚类结果图.pdf")

