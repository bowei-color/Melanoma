# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 12:20:18 2025

@author: Administrator
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import fcluster

ssgsea_data = pd.read_csv("E:/工作/论文/黑色素瘤/code/MelanomaRcode/result3/data/result/Adjusted_ssGSEA_Matrix.csv", index_col=0)
clinic_data = pd.read_csv("../../data/result6/source/GSE190113_clinical_data.csv")
clinic_data.insert(0, "Sample", ["Sample" + str(i + 1) for i in range(len(clinic_data))])
clinic_data_filter = clinic_data[["Sample", "sample", "group"]]

ssgsea_data_new = ssgsea_data.T
ssgsea_data_new = ssgsea_data_new.reset_index()
ssgsea_data_new = ssgsea_data_new.rename(columns={"index": "Sample"})

am_clinic = clinic_data_filter[clinic_data_filter["group"]=="CM"]

merged_data = am_clinic.merge(ssgsea_data_new, on="Sample", how="inner")

merged_data = merged_data.drop(columns=["group", "Sample"])


sample_ids = merged_data["sample"]
data_only = merged_data.drop(columns=["sample"]).iloc[:, 4:]

data_scaled = data_only.sub(data_only.mean(axis=1), axis=0).div(data_only.std(axis=1), axis=0)



data_scaled.index = sample_ids
# data_scaled.to_csv("../../data/result6/result/CM按四个功能特征聚类后的按行归一化结果数据.csv")
# data_scaled = pd.read_csv("../../data/result6/result/CM按四个功能特征聚类后的按行归一化结果数据.csv", index_col=0)


sns.set(font_scale=0.8)
g = sns.clustermap(data_scaled,
                   method='ward',
                   metric='euclidean',
                   row_cluster=True,
                   col_cluster=False,
                   cmap="vlag",
                   figsize=(16, 10),
                   xticklabels=True,
                   yticklabels=False,
                   dendrogram_ratio=(0.2, 0),
                   cbar_pos=(0.02, 0.8, 0.03, 0.18))


linkage_matrix = g.dendrogram_row.linkage
cluster_labels = pd.Series(fcluster(linkage_matrix, t=2, criterion='maxclust'), index=data_scaled.index)
cluster_labels.to_csv("../../data/result6/result/CM四个功能特征聚类结果.csv")
row_colors = cluster_labels.map({1: "#e7c161", 2: "#3c87b2"})



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

g.savefig("../../figure/result6/基于黑色素细胞功能的结果图/CM按四个功能特征评分聚类结果图.pdf")








