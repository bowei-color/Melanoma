# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 20:02:02 2025

@author: Administrator
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

data = pd.read_csv("E:/工作/论文/黑色素瘤/code/MelanomaRcode/result1/data/result/每个样本中各细胞类型的细胞数量.csv")
data["sample"] = data["sample"].str.replace(r"^.*?_", "", regex=True)
data = data.set_index('sample')
data_prop = data.div(data.sum(axis=1), axis=0)

am_samples = ['WBZ1', 'FGQI', 'HYM1', 'LY1', 'SL1']

cm_samples = [s for s in data.index if s not in am_samples]
ordered_samples = am_samples + cm_samples
data_prop = data_prop.loc[ordered_samples]

col_colors = pd.Series(ordered_samples, index=ordered_samples).map(lambda x: '#8e619f' if x in am_samples else '#6e97b7')
plot_matrix = data_prop.transpose()
cmap_custom = LinearSegmentedColormap.from_list("blue_white_red", ["#2166ac", "white", "#b2182b"])

plt.figure(figsize=(20, 20))
sns.set(font_scale=1.2)
g = sns.clustermap(plot_matrix,  col_colors=col_colors, row_cluster=False, col_cluster=False, cmap=cmap_custom, linewidths=0.4,  
                   xticklabels=True, yticklabels=True, figsize=(20, 20), cbar_pos=(0.95, 0.04, 0.05, 0.73))
g.ax_heatmap.yaxis.tick_left()
g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0)
g.ax_heatmap.tick_params(length=0, labelsize=60)
g.ax_heatmap.tick_params(axis="x", rotation=45)
g.ax_heatmap.set_xlabel("")
g.cax.tick_params(length=20, width=6, labelsize=60)
g.ax_col_dendrogram.text(0.27, 0.1, 'AM', color='#6baed6', fontsize=60, transform=g.ax_col_dendrogram.transAxes)
g.ax_col_dendrogram.text(0.765, 0.1, 'CM', color='#fd8d3c', fontsize=60, transform=g.ax_col_dendrogram.transAxes)

# plt.savefig("../../figure/result1/AM和CM中各样本细胞类型比例热图.pdf", format='pdf', dpi=600, bbox_inches='tight')
plt.show()



