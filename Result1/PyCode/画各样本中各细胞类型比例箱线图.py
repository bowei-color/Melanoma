# -*- coding: utf-8 -*-
"""
Created on Fri Jun  6 14:17:43 2025

@author: Administrator
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("E:/工作/论文/黑色素瘤/code/MelanomaRcode/result1/data/result/每个样本中各细胞类型的细胞数量.csv")
data = data.set_index('sample')
data_prop = data.div(data.sum(axis=1), axis=0)



data_long = data_prop.reset_index().melt(id_vars='sample', var_name='CellType', value_name='Proportion')
colors = ["#d48fb2", "#db645d", "#e78555", "#ee9f62", "#f7c376", "#9ad2d5", "#6bb3c8", "#568ba5", "#436a92"]
celltypes = data_prop.index.tolist()




plt.figure(figsize=(20, 20))
boxprops = dict(linewidth=6)
whiskerprops = dict(linewidth=6)
capprops = dict(linewidth=6)
medianprops = dict(linewidth=2, color="white")

sns.boxplot(x='CellType', y='Proportion', data=data_long, patch_artist=True, color=colors[0],
                      boxprops=boxprops, whiskerprops=whiskerprops, capprops=capprops, medianprops=medianprops)
plt.xlabel('CellType', fontsize=0, labelpad=20)
plt.ylabel('Proportion', fontsize=60, labelpad=20)
plt.xticks(rotation=60, ha='right', fontsize=60)
plt.yticks(fontsize=60)
plt.tick_params(length=20, width=6, pad=20)
# plt.gca().spines['top'].set_visible(False)
# plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_linewidth(6)
plt.gca().spines['right'].set_linewidth(6)
plt.gca().spines['top'].set_linewidth(6)
plt.gca().spines['bottom'].set_linewidth(6)
plt.savefig("../../figure/result1/各样本中各细胞类型比例箱线图.pdf", format='pdf', dpi=600, bbox_inches='tight')
plt.show()