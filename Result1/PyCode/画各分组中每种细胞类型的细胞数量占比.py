# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 12:39:50 2025

@author: Administrator
"""

import pandas as pd
import matplotlib.pyplot as plt

desired_celltype_order = [
    'TCells', 'BCells', 'MPs', 'Keratinocytes', 'Melanocytes', 'Fibroblasts',
    'LECs', 'VascularECs', 'MastCells', 'SGCs', 'SMCs', 'SchwannCells', 'Adipocytes'
]
data = pd.read_csv("E:/工作/论文/黑色素瘤/code/MelanomaRcode/result1/data/result/每个分组中各细胞类型的细胞数量.csv")
data = data.set_index('sample')
data = data[desired_celltype_order]
celltypes = data.columns.tolist()
groups = data.index.tolist()

celltype_colors = {
    'TCells': '#652463',
    'BCells': '#936ba6',
    'MPs': '#df568d',
    'Keratinocytes': '#7a7592',
    'Melanocytes': '#dbc2dc',
    'Fibroblasts': '#d2a4c5',
    'LECs': '#df84b3',
    'VascularECs': '#9392c1',
    'MastCells': '#bfb76e',
    'SGCs': '#acbb7b',
    'SMCs': '#79b688',
    'SchwannCells': '#1d867e',
    'Adipocytes': '#2a86c5',
    }


colors = [celltype_colors.get(sample, '#cccccc') for sample in celltypes]
data_prop = data.div(data.sum(axis=1), axis=0)
data_prop.to_csv("../../data/result1/result/AM和CM中各细胞类型的数量占比.csv")
ax = data_prop.plot(kind='bar', stacked=True, color=colors, width=0.8, figsize=(20, 40))
plt.xticks(rotation=0, fontsize=100)
plt.yticks(fontsize=100)
plt.tick_params(length=40, width=16, pad=20)
plt.ylabel('')
plt.xlabel('')
ax.spines['top'].set_linewidth(10)
ax.spines['right'].set_linewidth(10)
ax.spines['left'].set_linewidth(10)
ax.spines['bottom'].set_linewidth(10)
plt.legend(bbox_to_anchor=(1.1, 0.93), loc='upper left', frameon=False, borderaxespad=0, handlelength=1.2, handleheight=1.7, labelspacing=0.1,  fontsize=100)
# plt.savefig("../../figure/result1/各分组中每种细胞类型的细胞数量占比.pdf", format='pdf', dpi=600, bbox_inches='tight')
plt.show()