# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 11:53:39 2025

@author: Administrator
"""

import pandas as pd
import matplotlib.pyplot as plt

desired_order = [
    "TCells", "BCells", "MPs", "Keratinocytes", "Melanocytes", "Fibroblasts",
    "LECs", "VascularECs", "MastCells", "SGCs", "SMCs", "SchwannCells", "Adipocytes"
]
data = pd.read_csv("E:/工作/论文/黑色素瘤/code/MelanomaRcode/result1/data/result/每种细胞类型中每个分组的细胞数量.csv")
data = data.set_index("sample")
data = data.loc[desired_order]
data_prop = data.div(data.sum(axis=1), axis=0)
groups = data_prop.columns.tolist()
group_colors = {
    'AM': '#9665a8',
    'CM': '#719dbf'
}

colors = [group_colors.get(group, '#cccccc') for group in groups]
celltypes = data_prop.index.tolist()

ax = data_prop.plot(kind='bar', stacked=True, color=colors, width=0.8, figsize=(40, 50))
ax.set_xticklabels([])
xticks = ax.get_xticks()
for i, label in enumerate(celltypes):
    ax.text(xticks[i] + 0.4, -0.02, label, ha='right', va='top', rotation=60, fontsize=150, transform=ax.transData, clip_on=False)
ax.set_ylabel('')

plt.yticks(fontsize=120)
plt.tick_params(length=60, width=16, pad=20)
ax.spines['top'].set_linewidth(20)
ax.spines['right'].set_linewidth(20)
ax.spines['left'].set_linewidth(20)
ax.spines['bottom'].set_linewidth(20)
plt.legend(loc='upper left', fontsize=120, frameon=False, borderaxespad=0, handlelength=1.5, handleheight=2, labelspacing=0.1, bbox_to_anchor=(1.01, 0.6))
plt.savefig("../../figure/result1/每种细胞类型中各分组的细胞数量占比.pdf", format='pdf', dpi=600, bbox_inches='tight')
plt.show()