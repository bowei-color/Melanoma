# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 13:12:20 2025

@author: Administrator
"""

import pandas as pd
import matplotlib.pyplot as plt


desired_order = [
    'TCells', 'BCells', 'MPs', 'Keratinocytes', 'Melanocytes', 'Fibroblasts',
    'LECs', 'VascularECs', 'MastCells', 'SGCs', 'SMCs', 'SchwannCells', 'Adipocytes'
]

data = pd.read_csv("E:/工作/论文/黑色素瘤/code/MelanomaRcode/result1/data/result/每种细胞类型中每个分组的细胞数量.csv")
data = data.set_index('sample')
data = data.loc[desired_order]

# =============================================================================
# 画AM的数量柱状图
# =============================================================================

ax = plt.figure(figsize=(40, 30))
bars = plt.bar(data.index, data['AM'], color='#8e619f', edgecolor='black', lw=4, width=0.8)
plt.xlabel('AM', fontsize=110, labelpad=20)
plt.ylabel('Cell Number', fontsize=110, labelpad=20)
plt.xticks(rotation=60, fontsize=80, ha='right')
plt.yticks(fontsize=80)
plt.tick_params(length=30, width=15, pad=20)
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height, f'{int(height)}', ha='center', va='bottom', fontsize=70)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_linewidth(10)
plt.gca().spines['bottom'].set_linewidth(10)
plt.savefig('../../figure/result1/AM中每种细胞类型的数量柱状图.pdf', format='pdf', dpi=600, bbox_inches='tight')
plt.show()


# =============================================================================
# 画CM的数量柱状图
# =============================================================================

ax = plt.figure(figsize=(40, 30))
bars = plt.bar(data.index, data['CM'], color='#6e97b7', edgecolor='black', lw=4, width=0.8)
plt.xlabel('CM', fontsize=110, labelpad=20)
plt.ylabel('Cell Number', fontsize=110, labelpad=20)
plt.xticks(rotation=60, fontsize=80, ha='right')
plt.yticks(fontsize=80)
plt.tick_params(length=30, width=15, pad=20)
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height, f'{int(height)}', ha='center', va='bottom', fontsize=70)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_linewidth(10)
plt.gca().spines['bottom'].set_linewidth(10)
plt.savefig('../../figure/result1/CM中每种细胞类型的数量柱状图.pdf', format='pdf', dpi=600, bbox_inches='tight')
plt.show()
