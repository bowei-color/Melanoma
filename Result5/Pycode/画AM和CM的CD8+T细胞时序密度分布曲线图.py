# -*- coding: utf-8 -*-
"""
Created on Sun Jun  8 15:23:25 2025

@author: Administrator
"""



import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("E:/工作/论文/黑色素瘤/code/MelanomaRcode/result5/data/result/AM和CMCD8+T细胞的时序密度分布曲线数据.csv")

plt.figure(figsize=(30, 20))
for grp, color in zip(['AM', 'CM'], ['#f8766d', '#00bfc4']):
    subset = data[data['Group'] == grp]
    sns.kdeplot(data=subset, x="Pseudotime", fill=True, label=grp, alpha=0.5, color=color, linewidth=4)
plt.xlabel("Pseudotime", fontsize=80, labelpad=20)
plt.ylabel("Density", fontsize=80, labelpad=20)
plt.xticks(fontsize=70)
plt.yticks(fontsize=70)
plt.tick_params(length=20, width=8, pad=20)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_linewidth(8)
plt.gca().spines['bottom'].set_linewidth(8)
plt.legend(title="",  fontsize=60, loc="upper left")
plt.savefig("../../figure/result5/AM和CM的CD8+T细胞时序密度分布曲线图.pdf", format="pdf", dpi=600, bbox_inches="tight")
plt.show()
