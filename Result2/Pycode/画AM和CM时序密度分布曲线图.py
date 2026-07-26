# -*- coding: utf-8 -*-
"""
Created on Sun Jun  8 15:23:25 2025

@author: Administrator
"""



import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("E:/工作/论文/黑色素瘤/code/MelanomaRcode/result2/data/result/AM和CM时序密度分布曲线数据.csv")

plt.figure(figsize=(20, 20))
for grp, color in zip(['AM', 'CM'], ['#cd161d', '#ff7f0e']):
    subset = data[data['group'] == grp]
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
plt.legend(title="",  fontsize=60, loc="upper right")
plt.savefig("../../figure/result2/AM和CM时序密度分布曲线图.pdf", format="pdf", dpi=600, bbox_inches="tight")
plt.show()
