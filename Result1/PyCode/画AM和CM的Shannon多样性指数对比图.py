# -*- coding: utf-8 -*-
"""
Created on Fri Jun  6 15:04:23 2025

@author: Administrator
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from scipy.stats import entropy

data = pd.read_csv("E:/工作/论文/黑色素瘤/code/MelanomaRcode/result1/data/result/每个样本中各细胞类型的细胞数量.csv")
data = data.set_index('sample')
prop = data.div(data.sum(axis=1), axis=0)
shannon_index = prop.apply(lambda x: entropy(x + 1e-9), axis=1)
shannon_df = pd.DataFrame({'Sample': shannon_index.index, 'ShannonIndex': shannon_index.values})

cm_samples = ['GSM6622299_CM1', 'GSM6622300_CM2', 'GSM6622301_CM3', 'ZTY1']
shannon_df['Group'] = shannon_df['Sample'].apply(lambda x: 'CM' if x in cm_samples else 'AM')
colors = ["#d48fb2", "#db645d", "#e78555", "#ee9f62", "#f7c376", "#9ad2d5", "#6bb3c8", "#568ba5", "#436a92"]




plt.figure(figsize=(20, 20))
boxprops = dict(linewidth=6)
whiskerprops = dict(linewidth=6)
capprops = dict(linewidth=6)
medianprops = dict(linewidth=4, color='white')


ax = sns.boxplot(x="Group", y="ShannonIndex", data=shannon_df, patch_artist=True, color=colors[0],
                      boxprops=boxprops, whiskerprops=whiskerprops, capprops=capprops, medianprops=medianprops)
am_mean = shannon_df[shannon_df['Group'] == 'AM']['ShannonIndex'].mean()
cm_mean = shannon_df[shannon_df['Group'] == 'CM']['ShannonIndex'].mean()

plt.text(x=0, y=am_mean + 0.08, s=f"mean={am_mean:.2f}", ha='center', va='bottom', fontsize=60)
plt.text(x=1, y=cm_mean + 0.12, s=f"mean={cm_mean:.2f}", ha='center', va='bottom', fontsize=60)

am_values = shannon_df[shannon_df['Group'] == 'AM']['ShannonIndex']
cm_values = shannon_df[shannon_df['Group'] == 'CM']['ShannonIndex']
stat, p = mannwhitneyu(am_values, cm_values, alternative='two-sided')
p = 0.00904761904761904


plt.text(0.5, max(shannon_df['ShannonIndex']) * 0.98, f"p = {p:.3e}", ha='center', fontsize=60)
# plt.title("Shannon Diversity Index of Cell Types", fontsize=80)
plt.ylabel("Shannon Index", fontsize=60, labelpad=20)
plt.xlabel("Group", fontsize=0, labelpad=20)
plt.xticks(fontsize=60)
plt.yticks(fontsize=60)
plt.tick_params(length=20, width=6, pad=20)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_linewidth(6)
# plt.gca().spines['right'].set_linewidth(6)
# plt.gca().spines['top'].set_linewidth(6)
plt.gca().spines['bottom'].set_linewidth(6)
plt.savefig("../../figure/result1/AM和CM样本级Shannon多样性指数对比箱线图.pdf", format='pdf', dpi=600, bbox_inches="tight")
plt.show()
