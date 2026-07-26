# -*- coding: utf-8 -*-
"""
Created on Fri Jun  6 17:09:26 2025

@author: Administrator
"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from matplotlib.patches import FancyArrowPatch

file_path = "E:/工作/论文/黑色素瘤/code/MelanomaRcode/result2/data/result/各样本中各黑色素细胞功能特征占比.csv"
data = pd.read_csv(file_path)
data['sample'] = data['sample'].str.replace(r'^.*?_', '', regex=True)
data['subgroup'] = data['subgroup'].str.replace('Subgroup', 'Signature')

data_pivot = data.pivot_table(index='sample', columns='subgroup', values='percentage', aggfunc='sum').fillna(0)
data_pivot = data_pivot/100

fig, ax = plt.subplots(figsize=(12, 12))
bar_width = 0.7
group_gap = 0.5
samples = data['sample'].unique()
sample_groups = data.groupby('group')
group_positions = []
current_position = 0
for group, group_data in sample_groups:
    group_samples = group_data['sample'].unique()
    group_size = len(group_samples)
    group_positions.extend(np.arange(current_position, current_position + group_size) * (bar_width + group_gap))
    current_position += group_size * (bar_width + group_gap) + group_gap


sample_order = []
for group, group_data in sample_groups:
    sample_order.extend(group_data['sample'].unique())

bottoms = np.zeros(len(samples))
colors = ['grey', '#cd161d', '#ff840e', '#1e6bae', '#2ca031']
# colors = cm.viridis(np.linspace(0, 1, len(data_pivot.columns)))
for i, column in enumerate(data_pivot.columns):
    ax.bar(group_positions, data_pivot.loc[sample_order, column], bar_width, bottom=bottoms, label=column, color=colors[i], edgecolor='black', linewidth=2)
    bottoms += data_pivot.loc[sample_order, column].values

plt.tick_params(length=15, width=4)
plt.yticks(fontsize=30)
ax.set_xticks(group_positions)
ax.set_xticklabels(sample_order, rotation=45, fontsize=30)
ax.set_xlabel('', fontsize=12)
ax.set_ylabel('Ratio of Signatures', fontsize=40, labelpad=20)
ax.set_ylim(-0.05, 1.05)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(4)
ax.spines['bottom'].set_linewidth(4)

y_position = 1.02


line_colors = ['#cd161d', '#1e6bae'] 
line_widths = [5] 

for i, (group, group_data) in enumerate(sample_groups):
    group_sample_positions = [pos for pos, sample in zip(group_positions, sample_order) if sample in group_data['sample'].values]
    group_mean_position = np.mean(group_sample_positions)

    line_color = line_colors[i % len(line_colors)] 
    line_width = line_widths[i % len(line_widths)]
    if len(group_sample_positions) == 1:
        x_start = group_sample_positions[0] - bar_width / 2
        x_end = group_sample_positions[0] + bar_width / 2
    else:
        x_start = group_sample_positions[0] - bar_width / 2
        x_end = group_sample_positions[-1] + bar_width / 2

    ax.plot([x_start, x_end], [y_position] * 2, color=line_color, linewidth=line_width)
    ax.text(group_mean_position, y_position + 0.02, group, horizontalalignment='center', fontsize=30)

ax.legend(title='Signature', title_fontsize=30, bbox_to_anchor=(1.02, 0.9), loc='upper left', fontsize=25)
plt.savefig('../../figure/result2/黑色素细胞中各功能特征占比.pdf', format='pdf', bbox_inches='tight', dpi=600)
plt.show()

