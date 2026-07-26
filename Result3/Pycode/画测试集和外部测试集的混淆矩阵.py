# -*- coding: utf-8 -*-
"""
Created on Wed May 28 18:35:24 2025

@author: Administrator
"""




import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

methods = "LR"
colors = ['#f7fbff', '#f5cea3']

# =============================================================================
# 测试集
# =============================================================================

cm = pd.read_csv(f"../../data/result3/confunsion_matrix/{methods}_confusion_matrix.csv", index_col=0).values
labels = ['AM', 'CM']

cm_row_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)

custom_cmap = LinearSegmentedColormap.from_list("custom", colors)


plt.figure(figsize=(20, 18))
ax = sns.heatmap(cm_row_normalized,
                 annot=cm,
                 fmt='d',
                 cmap=custom_cmap,
                 xticklabels=labels,
                 yticklabels=labels,
                 vmin=0, vmax=1,
                 cbar=True,
                 annot_kws={'fontsize': 70})

colorbar = ax.collections[0].colorbar
colorbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
colorbar.set_ticklabels(['0.0', '0.25', '0.5', '0.75', '1.0'])
colorbar.ax.tick_params(labelsize=60)
plt.xlabel('Predicted Label', fontsize=80, labelpad=40)
plt.ylabel('True Label', fontsize=80, labelpad=40)
plt.tick_params(length=30, width=8, pad=20)
plt.xticks(fontsize=70)
plt.yticks(fontsize=70)
plt.savefig(f"../../figure/result3/confusion_matrix/{methods}方法在测试集上的混淆矩阵.pdf", format="pdf", dpi=600, bbox_inches='tight')
plt.show()



# =============================================================================
# 外部测试集
# =============================================================================

cm = pd.read_csv(f"../../data/result3/confunsion_matrix_external/{methods}_confusion_matrix_external.csv", index_col=0).values
labels = ['AM', 'CM']

cm_row_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)

custom_cmap = LinearSegmentedColormap.from_list("custom", colors)

plt.figure(figsize=(20, 18))
ax = sns.heatmap(cm_row_normalized,
                  annot=cm,
                  fmt='d',
                  cmap=custom_cmap,
                  xticklabels=labels,
                  yticklabels=labels,
                  vmin=0, vmax=1,
                  cbar=True,
                  annot_kws={'fontsize': 70})

colorbar = ax.collections[0].colorbar
colorbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
colorbar.set_ticklabels(['0.0', '0.25', '0.5', '0.75', '1.0'])
colorbar.ax.tick_params(labelsize=50)
plt.xlabel('Predicted Label', fontsize=80, labelpad=40)
plt.ylabel('True Label', fontsize=80, labelpad=40)
plt.tick_params(length=30, width=8, pad=20)
plt.xticks(fontsize=70)
plt.yticks(fontsize=70)
plt.savefig(f"../../figure/result3/confusion_matrix/{methods}方法在外部测试集上的混淆矩阵.pdf", format="pdf", dpi=600, bbox_inches='tight')
plt.show()
