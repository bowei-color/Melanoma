# -*- coding: utf-8 -*-
"""
Created on Wed May 28 19:04:59 2025

@author: Administrator
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# =============================================================================
# 画测试集的Accuracy和Auroc
# =============================================================================

test_data = pd.read_csv('../../data/result3/result/test_results.csv')
test_data = test_data.rename(columns={"AUC": "Auroc"})
metrics = ['Accuracy', 'Auroc']
models = test_data['Model'].tolist()
x = np.arange(len(metrics))
width = 0.1
colors = ["#de5253", "#c98182", "#caca4e", "#ea92ce", "#a986ca", "#56b456", "#fe993d", "#4c92c3"][:len(models)]
# colors = ["#db645d", "#e78555", "#ee9f62", "#f7c376", "#9ad2d5", "#6bb3c8", "#568ba5", "#436a92"][:len(models)]


plt.figure(figsize=(20, 20))

for i, model in enumerate(models):
    values = test_data.loc[i, metrics].values
    plt.bar(x + i * width, values, width, label=model, color=colors[i], edgecolor='black', linewidth=4)

plt.xticks(x + width * (len(models) - 1) / 2, metrics, fontsize=70)
plt.yticks(fontsize=70)
plt.ylim(0, 1)
plt.tick_params(length=20, width=6, pad=20)
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.gca().spines["left"].set_linewidth(6)
plt.gca().spines["bottom"].set_linewidth(6)
plt.legend(fontsize=60, frameon=False, borderaxespad=0, bbox_to_anchor=(1.02, 0.9), loc='upper left')
# plt.savefig("../../figure/result3/测试集Accuracy和Auroc柱状图.pdf", format="pdf", dpi=600, bbox_inches="tight")
plt.show()


# =============================================================================
# 画外部测试集的Accuracy和Auroc
# =============================================================================


external_test_results = pd.read_csv('../../data/result3/result/external_test_results.csv')
external_test_results = external_test_results.rename(columns={"AUC": "Auroc"})
metrics = ['Accuracy', 'Auroc']
models = external_test_results['Model'].tolist()
x = np.arange(len(metrics))
width = 0.1
colors = ["#de5253", "#c98182", "#caca4e", "#ea92ce", "#a986ca", "#56b456", "#fe993d", "#4c92c3"][:len(models)]
# colors = ["#db645d", "#e78555", "#ee9f62", "#f7c376", "#9ad2d5", "#6bb3c8", "#568ba5", "#436a92"][:len(models)]


plt.figure(figsize=(20, 20))

for i, model in enumerate(models):
    values = external_test_results.loc[i, metrics].values
    plt.bar(x + i * width, values, width, label=model, color=colors[i], edgecolor='black', linewidth=4)

plt.xticks(x + width * (len(models) - 1) / 2, metrics, fontsize=70)
plt.yticks(fontsize=70)
plt.ylim(0, 1)
plt.tick_params(length=20, width=6, pad=20)
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.gca().spines["left"].set_linewidth(8)
plt.gca().spines["bottom"].set_linewidth(8)
plt.legend(fontsize=60, frameon=False, borderaxespad=0, bbox_to_anchor=(1.02, 0.9), loc='upper left')
# plt.savefig("../../figure/result3/外部测试集Accuracy和Auroc柱状图.pdf", format="pdf", dpi=600, bbox_inches="tight")
plt.show()


