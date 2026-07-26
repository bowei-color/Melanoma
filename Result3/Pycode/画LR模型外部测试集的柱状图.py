# -*- coding: utf-8 -*-
"""
Created on Sat Jun 21 14:50:48 2025

@author: Administrator
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


model = "LR"

test_data = pd.read_csv('../../data/result3/result/external_test_results.csv')
test_deg_data = pd.read_csv('../../data/result3/result/external_test_results_deg.csv')

test_filter = test_data[test_data["Model"]==model]
test_deg_filter = test_deg_data[test_deg_data["Model"]==model]


test_filter.insert(1, "Methods", "External LR")
test_deg_filter.insert(1, "Methods", "External LR-DEG")


merged_data = pd.concat([test_filter, test_deg_filter], axis=0)


external_test_results = merged_data.rename(columns={"AUC": "Auroc"})
metrics = ['Accuracy', 'Auroc']
methods = external_test_results['Methods'].tolist()
x = np.arange(len(metrics))
width = 0.25
colors = ["#d8d8d5", "#fad2a5"]
# colors = ["#db645d", "#e78555", "#ee9f62", "#f7c376", "#9ad2d5", "#6bb3c8", "#568ba5", "#436a92"][:len(models)]


plt.figure(figsize=(20, 20))

for i, method in enumerate(methods):
    values = external_test_results.iloc[i][metrics].values.astype(float)
    plt.bar(x + i * width, values, width=width, label=method, color=colors[i], edgecolor='black', linewidth=4)

plt.xticks(x + width * (len(methods) - 1) / 2, metrics, fontsize=70)
plt.yticks(fontsize=70)
plt.ylim(0, 1)
plt.tick_params(length=20, width=6, pad=20)
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.gca().spines["left"].set_linewidth(8)
plt.gca().spines["bottom"].set_linewidth(8)
plt.legend(fontsize=60, frameon=False, borderaxespad=0, bbox_to_anchor=(0.16, 1), loc='upper left')
plt.savefig(f"../../figure/result3/{model}方法外部测试集Accuracy和Auroc柱状图.pdf", format="pdf", dpi=600, bbox_inches="tight")
plt.show()