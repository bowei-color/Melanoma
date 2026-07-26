# -*- coding: utf-8 -*-
"""
Created on Sun Jun  8 10:50:51 2025

@author: Administrator
"""


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


model_names = ["LR", "RF", "KNN", "XGBoost", "NB", "SVM", "LDA", "QDA"]


colors = ['#ebc847', '#58a04d', '#ef8d28', '#af7aa0', '#4f79a7', '#75b6b1', '#e0595b', '#c9bc9c']

i = 0
plt.figure(figsize=(20, 20))
for name in model_names:
    df = pd.read_csv(f"../../data/result3/auroc/roc_internal_{name}.csv")
    fpr, tpr, _ = roc_curve(df["y_true"], df["y_prob"])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})", color=colors[i], linewidth=6)
    i = i + 1
# plt.plot([0, 1], [0, 1], "k--", linewidth=6)
# plt.title("ROC Curve - Test Set", fontsize=80, pad=40)
plt.xlabel("False Positive Rate", fontsize=80, labelpad=20)
plt.ylabel("True Positive Rate", fontsize=80, labelpad=20)
plt.xticks(fontsize=80)
plt.yticks(fontsize=80)
plt.tick_params(length=20, width=6, pad=20)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_linewidth(8)
plt.gca().spines['left'].set_linewidth(8)
plt.legend(loc="lower right", fontsize=60, bbox_to_anchor=(1.2, 0))
plt.savefig("../../figure/result3/测试集的auroc曲线图.pdf", format='pdf', dpi=600, bbox_inches='tight')
plt.show()



i = 0
plt.figure(figsize=(20, 20))
for name in model_names:
    df = pd.read_csv(f"../../data/result3/auroc/roc_external_{name}.csv")
    fpr, tpr, _ = roc_curve(df["y_true"], df["y_prob"])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})", color=colors[i], linewidth=6)
    i = i + 1
# plt.title("ROC Curve - External Test Set", fontsize=80, pad=40)
# plt.plot([0, 1], [0, 1], "k--", linewidth=6)
plt.xlabel("False Positive Rate", fontsize=80, labelpad=20)
plt.ylabel("True Positive Rate", fontsize=80, labelpad=20)
plt.xticks(fontsize=80)
plt.yticks(fontsize=80)
plt.tick_params(length=20, width=6, pad=20)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_linewidth(8)
plt.gca().spines['left'].set_linewidth(8)
plt.legend(loc="lower right", fontsize=60, bbox_to_anchor=(1.2, 0))
plt.savefig("../../figure/result3/外部测试集的auroc曲线图.pdf", format='pdf', dpi=600, bbox_inches='tight')
plt.show()
