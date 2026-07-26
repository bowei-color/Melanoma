# -*- coding: utf-8 -*-
"""
Created on Sat Jun 21 15:10:11 2025

@author: Administrator
"""



import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

model_name = "LR"

df = pd.read_csv(f"../../data/result3/auroc/roc_external_{model_name}.csv")
df_deg = pd.read_csv(f"../../data/result3/auroc/roc_external_{model_name}_deg.csv")

fpr, tpr, _ = roc_curve(df["y_true"], df["y_prob"])
roc_auc = auc(fpr, tpr)
fpr_deg, tpr_deg, _ = roc_curve(df_deg["y_true"], df_deg["y_prob"])
roc_auc_deg = auc(fpr_deg, tpr_deg)


plt.figure(figsize=(20, 20))
plt.plot(fpr, tpr, label=f"LR (AUC = {roc_auc:.3f})", linewidth=6)
plt.plot(fpr_deg, tpr_deg, label=f"LR-DEG (AUC = {roc_auc_deg:.3f})", linewidth=6)
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
plt.savefig("../../figure/result3/LR方法在外部测试集上的auroc曲线图.pdf", format='pdf', dpi=600, bbox_inches='tight')
plt.show()