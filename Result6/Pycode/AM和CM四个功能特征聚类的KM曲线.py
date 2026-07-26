# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 13:25:50 2025

@author: Administrator
"""

import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import matplotlib.pyplot as plt


clinic_data = pd.read_csv("../../data/result6/source/GSE190113_clinical_data.csv")



# =============================================================================
# 画AM生存曲线
# =============================================================================

cluster_labels = pd.read_csv("../../data/result6/result/AM四个功能特征聚类结果.csv")
sample_cluster = pd.DataFrame(cluster_labels.reset_index()).rename(columns={"0": "cluster"})
clinc_merged = sample_cluster.merge(clinic_data, on="sample", how="inner")
data1 = clinc_merged[clinc_merged["cluster"] == 1]
data2 = clinc_merged[clinc_merged["cluster"] == 2]
kmf1 = KaplanMeierFitter()
kmf2 = KaplanMeierFitter()
p_value = logrank_test( data1["diagnosis_survival_period"], data2["diagnosis_survival_period"], event_observed_A=data1["survival"], event_observed_B=data2["survival"]).p_value

plt.figure(figsize=(20, 20))
kmf1.fit(data1["diagnosis_survival_period"], event_observed=data1["survival"], label="amcluster 1")
ax = kmf1.plot_survival_function(ci_show=True, linewidth=4)
kmf2.fit(data2["diagnosis_survival_period"], event_observed=data2["survival"], label="amcluster 2")
kmf2.plot_survival_function(ax=ax, ci_show=True, linewidth=4)
plt.xlabel("Survival Time (Years)", fontsize=70, labelpad=20)
plt.ylabel("Survival Probability", fontsize=70, labelpad=20)
plt.xticks(fontsize=70)
plt.yticks(fontsize=70)
plt.tick_params(length=16, width=6, pad=20)
plt.gca().spines["bottom"].set_linewidth(6)
plt.gca().spines["left"].set_linewidth(6)
plt.gca().spines["right"].set_visible(False)
plt.gca().spines["top"].set_visible(False)
plt.text(0.6, 0.7, f"p = {p_value:.3e}", fontsize=60, transform=ax.transAxes)
plt.legend(loc="upper right", fontsize=60)
plt.savefig("../../figure/result6/基于黑色素细胞功能的结果图/AM四个功能特征聚成2类的生存曲线.pdf", format="pdf", dpi=600, bbox_inches="tight")
plt.show()


# =============================================================================
# 画CM生存曲线
# =============================================================================



cluster_labels1 = pd.read_csv("../../data/result6/result/CM四个功能特征聚类结果.csv")
sample_cluster1 = pd.DataFrame(cluster_labels1.reset_index()).rename(columns={"0": "cluster"})
clinc_merged1 = sample_cluster1.merge(clinic_data, on="sample", how="inner")
data3 = clinc_merged1[clinc_merged1["cluster"] == 1]
data4 = clinc_merged1[clinc_merged1["cluster"] == 2]
kmf3 = KaplanMeierFitter()
kmf4 = KaplanMeierFitter()
p_value = logrank_test( data3["diagnosis_survival_period"], data4["diagnosis_survival_period"], event_observed_A=data3["survival"], event_observed_B=data4["survival"]).p_value

plt.figure(figsize=(20, 20))
kmf3.fit(data3["diagnosis_survival_period"], event_observed=data3["survival"], label="cmcluster 1")
ax = kmf3.plot_survival_function(ci_show=True, linewidth=4)
kmf4.fit(data4["diagnosis_survival_period"], event_observed=data4["survival"], label="cmcluster 2")
kmf4.plot_survival_function(ax=ax, ci_show=True, linewidth=4)
plt.xlabel("Survival Time (Years)", fontsize=70, labelpad=20)
plt.ylabel("Survival Probability", fontsize=70, labelpad=20)
plt.xticks(fontsize=70)
plt.yticks(fontsize=70)
plt.tick_params(length=16, width=6, pad=20)
plt.gca().spines["bottom"].set_linewidth(6)
plt.gca().spines["left"].set_linewidth(6)
plt.gca().spines["right"].set_visible(False)
plt.gca().spines["top"].set_visible(False)
plt.text(0.6, 0.7, f"p = {p_value:.3e}", fontsize=60, transform=ax.transAxes)
plt.legend(loc="upper right", fontsize=60)
plt.savefig("../../figure/result6/基于黑色素细胞功能的结果图/CM四个功能特征聚成2类的生存曲线.pdf", format="pdf", dpi=600, bbox_inches="tight")
plt.show()
