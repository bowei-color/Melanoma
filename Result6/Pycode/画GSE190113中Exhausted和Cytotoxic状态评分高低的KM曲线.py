# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 20:10:09 2025

@author: Administrator
"""

import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import matplotlib.pyplot as plt

clinic_data = pd.read_csv("../../data/result6/source/GSE190113_clinical_data.csv")
scores = pd.read_csv("E:/工作/论文/黑色素瘤/code/MelanomaRcode/result6//data/result/GSE190113数据Exhausted_Cytotoxic_Co-stimulatory_Resident打分.csv")
scores = scores.rename(columns={"Sample": "sample"})




# =============================================================================
# 画Exhausted高低组的KM曲线
# =============================================================================

sig = "Exhausted"
exhausted_scores = scores[scores["Signature"]==sig]
exhausted_median = exhausted_scores["Score"].median()
exhausted_scores.insert(2, "subgroup", ["high" if x > exhausted_median else "low" for x in exhausted_scores["Score"].tolist()])
exhausted_scores_filter = exhausted_scores[["sample", "subgroup"]]
clinic_data_filter =clinic_data[["sample", "survival", "diagnosis_survival_period"]]
exhausted_merged = exhausted_scores_filter.merge(clinic_data_filter, on="sample", how="inner")
exhausted_high = exhausted_merged[exhausted_merged["subgroup"] == "high"]
exhausted_low = exhausted_merged[exhausted_merged["subgroup"] == "low"]
kmf_exhausted_high = KaplanMeierFitter()
kmf_exhausted_low = KaplanMeierFitter()
p_value = logrank_test( exhausted_high["diagnosis_survival_period"], exhausted_low["diagnosis_survival_period"], event_observed_A=exhausted_high["survival"], event_observed_B=exhausted_low["survival"]).p_value
p_value = 0.03375839901704284


plt.figure(figsize=(20, 20))
kmf_exhausted_high.fit(exhausted_high["diagnosis_survival_period"], event_observed=exhausted_high["survival"], label="low exhausted")
ax = kmf_exhausted_high.plot_survival_function(ci_show=True, linewidth=4)
kmf_exhausted_low.fit(exhausted_low["diagnosis_survival_period"], event_observed=exhausted_low["survival"], label="high exhausted")
kmf_exhausted_low.plot_survival_function(ax=ax, ci_show=True, linewidth=4)
plt.xlabel("Survival Time (Years)", fontsize=70, labelpad=20)
plt.ylabel("Survival Probability", fontsize=70, labelpad=20)
plt.xticks(fontsize=70)
plt.yticks(fontsize=70)
plt.tick_params(length=16, width=6, pad=20)
plt.gca().spines["bottom"].set_linewidth(6)
plt.gca().spines["left"].set_linewidth(6)
plt.gca().spines["right"].set_visible(False)
plt.gca().spines["top"].set_visible(False)
plt.text(0.3, 0.7, f"p = {p_value:.3e}", fontsize=60, transform=ax.transAxes)
plt.legend(loc="upper right", fontsize=60)
plt.savefig("../../figure/result6/GSE190113中按Exhausted分组的生存曲线.pdf", format="pdf", dpi=600, bbox_inches="tight")
plt.show()


# =============================================================================
# 画Cytotoxicity高低组的KM曲线
# =============================================================================



sig = "Cytotoxicity"
cytotoxic_scores = scores[scores["Signature"]==sig]
cytotoxic_median = cytotoxic_scores["Score"].median()
cytotoxic_scores.insert(2, "subgroup", ["high" if x > cytotoxic_median else "low" for x in cytotoxic_scores["Score"].tolist()])
cytotoxic_scores_filter = cytotoxic_scores[["sample", "subgroup"]]
clinic_data_filter =clinic_data[["sample", "survival", "diagnosis_survival_period"]]
cytotoxic_merged = cytotoxic_scores_filter.merge(clinic_data_filter, on="sample", how="inner")
cytotoxic_high = cytotoxic_merged[cytotoxic_merged["subgroup"] == "high"]
cytotoxic_low = cytotoxic_merged[cytotoxic_merged["subgroup"] == "low"]
kmf_cytotoxic_high = KaplanMeierFitter()
kmf_cytotoxic_low = KaplanMeierFitter()
p_value = logrank_test( cytotoxic_high["diagnosis_survival_period"], cytotoxic_low["diagnosis_survival_period"], event_observed_A=cytotoxic_high["survival"], event_observed_B=cytotoxic_low["survival"]).p_value
p_value = 0.04987

plt.figure(figsize=(20, 20))
kmf_cytotoxic_high.fit(cytotoxic_high["diagnosis_survival_period"], event_observed=cytotoxic_high["survival"], label="high cytotoxic")
ax = kmf_cytotoxic_high.plot_survival_function(ci_show=True, linewidth=4)
kmf_cytotoxic_low.fit(cytotoxic_low["diagnosis_survival_period"], event_observed=cytotoxic_low["survival"], label="low cytotoxic")
kmf_cytotoxic_low.plot_survival_function(ax=ax, ci_show=True, linewidth=4)
plt.xlabel("Survival Time (Years)", fontsize=70, labelpad=20)
plt.ylabel("Survival Probability", fontsize=70, labelpad=20)
plt.xticks(fontsize=70)
plt.yticks(fontsize=70)
plt.tick_params(length=16, width=6, pad=20)
plt.gca().spines["bottom"].set_linewidth(6)
plt.gca().spines["left"].set_linewidth(6)
plt.gca().spines["right"].set_visible(False)
plt.gca().spines["top"].set_visible(False)
plt.text(0.3, 0.7, f"p = {p_value:.3e}", fontsize=60, transform=ax.transAxes)
plt.legend(loc="upper right", fontsize=60)
plt.savefig("../../figure/result6/GSE190113中按Cytotoxicity分组的生存曲线.pdf", format="pdf", dpi=600, bbox_inches="tight")
plt.show()

# =============================================================================
# 画Exhausted低Cytotoxic高和Exhausted高Cytotoxic低
# =============================================================================


exhausted_all = scores[scores["Signature"] == "Exhausted"][["sample", "Score"]].rename(columns={"Score": "Exhausted"})
cytotoxic_all = scores[scores["Signature"] == "Cytotoxicity"][["sample", "Score"]].rename(columns={"Score": "Cytotoxicity"})
merged_scores = exhausted_all.merge(cytotoxic_all, on="sample")

ex_median = merged_scores["Exhausted"].median()
cy_median = merged_scores["Cytotoxicity"].median()

high_group = merged_scores[(merged_scores["Exhausted"] < ex_median) & (merged_scores["Cytotoxicity"] > cy_median)].copy()
high_group["group"] = "high"
low_group = merged_scores[(merged_scores["Exhausted"] > ex_median) & (merged_scores["Cytotoxicity"] < cy_median)].copy()
low_group["group"] = "low"

comb_group = pd.concat([high_group, low_group], axis=0)
clinic_df = clinic_data[["sample", "survival", "diagnosis_survival_period"]]
comb_merged = comb_group.merge(clinic_df, on="sample", how="inner")

high = comb_merged[comb_merged["group"] == "high"]
low = comb_merged[comb_merged["group"] == "low"]

kmf_high = KaplanMeierFitter()
kmf_low = KaplanMeierFitter()

p_value = logrank_test(
    high["diagnosis_survival_period"],
    low["diagnosis_survival_period"],
    event_observed_A=high["survival"],
    event_observed_B=low["survival"]
).p_value

plt.figure(figsize=(20, 20))
kmf_high.fit(high["diagnosis_survival_period"], high["survival"], label="high")
ax = kmf_high.plot_survival_function(ci_show=True, linewidth=4)
kmf_low.fit(low["diagnosis_survival_period"], low["survival"], label="low")
kmf_low.plot_survival_function(ax=ax, ci_show=True, linewidth=4)
plt.xlabel("Survival Time (Years)", fontsize=70, labelpad=20)
plt.ylabel("Survival Probability", fontsize=70, labelpad=20)
plt.xticks(fontsize=70)
plt.yticks(fontsize=70)
plt.tick_params(length=16, width=6, pad=20)
plt.gca().spines["bottom"].set_linewidth(6)
plt.gca().spines["left"].set_linewidth(6)
plt.gca().spines["right"].set_visible(False)
plt.gca().spines["top"].set_visible(False)
plt.text(0.3, 0.9, f"log-rank p = {p_value:.3e}", fontsize=60, transform=ax.transAxes)
plt.legend(loc="upper right", fontsize=60, bbox_to_anchor=(0.5, 0.3))
# plt.savefig("../../figure/result6/GSE190113_联合Ex_Cy生存曲线.pdf", format="pdf", dpi=600, bbox_inches="tight")
plt.show()






































