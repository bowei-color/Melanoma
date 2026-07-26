# -*- coding: utf-8 -*-
"""
Created on Sat Jun 14 12:26:10 2025

@author: Administrator
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv("E:/工作/论文/黑色素瘤/code/MelanomaRcode/result3/data/result/Adjusted_ssGSEA_Matrix.csv")
go_route_data = pd.read_csv("E:/工作/论文/黑色素瘤/code/MelanomaRcode//result2/data/source/代表性功能通路_用于ssGSEA.csv")
sample_label_map = pd.read_csv("E:/工作/论文/黑色素瘤/code/MelanomaRcode/result3//data/result/bulk数据打分后样本和标签的对应关系.csv")



data = data.rename(columns={"Unnamed: 0": "Description"})
go_route_data["Functional_Category"] = go_route_data["Functional_Category"].replace({
    "Immune_Response": "Signature1",
    "Cell_Cycle": "Signature2",
    "Translation_Ribosome": "Signature3",
    "Stress_Response": "Signature4"
    })
data = data.iloc[4:, :]
data_filter = data.copy()
data_merged = pd.merge(data_filter, go_route_data[['Description', 'Functional_Category']], on='Description', how='left')
data_long = pd.melt(data_merged, id_vars=['Description', 'Functional_Category'], var_name='Sample', value_name='ssGSEA_score')
data_long = pd.merge(data_long, sample_label_map, on='Sample', how='left')

colors = ['#f5cea3', '#d7d7d4']

plt.figure(figsize=(22, 20))
boxprops = dict(linewidth=4)
whiskerprops = dict(linewidth=4)
capprops = dict(linewidth=4)
medianprops = dict(linewidth=4, color="black")
sns.boxplot(x='Functional_Category', y='ssGSEA_score', hue='Group', data=data_long, patch_artist=True, palette=colors,
                      boxprops=boxprops, whiskerprops=whiskerprops, capprops=capprops, medianprops=medianprops)
sns.stripplot(data=data_long, x='Functional_Category', y='ssGSEA_score', hue='Group', dodge=True, palette=colors, size=8, jitter=True, alpha=0.8)
handles, labels = plt.gca().get_legend_handles_labels()
plt.xlabel("Functional Category", fontsize=0, labelpad=20)
plt.ylabel("ssGSEA Score", fontsize=0, labelpad=20)
plt.xticks(rotation=0, fontsize=50, ha='center')
plt.yticks(fontsize=60)
plt.tick_params(length=20, width=4, pad=20)
# plt.gca().spines['top'].set_visible(False)
# plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_linewidth(4)
plt.gca().spines['right'].set_linewidth(4)
plt.gca().spines['top'].set_linewidth(4)
plt.gca().spines['bottom'].set_linewidth(4)
plt.legend(handles[:2], labels[:2], title='', loc='upper right', fontsize=50, bbox_to_anchor=(0.25, 0.25))
plt.savefig("../../figure/result3/bulk数据的四个功能特征评分分布箱线图.pdf", format="pdf", dpi=600, bbox_inches="tight")
plt.show()