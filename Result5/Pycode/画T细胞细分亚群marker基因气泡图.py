# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 20:54:58 2025

@author: Administrator
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from sklearn.preprocessing import MinMaxScaler



data = pd.read_csv("../../data/result5/source/T细胞细分亚群及其marker基因表.csv")
ratio_df = pd.read_csv("E:/工作/论文/黑色素瘤/code/MelanomaRcode/result5/data/result/T细胞各细分类型中标记基因的表达占比.csv", index_col=0)
mean_df = pd.read_csv("E:/工作/论文/黑色素瘤/code/MelanomaRcode/result5/data/result/T细胞各细分类型中标记基因的平均表达值.csv", index_col=0)


scaler = MinMaxScaler()
ratio_df = pd.DataFrame(
    scaler.fit_transform(ratio_df),
    columns=ratio_df.columns,
    index=ratio_df.index
)

cell_type_order = data['CellType'].unique().tolist()


ratio_df = ratio_df.loc[cell_type_order]
mean_df = mean_df.loc[cell_type_order]
ratio_long = ratio_df.reset_index().melt(id_vars="cell_type", var_name="Gene", value_name="Percent")
mean_long = mean_df.reset_index().melt(id_vars="cell_type", var_name="Gene", value_name="Expression")
plot_df = pd.merge(ratio_long, mean_long, on=["cell_type", "Gene"])
plot_df["cell_type"] = pd.Categorical(plot_df["cell_type"], categories=cell_type_order, ordered=True)
gene_order = list(ratio_df.columns)[::-1]
plot_df["Gene"] = pd.Categorical(plot_df["Gene"], categories=gene_order, ordered=True)







fig, ax = plt.subplots(figsize=(10, 20))
norm = mpl.colors.Normalize(vmin=-1, vmax=plot_df["Expression"].max(), clip=True)
sc = ax.scatter(
    x=plot_df["cell_type"],
    y=plot_df["Gene"],
    s=plot_df["Percent"] * 1000,
    c=plot_df["Expression"],
    cmap="PuBu",
    norm=norm,  # 
    edgecolors="black",
    linewidths=0.3
)
# cax = fig.add_axes([-0.35, 0.7, 0.04, 0.14])  # [left, bottom, width, height]
cax = fig.add_axes([-0.8, 0.7, 0.04, 0.14])  # [left, bottom, width, height]
cb = mpl.colorbar.ColorbarBase(cax, cmap="PuBu", norm=norm, orientation='vertical')
cb.ax.tick_params(labelsize=50)
# cax.set_title("Average Expression", fontsize=20, pad=10)
cax.set_title("")
cax.text(
    x=0, y=1.05, s="Average Expression",
    fontsize=50, ha='left', va='bottom', transform=cax.transAxes
)
size_levels = [0, 25, 50, 75]
size_handles = [
    Line2D([0], [0], marker='o', linestyle='None',
           markerfacecolor='black', markeredgecolor='black',
           markersize=(s * 0.4 + 4), label=f"{s}")
    for s in size_levels
]
legend = ax.legend(
    handles=size_handles,
    title="Percent Expressed",
    fontsize=40,
    loc="upper left",
    bbox_to_anchor=(-1.3, 0.75), 
    frameon=False,
    labelspacing=1,
    handletextpad=1.5
)
legend._legend_box.align = "left"
legend.get_title().set_fontsize(50)
ax.set_xticklabels(ax.get_xticklabels(), rotation=60, ha='right', fontsize=50)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=50)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(6)
ax.spines['bottom'].set_linewidth(6)
ax.set_xlabel("")
ax.set_ylabel("")
ax.tick_params(length=15, width=6)
plt.savefig('../../figure/result5/T细胞细分亚群marker基因气泡图.pdf', format='pdf', bbox_inches='tight', dpi=600)
plt.show()