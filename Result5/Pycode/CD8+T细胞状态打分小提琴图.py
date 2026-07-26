# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 21:04:17 2025

@author: Administrator
"""





import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("E:/工作/论文/黑色素瘤/code/MelanomaRcode/result5/data/result/CD8+T细胞状态打分结果.csv")


palette = {"AM": "#e7c161", "CM": "#3c87b2"}
group_order = ["AM", "CM"]
positions = [0, 1]

# sns.set(style="whitegrid")



# =============================================================================
# Exhausted评分小提琴图
# =============================================================================

sig = "Exhausted"
sub_df = df[df["Signature"] == sig].copy()
p_text = "p_adj=" + format(sub_df["p_adj"].iloc[0], ".2e")


for group in group_order:
    median_score = sub_df[sub_df["group"] == group]["Score"].median()
    print(f"{group}组 Exhausted 中位数: {median_score:.3f}")


plt.figure(figsize=(10, 20))
sns.set_style("white")
ax = plt.gca()
sns.violinplot(data=sub_df, x="group", y="Score", palette=palette,
                inner=None, linewidth=4, cut=0, order=group_order, ax=ax)

for i, group in enumerate(group_order):
    y = sub_df[sub_df["group"] == group]["Score"]
    ax.boxplot(y, positions=[positions[i]], widths=0.2, patch_artist=True,
                boxprops=dict(facecolor=palette[group], linewidth=4),
                medianprops=dict(color="black", linewidth=4),
                whiskerprops=dict(linewidth=4),
                capprops=dict(linewidth=4),
                flierprops=dict(marker='o', markersize=0))

plt.text(0.5, sub_df["Score"].max() * 1.25, p_text, ha="center", va="bottom", fontsize=70)
plt.title("Exhausted", fontsize=70, pad=40)
plt.xlabel("")
plt.ylabel("")
plt.xticks(positions, group_order, fontsize=70)
plt.yticks(fontsize=70)
plt.tick_params(length=20, width=6, pad=20)
plt.ylim(-1.6, 3.1)

ax.spines["bottom"].set_linewidth(6)
ax.spines["left"].set_linewidth(6)
# ax.spines["left"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.legend([], [], frameon=False)

for tick in ax.xaxis.get_major_ticks():
    tick.tick1line.set_visible(True)
    tick.tick1line.set_markersize(20)
    tick.tick1line.set_markeredgewidth(6)
    tick.tick1line.set_color("black")
for tick in ax.yaxis.get_major_ticks():
    tick.tick1line.set_visible(True)
    tick.tick1line.set_markersize(20)
    tick.tick1line.set_markeredgewidth(6)
    tick.tick1line.set_color("black")
plt.savefig("../../figure/result5/CD8+T细胞Exhausted评分小提琴图.pdf", format="pdf", dpi=600, bbox_inches="tight")
plt.show()





# =============================================================================
#  Cytotoxicity评分小提琴图
# =============================================================================

sig = "Cytotoxicity"
sub_df = df[df["Signature"] == sig]
p_text = "p_adj=" + format(sub_df["p_adj"].iloc[0], ".2e")


for group in group_order:
    median_score = sub_df[sub_df["group"] == group]["Score"].median()
    print(f"{group}组 Exhausted 中位数: {median_score:.3f}")

plt.figure(figsize=(10, 20))
sns.set_style("white")
ax = plt.gca()
sns.violinplot(data=sub_df, x="group", y="Score", palette=palette,
               inner=None, linewidth=4, cut=0, bw=0.3, scale="width", order=["AM", "CM"])


for i, group in enumerate(group_order):
    y = sub_df[sub_df["group"] == group]["Score"]
    ax.boxplot(y, positions=[positions[i]], widths=0.2, patch_artist=True,
               boxprops=dict(facecolor=palette[group], linewidth=4),
               medianprops=dict(color="black", linewidth=4),
               whiskerprops=dict(linewidth=4),
               capprops=dict(linewidth=4),
               flierprops=dict(marker='o', markersize=0))
plt.text(0.5, sub_df["Score"].max() * 1.05, p_text, ha="center", va="bottom", fontsize=70)
plt.title("Cytotoxicity", fontsize=70, pad=40)
plt.xlabel("")
plt.ylabel("")
plt.xticks(positions, group_order, fontsize=70)
plt.yticks([], fontsize=0)
plt.tick_params(axis="x",length=20, width=6, pad=20)
plt.ylim(-1.6, 3.1)
plt.gca().spines["bottom"].set_linewidth(6)
# plt.gca().spines["left"].set_linewidth(6)
plt.gca().spines["left"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.gca().spines["top"].set_visible(False)
plt.legend([], [], frameon=False)
ax = plt.gca()
for tick in ax.xaxis.get_major_ticks():
    tick.tick1line.set_visible(True)
    tick.tick1line.set_markersize(20)
    tick.tick1line.set_markeredgewidth(6)
    tick.tick1line.set_color("black")
for tick in ax.yaxis.get_major_ticks():
    tick.tick1line.set_visible(True)
    tick.tick1line.set_markersize(20)
    tick.tick1line.set_markeredgewidth(6)
    tick.tick1line.set_color("black")
plt.savefig("../../figure/result5/CD8+T细胞Cytotoxicity评分小提琴图.pdf", format="pdf", dpi=600, bbox_inches="tight")
plt.show()

# =============================================================================
# Co-stimulatory评分小提琴图
# =============================================================================

sig = "Co-stimulatory"
sub_df = df[df["Signature"] == sig]
p_text = "p_adj=" + format(sub_df["p_adj"].iloc[0], ".2e")

plt.figure(figsize=(10, 20))
sns.set_style("white")
ax = plt.gca()
sns.violinplot(data=sub_df, x="group", y="Score", palette=palette,
               inner=None, linewidth=4, cut=0, order=group_order, ax=ax)

for i, group in enumerate(group_order):
    y = sub_df[sub_df["group"] == group]["Score"]
    ax.boxplot(y, positions=[positions[i]], widths=0.2, patch_artist=True,
               boxprops=dict(facecolor=palette[group], linewidth=4),
               medianprops=dict(color="black", linewidth=4),
               whiskerprops=dict(linewidth=4),
               capprops=dict(linewidth=4),
               flierprops=dict(marker='o', markersize=0))
plt.text(0.5, sub_df["Score"].max() * 1.56, p_text, ha="center", va="bottom", fontsize=70)
plt.title("Co-stimulatory", fontsize=70, pad=40)
plt.xlabel("")
plt.ylabel("")
plt.xticks(positions, group_order, fontsize=70)
plt.yticks([], fontsize=0)
plt.tick_params(length=20, width=6, pad=20)
plt.ylim(-1.6, 3.1)
plt.gca().spines["bottom"].set_linewidth(6)
# plt.gca().spines["left"].set_linewidth(6)
plt.gca().spines["left"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.gca().spines["top"].set_visible(False)
plt.legend([], [], frameon=False)
ax = plt.gca()
for tick in ax.xaxis.get_major_ticks():
    tick.tick1line.set_visible(True)
    tick.tick1line.set_markersize(20)
    tick.tick1line.set_markeredgewidth(6)
    tick.tick1line.set_color("black")
for tick in ax.yaxis.get_major_ticks():
    tick.tick1line.set_visible(True)
    tick.tick1line.set_markersize(20)
    tick.tick1line.set_markeredgewidth(6)
    tick.tick1line.set_color("black")
plt.savefig("../../figure/result5/CD8+T细胞Co-stimulatory评分小提琴图.pdf", format="pdf", dpi=600, bbox_inches="tight")
plt.show()




# =============================================================================
# Resident评分小提琴图
# =============================================================================

sig = "Resident"
sub_df = df[df["Signature"] == sig]
p_text = "p_adj=" + format(sub_df["p_adj"].iloc[0], ".2e")


for group in group_order:
    median_score = sub_df[sub_df["group"] == group]["Score"].median()
    print(f"{group}组 Exhausted 中位数: {median_score:.3f}")


plt.figure(figsize=(10, 20))
sns.set_style("white")
ax = plt.gca()
sns.violinplot(data=sub_df, x="group", y="Score", palette=palette,
               inner=None, linewidth=4, cut=0, bw=0.3, scale="width", order=["AM", "CM"])


for i, group in enumerate(group_order):
    y = sub_df[sub_df["group"] == group]["Score"]
    ax.boxplot(y, positions=[positions[i]], widths=0.2, patch_artist=True,
               boxprops=dict(facecolor=palette[group], linewidth=4),
               medianprops=dict(color="black", linewidth=4),
               whiskerprops=dict(linewidth=4),
               capprops=dict(linewidth=4),
               flierprops=dict(marker='o', markersize=0))
plt.text(0.5, sub_df["Score"].max() * 1.40, p_text, ha="center", va="bottom", fontsize=70)
plt.title("Resident", fontsize=70, pad=40)
plt.xlabel("")
plt.ylabel("")
plt.xticks(positions, group_order, fontsize=70)
plt.xticks(fontsize=70)
plt.yticks([], fontsize=0)
plt.tick_params(length=20, width=6, pad=20)
plt.ylim(-1.6, 3.1)
plt.gca().spines["bottom"].set_linewidth(6)
# plt.gca().spines["left"].set_linewidth(6)
plt.gca().spines["left"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.gca().spines["top"].set_visible(False)
plt.legend([], [], frameon=False)
ax = plt.gca()
for tick in ax.xaxis.get_major_ticks():
    tick.tick1line.set_visible(True)
    tick.tick1line.set_markersize(20)
    tick.tick1line.set_markeredgewidth(6)
    tick.tick1line.set_color("black")
for tick in ax.yaxis.get_major_ticks():
    tick.tick1line.set_visible(True)
    tick.tick1line.set_markersize(20)
    tick.tick1line.set_markeredgewidth(6)
    tick.tick1line.set_color("black")
plt.savefig("../../figure/result5/CD8+T细胞Resident评分小提琴图.pdf", format="pdf", dpi=600, bbox_inches="tight")
plt.show()



