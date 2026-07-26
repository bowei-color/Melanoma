import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter, KaplanMeierFitter

df = pd.read_csv("../../data/result6/source/gse190113_cox_original_data.csv")
features = ["Immune","CellCycle","Translation_Ribosome","Stress","Exhausted","Cytotoxicity"]
time_col, event_col = "diagnosis_survival_period", "survival"
horizons = [5, 10, 15]  

cph = CoxPHFitter()
cph.fit(df[[time_col, event_col] + features], duration_col=time_col, event_col=event_col)

def predict_surv_prob_at(t, X):
    lp = cph.predict_partial_hazard(X).values.flatten()
    base = cph.baseline_cumulative_hazard_
    H0_t = np.interp(float(t), base.index.values.astype(float), base.values.flatten())
    return np.exp(-H0_t * lp)

def km_observed_at(t, idx):
    kmf = KaplanMeierFitter()
    sub = df.loc[idx, [time_col, event_col]]
    kmf.fit(sub[time_col], event_observed=sub[event_col])
    tv = kmf.survival_function_.index.values.astype(float)
    sv = kmf.survival_function_["KM_estimate"].values
    if t <= tv.min(): return float(sv[0])
    if t >= tv.max(): return float(sv[-1])
    return float(np.interp(t, tv, sv))

def calibration_curve_at(t, n_bins=10):
    p = predict_surv_prob_at(t, df[features])
    bins = pd.qcut(p, q=n_bins, labels=None, duplicates="drop")
    xs, ys = [], []
    for b in sorted(bins.unique()):
        idx = (bins == b)
        if idx.sum() < 10: continue
        xs.append(p[idx].mean())
        ys.append(km_observed_at(t, idx))
    return pd.DataFrame({"pred": xs, "obs": ys}).dropna().sort_values("pred")

plt.figure(figsize=(20, 20))
colors = ["#1b9e77", "#d95f02", "#7570b3"]
for i, t in enumerate(horizons):
    cal = calibration_curve_at(t)
    plt.plot(cal["pred"], cal["obs"], "o-", linewidth=8, color=colors[i], label=f"{t}-year")
plt.plot([0, 1], [0, 1], "--", color="gray", linewidth=7)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel("Predicted survival probability", fontsize=70)
plt.ylabel("Observed survival probability (KM)", fontsize=70)
plt.title("Calibration Curves of the Cox Model", fontsize=70, pad=20)
plt.legend(fontsize=50, loc="lower right", frameon=False)
plt.tick_params(length=8, width=2, labelsize=70)
plt.xticks(fontsize=70, rotation=45, ha="right")
plt.yticks(fontsize=70)
plt.tick_params(length=16, width=6, pad=20)
plt.gca().spines["bottom"].set_linewidth(6)
plt.gca().spines["left"].set_linewidth(6)
plt.gca().spines["right"].set_visible(False)
plt.gca().spines["top"].set_visible(False)
plt.savefig("../../figure/result6/Cox校准曲线（补充图1C）.pdf", format="pdf", dpi=600, bbox_inches="tight")
plt.show()
