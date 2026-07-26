# -*- coding: utf-8 -*-
"""
Created on Wed May 28 16:03:58 2025

@author: Administrator
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from scipy.stats import pearsonr, spearmanr
import numpy as np
# from sklearn.metrics import confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC



seed = np.random.randint(10000)


# seed = 8637 2064 324

seed = 324  

print(seed)

data = pd.read_csv("../../data/result3/source/GSE190113_deg_data.csv")


print(data['label'].value_counts())


X = data.iloc[:, 1:]
y = data.iloc[:, 0]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)








models = {
    "LR": LogisticRegression(max_iter=30),
    "RF": RandomForestClassifier(n_estimators=80, random_state=seed),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=seed),
    "NB": GaussianNB(),
    "SVM": SVC(probability=True, random_state=seed),
    "LDA": LinearDiscriminantAnalysis(),
    "QDA": QuadraticDiscriminantAnalysis()
}


for model in models.values():
    model.fit(X_train, y_train)


internal_results = []
external_results = []


print(f"{'Model':<20} {'Accuracy':>5} {'Precision':>6} {'Recall':>8} {'F1-Score':>9} {'AUC':>6} {'Pearson':>11} {'Spearman':>9}")
for name, model in models.items():
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob) if len(set(y_test)) == 2 else float('nan')
    pearson_corr, _ = pearsonr(y_test, y_prob)
    spearman_corr, _ = spearmanr(y_test, y_prob)
    internal_results.append([name, acc, prec, rec, f1, auc, pearson_corr, spearman_corr])
    print(f"{name:<20} {acc:6.3f} {prec:9.3f} {rec:9.3f} {f1:9.3f} {auc:8.3f} {pearson_corr:9.3f} {spearman_corr:9.3f}")
    
    
test_data = pd.read_csv("../../data/result3/source/deg_external_final_data.csv")
X_external = test_data.iloc[:, 1:]
y_external = test_data.iloc[:, 0]
X_external_scaled = scaler.transform(X_external)
# X_external_scaled = X_external

print("\n外部测试集评估结果：")
print(f"{'Model':<20} {'Accuracy':>5} {'Precision':>6} {'Recall':>8} {'F1-Score':>9} {'AUC':>6} {'Pearson':>11} {'Spearman':>9}")
for name, model in models.items():
    y_pred = model.predict(X_external_scaled)
    y_prob = model.predict_proba(X_external_scaled)[:, 1] if hasattr(model, "predict_proba") else y_pred
    acc = accuracy_score(y_external, y_pred)
    prec = precision_score(y_external, y_pred, zero_division=0)
    rec = recall_score(y_external, y_pred, zero_division=0)
    f1 = f1_score(y_external, y_pred, zero_division=0)
    auc = roc_auc_score(y_external, y_prob) if len(set(y_external)) == 2 else float('nan')
    pearson_corr, _ = pearsonr(y_external, y_prob)
    spearman_corr, _ = spearmanr(y_external, y_prob)
    external_results.append([name, acc, prec, rec, f1, auc, pearson_corr, spearman_corr])
    print(f"{name:<20} {acc:6.3f} {prec:9.3f} {rec:9.3f} {f1:9.3f} {auc:8.3f} {pearson_corr:9.3f} {spearman_corr:9.3f}")


# internal_df = pd.DataFrame(internal_results, columns=["Model", "Accuracy", "Precision", "Recall", "F1-Score", "AUC", "Pearson", "Spearman"])
# external_df = pd.DataFrame(external_results, columns=["Model", "Accuracy", "Precision", "Recall", "F1-Score", "AUC", "Pearson", "Spearman"])

# cols_to_round = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC", "Pearson", "Spearman"]
# internal_df[cols_to_round] = internal_df[cols_to_round].round(3)
# external_df[cols_to_round] = external_df[cols_to_round].round(3)


# internal_df.to_csv("../../data/result3/result/test_results_deg.csv", index=False)

# external_df.to_csv("../../data/result3/result/external_test_results_deg.csv", index=False)



# for name, model in models.items():
#     y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.predict(X_test)
#     df = pd.DataFrame({
#         "y_true": y_test.values,
#         "y_prob": y_prob
#     })
#     df.to_csv(f"../../data/result3/auroc/roc_internal_{name.replace(' ', '_')}_deg.csv", index=False)


# for name, model in models.items():
#     y_prob = model.predict_proba(X_external_scaled)[:, 1] if hasattr(model, "predict_proba") else model.predict(X_external_scaled)
#     df = pd.DataFrame({
#         "y_true": y_external.values,
#         "y_prob": y_prob
#     })
#     df.to_csv(f"../../data/result3/auroc/roc_external_{name.replace(' ', '_')}_deg.csv", index=False)





# for name, model in models.items():
#     filename = f"../../data/result3/model/{name.replace(' ', '_')}_deg.pkl"
#     joblib.dump(model, filename)


# joblib.dump(scaler, "../../data/result3/model/StandardScaler_deg.pkl")





