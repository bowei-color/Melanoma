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
# import joblib
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC



seed = np.random.randint(10000)

# seed =  9535  105 216 324 483 2565  9877


# seed = 9535  2565

seed = 2565
print(seed)


data = pd.read_csv("../../data/result3/source/GSE190113_final_data.csv")
external_test_data = pd.read_csv("../../data/result3/source/external_final_data.csv")

X = data.iloc[:, 1:]
y = data.iloc[:, 0]

X_external = external_test_data.iloc[:, 1:]
y_external = external_test_data.iloc[:, 0]


    
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



X_external_scaled = scaler.transform(X_external)
# X_external_scaled = scaler.fit_transform(X_external)
# X_external_scaled = X_external



models = {
    "LR": LogisticRegression(max_iter=30),
    "RF": RandomForestClassifier(n_estimators=80, random_state=seed),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "XGBoost": XGBClassifier(use_label_encoder=True, eval_metric='logloss', random_state=seed),
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



# mel_data = pd.read_csv("../../data/result3/source/mel_tsam_liang_2017_data.csv")
# skcm_data = pd.read_csv("../../data/result3/source/skcm_mskcc_2014_external_data.csv")
# gse215119_data = pd.read_csv("../../data/result3/source/GSE215119_external_data.csv")
# X_mel = mel_data.iloc[:, 1:]
# y_mel = mel_data.iloc[:, 0]

# X_skcm = skcm_data.iloc[:, 1:]
# y_skcm = skcm_data.iloc[:, 0]

# X_gse = gse215119_data.iloc[:, 1:]
# y_gse = gse215119_data.iloc[:, 0]

# X_mel_scaled = scaler.transform(X_mel)
# X_skcm_scaled = scaler.transform(X_skcm)
# X_gse_scaled = scaler.transform(X_gse)
# X_gse_recovered = scaler.inverse_transform(X_gse_scaled)


# X_external_all = np.vstack([X_mel_scaled, X_skcm_scaled, X_gse_recovered])
# y_external_all = np.hstack([y_mel.values, y_skcm.values, y_gse])


# X_external_all_df = pd.DataFrame(X_external_all, columns=X_mel.columns)
# X_external_all_df.insert(0, "label", y_external_all)

# X_external_scaled = X_external_all_df.iloc[:, 1:]
# y_external = X_external_all_df.iloc[:, 0]


# X_external_scaled = X_gse_scaled
# y_external = y_gse


# internal_df.to_csv("../../data/result3/result/test_results.csv", index=False)
# external_df.to_csv("../../data/result3/result/external_test_results.csv", index=False)



# for name, model in models.items():
#     y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.predict(X_test)
#     df = pd.DataFrame({
#         "y_true": y_test.values,
#         "y_prob": y_prob
#     })
#     df.to_csv(f"../../data/result3/auroc/roc_internal_{name.replace(' ', '_')}.csv", index=False)


# for name, model in models.items():
#     y_prob = model.predict_proba(X_external_scaled)[:, 1] if hasattr(model, "predict_proba") else model.predict(X_external_scaled)
#     df = pd.DataFrame({
#         "y_true": y_external.values,
#         "y_prob": y_prob
#     })
#     df.to_csv(f"../../data/result3/auroc/roc_external_{name.replace(' ', '_')}.csv", index=False)



# for name, model in models.items():
#     y_pred = model.predict(X_test)
#     cm = confusion_matrix(y_test, y_pred)
#     cm_df = pd.DataFrame(cm, index=['True_0', 'True_1'], columns=['Pred_0', 'Pred_1'])
#     cm_df.to_csv(f"../../data/result3/confunsion_matrix/{name.replace(' ', '_')}_confusion_matrix.csv")



# for name, model in models.items():
#     y_pred = model.predict(X_external_scaled)
#     cm = confusion_matrix(y_external, y_pred)
#     cm_df = pd.DataFrame(cm, index=['True_0', 'True_1'], columns=['Pred_0', 'Pred_1'])
#     cm_df.to_csv(f"../../data/result3/confunsion_matrix_external/{name.replace(' ', '_')}_confusion_matrix_external.csv")



# for name, model in models.items():
#     filename = f"../../data/result3/model/{name.replace(' ', '_')}1.pkl"
#     joblib.dump(model, filename)


# joblib.dump(scaler, "../../data/result3/model/StandardScaler1.pkl")





