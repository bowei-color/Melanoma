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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC



seed = 2565



data = pd.read_csv("../data/train_and_test_data.csv")
external_test_data = pd.read_csv("../data/external_data.csv")

X = data.iloc[:, 1:]
y = data.iloc[:, 0]

X_external = external_test_data.iloc[:, 1:]
y_external = external_test_data.iloc[:, 0]


    
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



X_external_scaled = scaler.transform(X_external)


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
    




