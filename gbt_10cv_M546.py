import numpy as np 
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier
from sklearn.metrics import balanced_accuracy_score, accuracy_score, precision_score, recall_score, roc_auc_score, multilabel_confusion_matrix, f1_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import scipy as sp
from math import sqrt
import pickle
import sys 
import os 
import pandas as pd
import multiprocessing as mp

def normalize(X):
    scaler = StandardScaler().fit(X)
    return scaler.transform(X)

X_val1 = np.load('./M546/X_M546_aux.npy')
X_val2 = np.load('./M546/X_M546_FRI.npy')
X_val3 = np.load('./M546/X_M546_SR_curves.npy')
X_val4 = np.load('./M546/X_M546_ESM.npy')
X_val = np.concatenate((X_val1, X_val2), axis=1)
X_val = np.concatenate((X_val,  X_val3), axis=1)
X_val = np.concatenate((X_val,  X_val4), axis=1)

X_norm = normalize(X_val)
X_val, X_blind = X_norm[:492], X_norm[492:]
y_val = np.load(f'./M546/Y_M546.npy')
y_val, y_blind = y_val[:492], y_val[492:]

scores = []
results = []

for i in range(1):
    kf = KFold(n_splits=5, shuffle=True)
    
    tmp = np.zeros(len(y_val)) 

    for idx, (train_idx, test_idx) in enumerate(kf.split(X_val)):

        X_train, X_test = X_val[train_idx], X_val[test_idx]
        y_train, y_test = y_val[train_idx], y_val[test_idx]

        clf1 = GradientBoostingClassifier(n_estimators = 20000, learning_rate=0.05, max_features='sqrt', max_depth=6, subsample=0.7, min_samples_split=3)
        clf2 = RandomForestClassifier(n_estimators = 20000, max_features='sqrt', max_depth=6, min_samples_split=3)
        clf3 = ExtraTreesClassifier(n_estimators = 20000, max_features='sqrt', max_depth=6, min_samples_split=3)
        
        reg = VotingClassifier(estimators=[('gb', clf1), ('rf', clf2), ('et', clf3)], voting='soft')
        reg.fit(X_train, y_train)
        #y_pred = (reg.predict_proba(X_test)[:,1] >= 0.3).astype(bool)
        y_pred = reg.predict(X_test)
        tmp[test_idx] = y_pred

        print(f'CV {i+1}, Fold {idx+1}.....................') 
        print(f'MCC: {matthews_corrcoef(y_test, y_pred)}') 
        print(f'Accuracy: {accuracy_score(y_test, y_pred)}') 
        print(f'AUC: {roc_auc_score(y_test, y_pred)}') 
        print(f'F1 : {f1_score(y_test, y_pred)}') 
        print(f'Precision: {precision_score(y_test, y_pred)}')
        print(f'Recall: {recall_score(y_test, y_pred)}')
        print(f'Balanced Acc: {balanced_accuracy_score(y_test,y_pred)}')
    
    results.append(tmp)

results = np.mean(results, axis = 0)
results = np.round(results)
#results = (results >= 0.3).astype(bool)
results = np.array(results, dtype=int) 

print(f'Final Result.....................')
print(f'MCC: {matthews_corrcoef(y_val, results)}')
print(f'ACC: {accuracy_score(y_val, results)}')
print(f'AUC: {roc_auc_score(y_val, results)}')
print(f'F1 : {f1_score(y_val, results)}')
print(f'Precision: {precision_score(y_val, results)}')
print(f'Recall: {recall_score(y_val, results)}')
print(f'Balanced Acc: {balanced_accuracy_score(y_val,results)}')

"""
PSR rates
MCC: 0.6854324459549164
ACC: 0.8780487804878049
AUC: 0.8157700810611644
F1 : 0.9186991869918699
Precision: 0.8828125
Recall: 0.9576271186440678
Balanced Acc: 0.8157700810611643

PSR curves 
MCC: 0.6962471951423991
ACC: 0.8821138211382114
AUC: 0.8185949398182264
F1 : 0.9216216216216216
Precision: 0.883419689119171
Recall: 0.963276836158192
Balanced Acc: 0.8185949398182264

PH
MCC: 0.6962471951423991
ACC: 0.8821138211382114
AUC: 0.8185949398182264
F1 : 0.9216216216216216
Precision: 0.883419689119171
Recall: 0.963276836158192
Balanced Acc: 0.8185949398182264


Laplacian
MCC: 0.7074340117350791
ACC: 0.8861788617886179
AUC: 0.8280520756570867
F1 : 0.9239130434782609
Precision: 0.8900523560209425
Recall: 0.96045197740113
Balanced Acc: 0.8280520756570867
"""