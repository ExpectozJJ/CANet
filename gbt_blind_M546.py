import numpy as np 
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, accuracy_score, precision_score, recall_score, roc_auc_score, multilabel_confusion_matrix, f1_score
from sklearn.metrics import accuracy_score
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
import argparse 

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
    reg = GradientBoostingClassifier(n_estimators = 20000, learning_rate=0.05, max_features='sqrt', max_depth=7, subsample=0.4, min_samples_split=3)
    reg.fit(X_val, y_val)
    y_pred = reg.predict(X_blind)
    results.append(y_pred)

    print(f'MCC: {matthews_corrcoef(y_blind, y_pred)}')
    print(f'ACC: {accuracy_score(y_blind, y_pred)}')
    print(f'AUC: {roc_auc_score(y_blind, y_pred)}')
    print(f'F1-Score: {f1_score(y_blind, y_pred)}')
    print(f'Precision: {precision_score(y_blind, y_pred)}')
    print(f'Recall: {recall_score(y_blind, y_pred)}')
    print(f'Balanced Acc: {balanced_accuracy_score(y_blind,y_pred)}')
    
    scores.append([matthews_corrcoef(y_blind, y_pred), accuracy_score(y_blind, y_pred), roc_auc_score(y_blind, y_pred), f1_score(y_blind, y_pred), precision_score(y_blind, y_pred), recall_score(y_blind, y_pred),balanced_accuracy_score(y_blind,y_pred)])

scores = np.array(scores)
print(f'MCC: {np.mean(scores[:,0])}')
print(f'ACC: {np.mean(scores[:,1])}')
print(f'AUC: {np.mean(scores[:,2])}')
print(f'F1-Score: {np.mean(scores[:,3])}')
print(f'Precision: {np.mean(scores[:,4])}')
print(f'Recall: {np.mean(scores[:,5])}')
print(f'Balanced Acc: {np.mean(scores[:,6])}')