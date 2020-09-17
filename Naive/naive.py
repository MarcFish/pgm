# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 17:16:16 2020

@author: MarcFish
"""

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler as sscale
import numpy as np
import scipy.stats as stats
from sklearn import metrics

X,y = load_iris(return_X_y=True)
scale = sscale()
X = scale.fit_transform(X)
feature_num = X.shape[1]
cat_num = len(np.unique(y))
def pdf(mean,std):
    return stats.norm(mean,std).pdf
p_y = np.zeros(cat_num)
for c in y:
    p_y[c] += 1
p_y = np.log(p_y/np.sum(p_y))  # compute cat p
p_f = [None]*cat_num
for i in range(cat_num):
    p_f[i] = [None]*feature_num
    for j in range(feature_num):
        p_f[i][j] = pdf(np.mean(X[np.where(y==i),i]),np.std(X[np.where(y==i),i]))  # compude cdf

y_pred = list()
for x in X:
    temp = [None]*cat_num
    for i in range(cat_num):
        temp[i] = p_y[i]
        for j in range(feature_num):
            temp[i] += np.log(p_f[i][j](x[j]))
    y_pred.append(np.argmax(temp))
y_pred = np.asarray(y_pred).astype(np.float32)
print("accuracy:{:.4f}".format(metrics.accuracy_score(y,y_pred)))
print("recall:{:.4f}".format(metrics.recall_score(y,y_pred,average='micro')))
