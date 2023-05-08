# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 20:38:14 2022

@author: 17187
"""

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing

data = pd.read_csv("E://Study//研1下//4大数据分析与应用//Homework//Work 1//Credit_handle.csv")
data.drop('ID', axis=1, inplace=True)
pca_data = PCA(n_components=0.99)
pca_result = pca_data.fit_transform(data)
print(pca_data.explained_variance_ratio_)
print(pca_data.explained_variance_)
print(pca_data.n_components_)

pca_result = pd.concat([pd.DataFrame(pca_result),data['Label']], axis=1)
output = "E://Study//研1下//4大数据分析与应用//Homework//Work 1//pca.csv"
pd.DataFrame(pca_result).to_csv(output, sep=',', index=False)
