# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 16:37:13 2022

@author: 17187
"""
import math
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from toad.transform import Combiner
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, auc, mean_squared_error, accuracy_score


# 数据导入
def data_import(file_name):
    return pd.read_csv(file_name)


# 卡方分箱
def chi_merge(data_X, data_Y):
    comb = Combiner()
    comb.fit(data_X, data_Y, method='chi')
    result = comb.transform(data_X)
    return result


# 决策树
def classify(data_X, data_Y):
    # 划分训练集和测试集
    X_train1, X_test1, Y_train1, Y_test1 = train_test_split(data_X, data_Y, test_size=0.2)
    clf = DecisionTreeClassifier()
    clf.fit(X_train1, Y_train1)
    return clf, X_test1, Y_test1


# 绘制ROC曲线
def check_fit_roc(predict, target):
    fpr, tpr, _ = roc_curve(target, predict, drop_intermediate=False)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([-0.1, 1.05])
    plt.ylim([-0.1, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    predicts = [1 if i >= 0.5 else 0 for i in predict]
    print("RMSE :" + str(math.sqrt(mean_squared_error(target, predict))))
    print("accuracy :" + str(accuracy_score(target, predicts)))
    print("AUC : " + str(roc_auc))


# 绘制KS曲线
def check_fit_ks(predict, target):
    fpr, tpr, thresholds = roc_curve(target, predict, drop_intermediate=False)
    plt.figure()
    plt.plot(fpr, color='darkorange', lw=2, label='False Positive Rate')
    plt.plot(tpr, color='darkblue', lw=2, label='True Positive Rate')
    plt.title('ks-curve')
    plt.show()
    print("ks_value :" + str(max(tpr - fpr)))


# 绘制直方图
def plot_hist(data):
    for index, row in data.iteritems():
        plt.figure()
        plt.hist(row, histtype='bar', rwidth=0.8)


file_name = "E://Study//研1下//4大数据分析与应用//Homework//Work 1//Credit_handle.csv"
data = data_import(file_name)

X = data.iloc[:, 2:12]
Y = data.iloc[:, 1]
chi_X = chi_merge(X, Y)
plot_hist(X)
plot_hist(chi_X)
clf, X_test1, Y_test1 = classify(chi_X, Y)

predict_target = clf.predict(X_test1)
print(sum(predict_target == Y_test1))
print(metrics.classification_report(Y_test1, predict_target))

check_fit_roc(predict_target, Y_test1)
print('\n')
check_fit_ks(predict_target, Y_test1)
