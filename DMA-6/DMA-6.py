# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 20:39:53 2018

@author: rache
"""

##### ECEN758-DMA-HW6

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from graphviz import Source
from sklearn.model_selection import train_test_split

"""
##### Part(1): Orig plot
data = pd.read_excel('traceDMA.xlsx', index_col=None, header=None)
data = pd.Series.tolist(data)
udp = list()
tcp = list()
for i in range(len(data)):
    if data[i][2] == 'udp':
        udp.append([data[i][0], data[i][1]])
    elif data[i][2] == 'tcp':
        tcp.append([data[i][0], data[i][1]])
udp = np.asarray(udp)
tcp = np.asarray(tcp)

plt.figure()
plt.scatter(udp[:,0], udp[:,1], marker='o', color='blue', label='udp')
plt.scatter(tcp[:,0], tcp[:,1], marker='*', color='red', label='tcp')
plt.legend(loc='upper right')
plt.xlabel('PORT')
plt.ylabel('SIZE')
plt.show()
plt.figure()
plt.scatter(udp[:,0], udp[:,1], marker='o', color='blue', label='udp')
plt.scatter(tcp[:,0], tcp[:,1], marker='*', color='red', label='tcp')
plt.legend(loc='lower right')
plt.xlabel('PORT')
plt.ylabel('SIZE')
plt.xscale('log')
plt.yscale('symlog')
plt.show()

############### Part(2-6): Decision Tree ###############
data = pd.read_excel('traceDMA.xlsx', index_col=None, header=None)
X_train = data.values[:, :2]
Y_train = data.values[:, 2:]
X_test = data.values[:, :2]
Y_test = data.values[:, 2:]
#Train
clf = DecisionTreeClassifier(criterion = "entropy")
clf.fit(X_train, Y_train)
acc_100_train= clf.score(X_train, Y_train, sample_weight=None)
#Tree
graph = Source(export_graphviz(clf, out_file=None))
graph.format = 'png'
graph.render('tree_100%',view=False)
#Test/Predict
Y_predict = clf.predict(X_test)
acc_100 = clf.score(X_test, Y_test, sample_weight=None)

udp = list()
tcp = list()
data = pd.Series.tolist(data)
for j in range(len(Y_predict)):
    if Y_predict[j] == 'udp':
        udp.append([data[j][0], data[j][1]])
    elif Y_predict[j] == 'tcp':
        tcp.append([data[j][0], data[j][1]])
udp = np.asarray(udp)
tcp = np.asarray(tcp)
plt.figure()
plt.scatter(udp[:,0], udp[:,1], color='blue', label='udp')
plt.scatter(tcp[:,0], tcp[:,1], color='red', label='tcp')
plt.legend(loc='upper right')
plt.show()
"""
############### Part(7-8): Split 70% ##############
data = pd.read_excel('traceDMA.xlsx', index_col=None, header=None)
X_train = data.values[:7000, :2]
Y_train = data.values[:7000, 2:]
X_test = data.values[7000:, :2]
Y_test = data.values[7000:, 2:]
#Train
clf_70 = DecisionTreeClassifier(criterion = "entropy", max_depth = 4)
clf_70.fit(X_train, Y_train)
acc_70 = clf_70.score(X_train, Y_train, sample_weight=None)
#Tree
graph = Source(export_graphviz(clf_70, out_file=None))
graph.format = 'png'
graph.render('tree_70%',view=False)
#Test/Predict
Y_predict = clf_70.predict(X_test)
acc_30 = clf_70.score(X_test, Y_test, sample_weight=None)

udp = list()
tcp = list()
udp_pred = list()
tcp_pred = list()
udp_train = list()
tcp_train = list()
data = pd.Series.tolist(data)
for i in range(len(Y_predict)):
    if Y_predict[i] == 'udp':
        udp_pred.append([data[i][0], data[i][1]])
    elif Y_predict[i] == 'tcp':
        tcp_pred.append([data[i][0], data[i][1]])
for j in range(len(Y_test)):
    if Y_test[j] == 'udp':
        udp.append([data[j][0], data[j][1]])
    elif Y_test[j] == 'tcp':
        tcp.append([data[j][0], data[j][1]])
for k in range(len(Y_train)):
    if Y_train[k] == 'udp':
        udp_train.append([data[k][0], data[k][1]])
    elif Y_train[k] == 'tcp':
        tcp_train.append([data[k][0], data[k][1]])
udp = np.asarray(udp)
tcp = np.asarray(tcp)
udp_pred = np.asarray(udp_pred)
tcp_pred = np.asarray(tcp_pred)
udp_train = np.asarray(udp_train)
tcp_train = np.asarray(tcp_train)
#plt.figure()
#plt.scatter(udp_train[:,0], udp_train[:,1], color='blue', label='udp_train_70')
#plt.scatter(tcp_train[:,0], tcp_train[:,1], color='red', label='tcp_train_70')
#plt.legend(loc='upper right')
#plt.xlabel('PORT')
#plt.ylabel('SIZE')
#plt.title('Train 70% Distribution')
#plt.show()
plt.scatter(X_train.T[0],X_train.T[1], marker = '*', color = 'Blue')
plt.xlabel('PORT')
plt.ylabel('SIZE')
plt.title('Train 70% Distribution')
"""
############### Part(9): Better Split 70% ###############
data = pd.read_excel('traceDMA.xlsx', index_col=None, header=None)
X = data.values[:, :2]
Y = data.values[:, 2:]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3)
#X_train = np.log10(X_train)
#Train
clf_best = DecisionTreeClassifier(criterion = "entropy", max_depth = 4)
clf_best.fit(X_train, Y_train)
acc_best_70 = clf_best.score(X_train, Y_train, sample_weight=None)
#Tree
graph = Source(export_graphviz(clf_best, out_file=None))
graph.format = 'png'
graph.render('tree_best',view=False)
#Test/Predict
Y_predict = clf_best.predict(X_test)
acc_best_30 = clf_best.score(X_test, Y_test, sample_weight=None)

#udp = list()
#tcp = list()
#udp_pred = list()
#tcp_pred = list()
udp_train = list()
tcp_train = list()
data = pd.Series.tolist(data)
#for i in range(len(Y_predict)):
#    if Y_predict[i] == 'udp':
#        udp_pred.append([data[i][0], data[i][1]])
#    elif Y_predict[i] == 'tcp':
#        tcp_pred.append([data[i][0], data[i][1]])
#for j in range(len(Y_test)):
#    if Y_test[j] == 'udp':
#        udp.append([data[j][0], data[j][1]])
#    elif Y_test[j] == 'tcp':
#        tcp.append([data[j][0], data[j][1]])
for k in range(len(Y_train)):
    if Y_train[k] == 'udp':
        udp_train.append([data[k][0], data[k][1]])
    elif Y_train[k] == 'tcp':
        tcp_train.append([data[k][0], data[k][1]])
#udp = np.asarray(udp)
#tcp = np.asarray(tcp)
#udp_pred = np.asarray(udp_pred)
#tcp_pred = np.asarray(tcp_pred)
#udp_train = np.asarray(udp_train)
#tcp_train = np.asarray(tcp_train)
#plt.figure()
#plt.scatter(udp_train[:,:1], udp_train[:,1:], color='blue', label='udp_train_best')
#plt.scatter(tcp_train[:,:1], tcp_train[:,1:], color='red', label='tcp_train_best')
#plt.legend(loc='upper right')
#plt.xlabel('PORT')
#plt.ylabel('SIZE')
#plt.title('Train Best Distribution')
#plt.show()
plt.scatter(X_train.T[0],X_train.T[1], marker = '*', color = 'Blue')
plt.xlabel('PORT')
plt.ylabel('SIZE')
plt.title('Train Best Distribution')
"""