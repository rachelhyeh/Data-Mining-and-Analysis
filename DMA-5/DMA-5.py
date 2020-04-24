# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 17:33:51 2018

@author: rache
"""

#ECEN758-DMA-HW5

import numpy as np
"""
#####Problem 18.5-Q2
data = np.asarray([[1,1,5.0], [1,1,7.0], [1,0,8.0], [0,0,3.0], [0,1,7.0], [0,1,4.0], [0,0,5.0], [1,0,6.0], [0,1,1.0]])
test = np.asarray([1,0,1.0])
d = np.asarray([[1, 2, 4, 8], [3, 5, 6, 7, 9]])
n = len(data)
ni = np.asarray([len(d[0]), len(d[1])])
pi = list()
ui = list()
z = [[],[]]
sig = [[],[]]
for i in range(len(d)):
    #Prior Probability
    pi.append(ni[i]/n)
    #Mean
    sum = 0
    for x in range(len(d[i])):
        sum = sum + data[d[i][x]-1]
    ui.append(sum/ni[i])
    #Centered Data
    for c in range(len(d[i])):
        z[i].append(data[d[i][c]-1]-ui[i].T)
    #Variance
    zi = z[i]
    for xi in range(len(data.T)):
        zj = list()
        for y in range(len(d[i])):
            zj.append(zi[y][xi])
        zj = np.asarray(zj)
        sig[i].append( np.dot(zj, zj.T)/ni[i] )
p = [[],[]]
for ci in range(len(pi)):
    f = list()
    for d in range(len(data.T)):
        
        a = np.exp(-(test[d]-ui[ci][d])**2 / (2*((sig[ci][d])**2))) / (np.sqrt(2*np.pi)*(sig[ci][d]))
        f.append(a)
    p[ci] = f[0]*f[1]*f[2]
if p[0] >= p[1]:
    test_y = "Y"
else:
    test_y = "N"

"""
#####Problem 18.5-Q3
test = [3, 4]
u = np.asarray([[1, 3], [5,5]])
sig1 = [[5,3], [3,2]]
sig2 = [[2,0], [0,1]]
sig = [sig1, sig2]
pi = np.asarray([0.5, 0.5])

p = [[],[]]
for i in range(len(u)):
    sig_inv = np.linalg.inv(sig[i])
    f = np.exp(- np.dot(np.dot((test-u[i]),sig_inv),(test-u[i]).T)/2) / (2*np.pi*np.sqrt(np.linalg.det(sig[i])))
    p[i] = f * pi[i]
if p[0] >= p[1]:
    test_y = "C1"
else:
    test_y = "C2"
