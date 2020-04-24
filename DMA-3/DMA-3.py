# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 14:09:07 2018

@author: rache
"""

import numpy as np
from matplotlib import pyplot as plt

"""
##### Problem 1 #####
def distance (x, u):
    dist = 0
    for i in range (2):
        dist = dist + abs(x[i]-u[i])**2
    dist = np.sqrt(dist)
    return dist

data = [[0,2], [0,0], [1.5,0], [5,0], [5,2]]
c1_ini = [data[1-1], data[2-1], data[4-1]]
c2_ini = [data[3-1], data[5-1]]
u_c1 = np.mean(c1_ini, axis = 0)
u_c2 = np.mean(c2_ini, axis = 0)
c1 = list()
c2 = list()
for i in range (len(data)):
    d1 = distance(data[i], u_c1)
    d2 = distance(data[i], u_c2)
    if d1 <= d2:
        c1.append(data[i])
    else:
        c2.append(data[i])
u_c1 = np.mean(c1, axis = 0)
u_c2 = np.mean(c2, axis = 0)
t = 1
        
if c1 != c1_ini:
    c1_ini = c1
    c2_ini = c2
    c1 = list()
    c2 = list()
    for i in range (len(data)):
        d1 = distance(data[i], u_c1)
        d2 = distance(data[i], u_c2)
        if d1 <= d2:
            c1.append(data[i])
        else:
            c2.append(data[i])
    u_c1 = np.mean(c1, axis = 0)
    u_c2 = np.mean(c2, axis = 0)
    t = t+1
"""

##### Problem 2 #####
def f (x, u, o):
    o_inv = np.linalg.inv(o)
    f = np.exp(- (x-u) * o_inv * (x-u).T/2) / (2*np.pi*np.sqrt(np.linalg.det(o)))    
    return f
    
data = np.asmatrix([[0.5,4.5], [2.2,1.5], [3.9,3.5], [2.1,1.9], [0.5,3.2], [0.8,4.3], [2.7,1.1], [2.5,3.5], [2.8,3.9], [0.1,4.1]])
u = np.asmatrix([[0.5, 4.5], [2.2, 1.6], [3, 3.5]])
sig = np.asarray([[1,0], [0,1]])
p_c1 = p_c2 = p_c3 = 1/3
##### A & C & D
w = np.array([[0.0 for x in range(10)] for y in range(3)])
u_re = []
p_re = []
sig_re = []
for i in range(3):
    u_num= 0
    w_sum = 0
    sig_num = 0;
    for j in range(10):
        denom = f(data[j], u[0], sig)*p_c1 + f(data[j], u[1], sig)*p_c2 + f(data[j], u[2], sig)*p_c3
        wij = f(data[j], u[i], sig)*p_c1 / denom
        w[i][j] = wij
        u_num = u_num + wij*data[j]
        w_sum = w_sum + wij
        sig_num = sig_num + wij.item(0)*(data[j]-u[i]).T*(data[j]-u[i])
    u1 = [(u_num/w_sum).item(0), (u_num/w_sum).item(1)]
    sig1 = sig_num/w_sum
    u_re.append(u1)
    p_re.append((w_sum/10).item(0))
    sig_re.append(sig1)
u_re = np.array(u_re)

w = np.array([[0.0 for x in range(10)] for y in range(3)])
u_new = []
p_new = []
sig_new = []
for i in range(3):
    u_num= 0
    w_sum = 0
    sig_num = 0;
    for j in range(10):
        denom = f(data[j], u[0], sig)*p_c1 + f(data[j], u[1], sig)*p_c2 + f(data[j], u[2], sig)*p_c3
        wij = f(data[j], u[i], sig)*p_c1 / denom
        w[i][j] = wij
        u_num = u_num + wij*data[j]
        w_sum = w_sum + wij
        sig_num = sig_num + wij.item(0)*(data[j]-u_re[i]).T*(data[j]-u_re[i])
    u1 = [(u_num/w_sum).item(0), (u_num/w_sum).item(1)]
    sig1 = sig_num/w_sum
    u_new.append(u1)
    p_new.append((w_sum/10).item(0))
    sig_new.append(sig1)



##### B
x = list(); y = list();
data = np.array(data)
u = np.array(u)
for k in range(10):
    x.append(data[k][0])
    y.append(data[k][1])
for k in range(3):
    x.append(u[k][0])
    y.append(u[k][1])
for k in range(3):
    x.append(u_re[k][0])
    y.append(u_re[k][1])
        
plt.scatter(x[:10], y[:10], color='blue', label='data')
plt.scatter(x[10:13], y[10:13], color='red', label='initial means')
plt.scatter(x[13:], y[13:], color='green', label='iterated means')
plt.legend(loc='bottom left')
plt.show()
