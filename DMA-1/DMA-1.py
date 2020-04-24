# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 16:21:19 2018

@author: rache
"""

#ECEN758-DMA-Assignment1

import numpy as np
from scipy import stats
from matplotlib import pyplot as plt

#Problem 2
x = [69, 74, 68, 70, 72, 67, 66, 70, 76, 68, 72, 79, 74, 67, 66, 71, 74, 75, 75, 76]
y = [153, 175, 155, 135, 172, 150, 115, 137, 200, 130, 140, 265, 185, 112, 140, 150, 165, 185, 210, 220]
#(a)
x_mean = np.array(x).mean()
x_median = np.median(x)
x_mode = stats.mode(x)[0]
x_var = np.var(x, ddof = 0)
x_std = np.sqrt(x_var)
#(b)
y_mean = np.array(y).mean()
y_median = np.median(y)
y_mode = stats.mode(y)[0]
y_var = np.var(y, ddof = 0)
y_std = np.sqrt(y_var)
#(b)using formula
y_sum = 0;
for i in range(len(y)):
    y_sum = y_sum + (y[i]-y_mean)**2
y_varf = y_sum/len(y)
#(c)
axis = np.linspace(x_mean-3*x_std, x_mean+3*x_std, 100)
x_pdf = stats.norm.pdf(axis, x_mean, x_std)
plt.plot(axis, x_pdf, label='PDF of X')
plt.show()
#(d)
x_freq = sum(i>80 for i in np.array(x))
#(e)
xy = np.array([x, y])
xy_mean = xy.mean(axis = 1)
xy_cov = np.cov(xy, ddof = 0)
#(f)
xy_corr = np.correlate(x, y)
#(g)
plt.scatter(x, y)
plt.title('Age vs. Weight')
plt.xlabel('age')
plt.ylabel('weight')
plt.show()



#Problem 3
x1 = [8, 0, 10, 10, 2]
x2 = [-20, -1, -19, -20, 0]
#(a)
d = np.array([x1, x2])
d_mean = d.mean(axis = 1)
d_cov = np.cov(d, ddof = 0)
#(b)
d_eigValue = np.linalg.eigh(d_cov)[0]
#(d)
fpc_eigValue = np.linalg.eigh(d_cov)[1]
fpc_index = np.argmax(d_eigValue)
fpc_eigVectors = (np.linalg.eigh(d_cov)[1]).T[fpc_index]
fpc_eigVectors = np.asmatrix(fpc_eigVectors).T
#(e)
d_centralize = np.asmatrix(d) - np.asmatrix(d_mean).T
fpc = d_centralize.T * fpc_eigVectors





