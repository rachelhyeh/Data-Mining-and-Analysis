# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 19:25:10 2018

@author: rache
"""

#ECEN758-DMA-Assignment 4

import numpy as np
from matplotlib import pyplot as plt

#Problem 3
def kz(x, xi, h):
    z = (x - xi)/h
    k = 0
    if abs(z) <= 1/2:
        k = 1
    else:
        k = 0
    return k

h = 3
pts = [1, 5, 6, 9, 15]
data = np.arange(0,20, 0.1)
f = list()
    
for i in range(len(data)):
    x = data[i]
    k_sum = 0
    for j in range(len(pts)):
        k_sum = k_sum + kz(x, pts[j], h)
    f.append( k_sum/(len(pts)*h) )
    
plt.plot(data, f)
plt.title('Discrete kernel density function with h = 3')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.show()