from scipy import io
from scipy.stats import mode
from heapq import nsmallest
import numpy as np
from matplotlib import pyplot as plt

Matrix = io.loadmat('C:\Users\lhren\Desktop\Fall 2017/4771\hw\hw1data.mat')
X_Y = np.concatenate((Matrix['X'],Matrix['Y']),axis=1)
n = len(X_Y)
d = 784
count = np.zeros(10)
sum_element = np.zeros(10)
for data in X_Y:
    i = int(data[d])
    count[i]+=1
    for j in range(784):
        sum_element[i]+=data[j]
def norm_f(v):
    return sum(v*v)
theta = np.zeros(10)
delta_f = 784*theta*count - sum_element
iterations=0
while (norm_f(delta_f)>=1e-16) :
        n = 0.000001
        theta = theta - n*delta_f
        delta_f = 784*theta*count - sum_element
        print delta_f
        iterations+=1
print('# Iterations:' + str(iterations))
print('Minimum of f appears at x='+str(theta))
print('First derivative at minimum point: '+str(delta_f))

    