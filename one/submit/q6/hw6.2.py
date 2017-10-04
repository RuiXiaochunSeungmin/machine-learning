
# Rui Ding | rd2622
# Xiaochun Ma | xm2203
# Seungmin Lee | sl3254

# -*- coding: utf-8 -*-

from scipy import io
from scipy.stats import mode

import numpy as np
from matplotlib import pyplot as plt
#load data
Matrix = io.loadmat('C:\Users\lhren\Desktop\Fall 2017/4771\hw\hw1data.mat')
X_Y = np.concatenate((Matrix['X'],Matrix['Y']),axis=1)
n = len(X_Y)
d = 784
count = np.zeros(10)
#prestore useful information so afterward calculation of gradient/f values are extremely convenient
sum_element = np.zeros(10)
sum_sq_element = np.zeros(10)
for data in X_Y:
    i = int(data[d])
    count[i]+=1
    sum_element[i]+=sum(data[:d])
    for j in range(784):    
        sum_sq_element[i]+=data[j]**2
#print sum_element, count
def norm_f(v):
    return sum(v*v)
def eval_f(theta):
    f = 0
    #squaring out the expression and use prestored data
    f+=sum(sum_sq_element) - 2*sum(sum_element*theta) + 784*sum(count*theta**2)
    f = f*0.5
    return f

#tests iterative algorithm on different step size(eta) input
it_max = 100000
step_list = [1e-6,1e-7,1e-8,1e-9]
#to test individual eta run results, use e.g [1e-7] in the place of the step_list
for n in step_list:
    theta = np.zeros(10)
    delta_f = 784*theta*count-sum_element
    f_values = []
    iterations=0
    while (norm_f(delta_f)>1e-8) and (iterations<it_max) :

            theta = theta - n*delta_f
            delta_f = 784*theta*count-sum_element
            f = eval_f(theta)
            f_values.append(f)
            #print delta_f
            iterations+=1

    print('# Iterations:' + str(iterations))
    print('Minimum of f appears at theta='+str(theta))
    print('First derivative at minimum point: '+str(delta_f))
    print('f = '+str(eval_f(theta)))
    plt.figure()
    plt.plot(range(iterations),f_values,'ro')
    plt.title('step size '+str(n))
    plt.show()

#comparison studies for different time step sizes
n = np.array([1e-6,1e-7,1e-8,1e-9])
iteration = np.array([22,358,3692,37039])
plt.figure()
plt.loglog(n,iteration,'ro-')
plt.xlabel('step size')
plt.ylabel('iterations')
plt.show()






