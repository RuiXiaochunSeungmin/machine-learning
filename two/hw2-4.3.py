import pandas as pd
import numpy as np
import scipy.io
import scipy as sp
from sklearn.model_selection import train_test_split
from numpy import linalg as LA
import random

# Data load and process
mat= scipy.io.loadmat('hw1data.mat')
df = pd.DataFrame(mat['X'], mat['Y'])
train, test = train_test_split(df, test_size=0.2, random_state=42)
train_r = train.reset_index()
test_r = test.reset_index()
train_r_m = train_r.as_matrix()
test_r_m = test_r.as_matrix()
train_y = train_r_m[:,0]
test_y = test_r_m[:,0]
train_x = train_r_m[:,1:]
test_x = test_r_m[:,1:]

# Kernel perceptron
def kernel_perceptron(T,X_train,Y_train):
    n = len(Y_train)
    Alpha = np.zeros((10,n))
    Y = []
    #ten binary classifiers
    for k in range(10):
        print k
        alpha = np.zeros(n)
        y_train = 2*(Y_train==k) - np.ones(n)
        Y.append(y_train)
        x_train = X_train
    
        t = 1
        while t<=T:
            #print t
            index = t%(n)
            x_i = x_train[index,:]
            y_i = y_train[index]
            ker_dot =(np.dot(x_train,x_i)+2*np.ones(n))**10
            if y_i*sum(alpha*(y_train*ker_dot))<=0:
                alpha[index]+=1
            t = t+1
        Alpha[k,:] = alpha
    return Alpha,Y
# Predict
T_list = [500, 1000, 2000]
testE_list = []
n = len(train_y)
for T in T_list:
    print T
    Alpha,Y = kernel__perceptron(T,train_x,train_y)
    wrong = 0
    for j in range(len(test_x)):
        print j
        y_j = test_y[j]
        x_j = test_x[j,:]
        prob = np.zeros(10)
        for k in range(10):
            train_y = Y[k]
            ker_dot =(np.dot(train_x,x_j)+2*np.ones(n))**10
            prob[k] = sum(Alpha[k,:]*(train_y*ker_dot))
        index = np.argmax(prob)
        if y_j!=index:
            wrong+=1
    test_error = wrong*1.0/len(test_x)
    testE_list.append(test_error)
    print testE_list
