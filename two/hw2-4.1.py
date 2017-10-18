# -*- coding: utf-8 -*-

from scipy import io
import numpy as np
from matplotlib import pyplot as plt
#get_ipython().magic(u'matplotlib inline')

Matrix = io.loadmat('C:\Users\lhren\Desktop\Fall 2017/4771\hw\hw1data.mat')
X_Y = np.concatenate((Matrix['X'],Matrix['Y']),axis=1)
np.random.shuffle(X_Y)
X_Y = np.array(X_Y)
N = len(X_Y)
d = 784
s = 8000
X_train = X_Y[0:s,0:d]
Y_train = X_Y[0:s,d]
X_test = X_Y[s:N,0:d]
Y_test = X_Y[s:N,d]

n = len(X_train)


def V0(T,X_train,Y_train):
    n = len(Y_train)
    W = np.zeros((10,d))
    B = np.zeros(10)
    #ten binary classifiers
    for k in range(10):
        w = np.zeros(d)
        b = 0
        
        x_train = X_train
        y_train = 2*(Y_train==k) - np.ones(n)
    
        t = 1
        while t<=T:
            index = t%(n)
            y_i = y_train[index]
            x_i = x_train[index,:]
            if y_i*(sum(w*x_i)+b)<=0:
                w = w+y_i*x_i
                b = b+y_i
            t = t+1
        W[k,:] = w
        B[k] = b
    return W,B


# V0 perceptron test


T_list = [500,1000,1500,2000,3000,4000,5000,6000,7000,8000,9000,10000]
#trainE_list = []
testE_list = []
for T in T_list:
    #V0
    print T
    #train_error = V0(T,X_train,Y_train,X_train,Y_train)
    #trainE_list.append(train_error)
    W,B = V0(T,X_train,Y_train)
    wrong = 0
    for j in range(len(X_test)):
        y_j = Y_test[j]
        x_j = X_test[j,:]
        prob = np.zeros(10)
        for i in range(10):
            prob[i] = sum(W[i,:]*x_j)+B[i]
        index = np.argmax(prob)
        if y_j!=index:
            wrong+=1
    test_error = wrong*1.0/len(X_test)
    testE_list.append(test_error)
#plt.plot(np.array(T_list),np.array(trainE_list),color='b')
plt.figure()
plt.plot(np.array(T_list),np.array(testE_list),color='r')
plt.show()

def V1(T,X_train,Y_train):
    n = len(Y_train)
    W = np.zeros((10,d))
    B = np.zeros(10)
    #ten binary classifiers
    for k in range(10):
        print k
        w = np.zeros(d)
        b = 0
       
        x_train = X_train
        y_train = 2*(Y_train==k)-np.ones(n)
    
        t = 1
        while t<=T:
            #print t
            a = (np.dot(x_train,w)+b*np.ones(n))*y_train
            index = np.argmin(a)
            y_i = y_train[index]
            x_i = x_train[index,:]
            if y_i*(sum(w*x_i)+b)<=0:
                w = w+y_i*x_i
                b = b+y_i
            t = t+1
        W[k,:] = w
        B[k] = b
    return W,B
   

    


# V1 perceptron test
T_list = [1000,2000,3000,4000]#,2500,3000]
#trainE_list = []
testE_list = []
for T in T_list:
    #V1
    print T
    #train_error = V1(T,X_train,Y_train,X_train,Y_train)
    #trainE_list.append(train_error)
    W,B = V1(T,X_train,Y_train)
    wrong = 0
    for j in range(len(X_test)):
        y_j = Y_test[j]
        x_j = X_test[j,:]
        prob = np.zeros(10)
        for i in range(10):
            prob[i] = sum(W[i,:]*x_j)+B[i]
        index = np.argmax(prob)
        if y_j!=index:
            wrong+=1
    test_error = wrong*1.0/len(X_test)
    testE_list.append(test_error)
#plt.plot(np.array(T_list),np.array(trainE_list),color='b')
plt.figure()
plt.plot(np.array(T_list),np.array(testE_list),color='r')
plt.show()

def V2(T,x_train,y_train):
    W = []
    B = []
    C = []
    #V2
    
    n = len(Y_train)
    #ten binary classifiers
    for k in range(10):
        print k
        w = []
        b = []
        c = []
        x_train = X_train
        y_train = 2*(Y_train==k)-np.ones(n)
    
        t = 1
        w_t = np.zeros(784)
        b_t=0
        c_t = 1
        while t<=T:
            index = t%(n)
            y_i = y_train[index]
            x_i = x_train[index,:]
            if y_i*sum(w_t*x_i+b_t)<=0:
                w.append(list(w_t))
                b.append(b_t)
                c.append(c_t)
                w_t += y_i*x_i
                b_t += y_i
                c_t = 1
            else: 
                c_t += 1
            t = t+1
        W.append(w)
        B.append(b)
        C.append(c)
    return W,B,C


# V2 perceptron test

T_list = [1000,2000,3000]
#trainE_list = []
testE_list = []
for T in T_list:
    #V1
    print T
    #train_error = V1(T,X_train,Y_train,X_train,Y_train)
    #trainE_list.append(train_error)
    W,B,C = V2(T,X_train,Y_train)
    wrong = 0
    for j in range(len(X_test)):
        y_j = Y_test[j]
        x_j = X_test[j,:]
        prob = np.zeros(10)
        for i in range(10):
            w = W[i]
            b = B[i]
            c = C[i]
            prob[i] = sum(c*(np.dot(w,x_j)+b))
        index = np.argmax(prob)
        if y_j!=index:
            wrong+=1
    test_error = wrong*1.0/len(X_test)
    testE_list.append(test_error)
#plt.plot(np.array(T_list),np.array(trainE_list),color='b')
plt.figure()
plt.plot(np.array(T_list),np.array(testE_list),color='r')
plt.show()
#ignore the below part
"""
def kernel_V0(T,X_train,Y_train):
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


# In[69]:

T_list = [1000]#,3000,4000,5000,6000,7000,8000,9000,10000]
#trainE_list = []
testE_list = []
trainE_list = []
n = len(Y_train)
for T in T_list:
    #V0
    print T
    #train_error = V0(T,X_train,Y_train,X_train,Y_train)
    #trainE_list.append(train_error)
    Alpha,Y = kernel_V0(T,X_train,Y_train)
    wrong = 0
    for j in range(len(X_test)):
        print j
        y_j = Y_test[j]
        x_j = X_test[j,:]
        prob = np.zeros(10)
        for k in range(10):
            y_train = Y[k]
            ker_dot =(np.dot(X_train,x_j)+2*np.ones(n))**10
            prob[k] = sum(Alpha[k,:]*(y_train*ker_dot))
        index = np.argmax(prob)
        if y_j!=index:
            wrong+=1
    test_error = wrong*1.0/len(X_test)
    testE_list.append(test_error)
    wrong = 0
    for j in range(len(X_train)):
        print j
        y_j = Y_train[j]
        x_j = X_train[j,:]
        prob = np.zeros(10)
        for k in range(10):
            y_train = Y[k]
            ker_dot =(np.dot(X_train,x_j)+2*np.ones(n))**10
            prob[k] = sum(Alpha[k,:]*(y_train*ker_dot))
        index = np.argmax(prob)
        if y_j!=index:
            wrong+=1
    train_error = wrong*1.0/len(X_train)
    trainE_list.append(train_error)
    
    print wrong
#plt.plot(np.array(T_list),np.array(trainE_list),color='b')
#plt.plot(np.array(T_list),np.array(testE_list),color='ro')

"""



