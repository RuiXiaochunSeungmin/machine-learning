from scipy import io
from scipy.stats import mode
from heapq import nsmallest
import numpy as np
from matplotlib import pyplot as plt

Matrix = io.loadmat('C:\Users\lhren\Desktop\Fall 2017/4771\hw\hw1data.mat')
X_Y = np.concatenate((Matrix['X'],Matrix['Y']),axis=1)
np.random.shuffle(X_Y)
train_data = X_Y[0:8000,:]
test_data = X_Y[8000:10000,:]
train_X = train_data[:,0:784]
train_Y = train_data[:,784]
test_X = test_data[:,0:784]
test_Y = test_data[:,784]
X = train_data
n = len(X)
k = int(np.sqrt(n))
d = len(X[0]) - 1
count_digit = np.zeros(10)
sum_digit = np.zeros((10,784))
for x in train_data:
    i = int(x[d])
    count_digit[i]+=1
    sum_digit[i,:]+=x[:d]
mu_mle = np.zeros((10,784))
P_y = 1.0*count_digit / n
print P_y
for i in range(10):
    mu_mle[i,:] = 1.0*sum_digit[i,:]/count_digit[i]
#print mu_mle
#Find significant indices
index_sig = []
Mu = []
for i in range(10):
    index = mu_mle[i]>=50
    Mu.append(mu_mle[i][index])
    index_sig.append(index)
#print Mu
sigma_mle = []
for i in range(10):
    sigma_mle.append(np.zeros((len(Mu[i]),len(Mu[i]))))
for x in train_data:
    i = int(x[d])
    mu = Mu[i]
    a = np.matrix(x[:d][index_sig[i]] - mu)
    a_t = a.transpose()
    sigma_mle[i] += 1.0*(np.dot(a_t,a))/count_digit[i]

#print np.linalg.det(sigma_mle[1])
count_correct = 0
for x in test_data:
    x_test = x[:d]
    y_test = int(x[d])
    P = P_y
    
    for i in range(10):
        mu = np.array(Mu[i])
        index = index_sig[i]
        x_t = x_test[index]
        D = len(x_t)
        v = x_t - mu
        #print len(c)
        sigma = np.array(sigma_mle[i])
        
        #print a
        #print np.linalg.det(a)
         #print v
        P[i] = P[i]*( (2*np.pi)**D*np.linalg.det(sigma))**(-0.5)
        P[i] = P[i]*np.exp(-0.5*np.dot(v,np.dot(np.linalg.inv(sigma),v.transpose())))
    #print P  
    label = int(np.argmax(P))
    if label==y_test:
        count_correct+=1
print count_correct 