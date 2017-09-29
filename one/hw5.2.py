
# coding: utf-8

# In[1]:

from scipy import io
from scipy.stats import mode
from heapq import nsmallest
import numpy as np
from matplotlib import pyplot as plt
Matrix = io.loadmat('C:\Users\lhren\Desktop\Fall 2017/4771\hw\hw1data.mat')
X_Y = np.concatenate((Matrix['X'],Matrix['Y']),axis=1)


# In[160]:

##
np.random.shuffle(X_Y)
N = len(X_Y)
train_data = X_Y[0:9000,:]
test_data = X_Y[9000:10000,:]
train_X = train_data[:,0:784]
train_Y = train_data[:,784]
test_X = test_data[:,0:784]
test_Y = test_data[:,784]
X = train_data
    ##construct kd tree
n = len(X)
k = 5
d = len(X[0]) - 1
dist = [[],[]]
for data in X:
    dist[1].append(data[d])
    x = data[:d]
    dist[0].append(sum(x**2))
#print dist


# In[165]:

count=0
correct = 0
def knn_predict_L1(test):
    t_x = test[:d]
    t_X = np.array(np.ones((len(train_X),d))*t_x)
    t_y = test[d]
    adj = np.matrix(abs(train_X-t_X)).sum(axis=1)
    #print adj.sum(axis=1)
    adj = np.array(np.transpose(adj))[0]
    k_neighbor = np.array(nsmallest(20,adj))
    knn_class=[]
    #print label
    
    for nn in k_neighbor:
        index = list(adj).index(nn)
        #print index
        knn_class.append(train_Y[int(index)])
    
    #print knn_class
    most_frequent = int(mode(knn_class)[0][0])
    return most_frequent
def knn_predict_Linf(test):
    t_x = test[:d]
    t_X = np.array(np.ones((len(train_X),d))*t_x)
    t_y = test[d]
    adj = np.matrix(abs(train_X-t_X)).max(axis=1)
    #print adj.sum(axis=1)
    adj = np.array(np.transpose(adj))[0]
    k_neighbor = np.array(nsmallest(20,adj))
    knn_class=[]
    #print label
    
    for nn in k_neighbor:
        index = list(adj).index(nn)
        #print index
        knn_class.append(train_Y[int(index)])
    
    #print knn_class
    most_frequent = int(mode(knn_class)[0][0])
    return most_frequent
def knn_predict_L2(test):
    t_x = test[:d]
    t_y = test[d]
    x_d = np.array(dist[0]).copy()
    adj = np.matrix(train_X*t_x)
    #print adj.sum(axis=1)
    x_d  = x_d-2*np.transpose(adj.sum(axis=1))
    x_d = np.array(x_d)[0]
    k_neighbor = np.array(nsmallest(20,x_d))
    knn_class=[]
    #print label
    
    for nn in k_neighbor:
        index = list(x_d).index(nn)
        #print index
        knn_class.append(dist[1][int(index)])
    
    #print knn_class
    most_frequent = int(mode(knn_class)[0][0])
    return most_frequent

for test in test_data:
    #y_hat1 = knn_predict_L1(test)
    y_hat2 = knn_predict_L2(test)
    #y_hatinf = knn_predict_Linf(test)
    #print y_hat
    y = test[d]
    if(y==y_hatinf):
        correct+=1
    
    count+=1
    print count
print 1.0*correct/count


# In[ ]:



